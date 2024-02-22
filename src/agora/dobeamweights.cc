/**
 * @file dobeamweights.cc
 * @brief Implementation file for the DoBeamWeights class.  Calculates Precoder/Detector  for one
 * subcarrier.
 */
#include "dobeamweights.h"

#include "comms-lib.h"
#include "concurrent_queue_wrapper.h"
#include "doer.h"
#include "logger.h"

static constexpr bool kUseSIMDGather = true;
// Calculate the zeroforcing receiver using the formula W_zf = inv(H' * H) * H'.
// This is faster but less accurate than using an SVD-based pseudoinverse.
static constexpr bool kUseInverseForZF = true;
static constexpr bool kUseUlZfForDownlink = true;

DoBeamWeights::DoBeamWeights(
    Config* config, int tid,
    PtrGrid<kFrameWnd, kMaxUEs, complex_float>& csi_buffers,
    Table<complex_float>& calib_dl_buffer,
    Table<complex_float>& calib_ul_buffer,
    Table<complex_float>& calib_dl_msum_buffer,
    Table<complex_float>& calib_ul_msum_buffer,
    Table<complex_float>& calib_buffer,
    PtrGrid<kFrameWnd, kMaxDataSCs, complex_float>& ul_beam_matrices,
    PtrGrid<kFrameWnd, kMaxDataSCs, complex_float>& dl_beam_matrices,
    MacScheduler* mac_sched, PhyStats* in_phy_stats, Stats* stats_manager)
    : Doer(config, tid),
      csi_buffers_(csi_buffers),
      calib_dl_buffer_(calib_dl_buffer),
      calib_ul_buffer_(calib_ul_buffer),
      calib_dl_msum_buffer_(calib_dl_msum_buffer),
      calib_ul_msum_buffer_(calib_ul_msum_buffer),
      calib_buffer_(calib_buffer),
      ul_beam_matrices_(ul_beam_matrices),
      dl_beam_matrices_(dl_beam_matrices),
      mac_sched_(mac_sched),
      phy_stats_(in_phy_stats) {
  duration_stat_ = stats_manager->GetDurationStat(DoerType::kBeam, tid);
  pred_csi_buffer_ =
      static_cast<complex_float*>(Agora_memory::PaddedAlignedAlloc(
          Agora_memory::Alignment_t::kAlign64,
          kMaxAntennas * kMaxUEs * sizeof(complex_float)));
  csi_gather_buffer_ =
      static_cast<complex_float*>(Agora_memory::PaddedAlignedAlloc(
          Agora_memory::Alignment_t::kAlign64,
          kMaxAntennas * kMaxUEs * sizeof(complex_float)));
  calib_gather_buffer_ = static_cast<complex_float*>(
      Agora_memory::PaddedAlignedAlloc(Agora_memory::Alignment_t::kAlign64,
                                       kMaxAntennas * sizeof(complex_float)));

  calib_sc_vec_ptr_ = std::make_unique<arma::cx_fvec>(
      reinterpret_cast<arma::cx_float*>(calib_gather_buffer_), cfg_->BfAntNum(),
      false);

  //Init to identity
  calib_sc_vec_ptr_->fill(arma::cx_float(1.0f, 0.0f));

  num_ext_ref_ = 0;
  for (size_t i = 0; i < cfg_->NumCells(); i++) {
    if (cfg_->ExternalRefNode(i)) {
      num_ext_ref_++;
    }
  }
  if (num_ext_ref_ > 0) {
    ext_ref_id_.zeros(num_ext_ref_ * cfg_->NumChannels());
    size_t ext_id = 0;
    for (size_t i = 0; i < cfg_->NumCells(); i++) {
      if (cfg_->ExternalRefNode(i)) {
        for (size_t j = 0; j < cfg_->NumChannels(); j++) {
          ext_ref_id_.at(ext_id * cfg_->NumChannels() + j) =
              (cfg_->RefRadio(i) * cfg_->NumChannels()) + j;
        }
        ext_id++;
      }
    }
  }
}

DoBeamWeights::~DoBeamWeights() {
  std::free(pred_csi_buffer_);
  std::free(csi_gather_buffer_);
  calib_sc_vec_ptr_.reset();
  std::free(calib_gather_buffer_);
}

EventData DoBeamWeights::Launch(size_t tag) {
  ComputeBeams(tag);
  return EventData(EventType::kBeam, tag);
}

void DoBeamWeights::ComputePrecoder(size_t frame_id, size_t cur_sc_id,
                                    const arma::cx_fmat& mat_csi,
                                    const arma::cx_fvec& calib_sc_vec,
                                    const float noise,
                                    complex_float* ul_beam_mem,
                                    complex_float* dl_beam_mem) {
#if !defined(TIME_EXCLUSIVE)
  if (kEnableMatLog) {
    phy_stats_->UpdateUlCsi(frame_id, cur_sc_id, mat_csi);
  }
#endif
  arma::cx_fmat mat_ul_beam(reinterpret_cast<arma::cx_float*>(ul_beam_mem),
                            cfg_->SpatialStreamsNum(), cfg_->BsAntNum(), false);
  arma::cx_fmat mat_ul_beam_tmp;
  switch (cfg_->BeamformingAlgo()) {
    case CommsLib::BeamformingAlgorithm::kZF:
      if (kUseInverseForZF) {
        try {
          mat_ul_beam_tmp =
              arma::inv_sympd(mat_csi.t() * mat_csi) * mat_csi.t();
        } catch (std::runtime_error&) {
          AGORA_LOG_WARN(
              "Failed to invert channel matrix, falling back to pinv()\n");
          arma::pinv(mat_ul_beam_tmp, mat_csi, 1e-2, "dc");
        }
      } else {
        arma::pinv(mat_ul_beam_tmp, mat_csi, 1e-2, "dc");
      }
      break;
    case CommsLib::BeamformingAlgorithm::kMMSE:
      mat_ul_beam_tmp =
          arma::inv_sympd(
              mat_csi.t() * mat_csi +
              noise * arma::eye<arma::cx_fmat>(cfg_->SpatialStreamsNum(),
                                               cfg_->SpatialStreamsNum())) *
          mat_csi.t();
      break;
    case CommsLib::BeamformingAlgorithm::kMRC:
      mat_ul_beam_tmp = mat_csi.t();
      break;
    default:
      AGORA_LOG_ERROR("Beamforming algorithm is not implemented!");
  }

  if (cfg_->Frame().NumDLSyms() > 0) {
    arma::cx_fmat mat_dl_beam_tmp;
    if (kUseUlZfForDownlink == true) {
      // With orthonormal calib matrix:
      // pinv(calib * csi) = pinv(csi)*inv(calib)
      // This probably causes a performance hit since we are throwing
      // magnitude info away by taking the sign of the calibration matrix
      // Inv is already acheived by UL over DL division outside this function
      arma::cx_fmat inv_calib_mat = arma::diagmat(arma::sign(calib_sc_vec));
      mat_dl_beam_tmp = mat_ul_beam_tmp * inv_calib_mat;
    } else {
      arma::cx_fmat mat_dl_csi = inv(arma::diagmat(calib_sc_vec)) * mat_csi;
      if (kEnableMatLog) {
        phy_stats_->UpdateDlCsi(frame_id, cur_sc_id, mat_dl_csi);
      }
      switch (cfg_->BeamformingAlgo()) {
        case CommsLib::BeamformingAlgorithm::kZF:
          if (kUseInverseForZF) {
            try {
              mat_dl_beam_tmp =
                  arma::inv_sympd(mat_dl_csi.t() * mat_dl_csi) * mat_dl_csi.t();
            } catch (std::runtime_error&) {
              AGORA_LOG_WARN(
                  "Failed to invert channel matrix, falling back to pinv()\n");
              arma::pinv(mat_dl_beam_tmp, mat_csi, 1e-2, "dc");
            }
          } else {
            arma::pinv(mat_dl_beam_tmp, mat_csi, 1e-2, "dc");
          }
          break;
        case CommsLib::BeamformingAlgorithm::kMMSE:
          mat_dl_beam_tmp =
              arma::inv_sympd(
                  mat_dl_csi.t() * mat_dl_csi +
                  noise * arma::eye<arma::cx_fmat>(cfg_->SpatialStreamsNum(),
                                                   cfg_->SpatialStreamsNum())) *
              mat_dl_csi.t();
          break;
        case CommsLib::BeamformingAlgorithm::kMRC:
          mat_dl_beam_tmp = mat_dl_csi.t();
          break;
        default:
          AGORA_LOG_ERROR("Beamforming algorithm is not implemented!");
      }
    }
    // We should be scaling the beamforming matrix, so the IFFT
    // output can be scaled with OfdmCaNum() across all antennas.
    // See Argos paper (Mobicom 2012) Sec. 3.4 for details.
    const float scale = 1 / (abs(mat_dl_beam_tmp).max());
    mat_dl_beam_tmp = mat_dl_beam_tmp * scale;

    for (size_t i = 0; i < cfg_->NumCells(); i++) {
      if (cfg_->ExternalRefNode(i)) {
        // Zero out all antennas on the reference radio
        mat_dl_beam_tmp.insert_cols(
            (cfg_->RefRadio(i) * cfg_->NumChannels()),
            arma::cx_fmat(cfg_->SpatialStreamsNum(), cfg_->NumChannels(),
                          arma::fill::zeros));
      }
    }
    arma::cx_fmat mat_dl_beam(reinterpret_cast<arma::cx_float*>(dl_beam_mem),
                              cfg_->BsAntNum(), cfg_->SpatialStreamsNum(),
                              false);
    mat_dl_beam = mat_dl_beam_tmp.st();
    if (kEnableMatLog) {
      phy_stats_->UpdateDlBeam(frame_id, cur_sc_id, mat_dl_beam);
    }
  }

  for (int i = (int)cfg_->NumCells() - 1; i >= 0; i--) {
    if (cfg_->ExternalRefNode(i) == true) {
      mat_ul_beam_tmp.insert_cols(
          (cfg_->RefRadio(i) * cfg_->NumChannels()),
          arma::cx_fmat(cfg_->SpatialStreamsNum(), cfg_->NumChannels(),
                        arma::fill::zeros));
    }
  }
  mat_ul_beam = mat_ul_beam_tmp;

#if !defined(TIME_EXCLUSIVE)
  if (kEnableMatLog) {
    phy_stats_->UpdateUlBeam(frame_id, cur_sc_id, mat_ul_beam.st());
  }
  if (kPrintBeamStats) {
    const float rcond = arma::rcond(mat_csi.t() * mat_csi);
    phy_stats_->UpdateCsiCond(frame_id, cur_sc_id, rcond);
  }
#endif
}

// Called for each frame_id / sc_id
// Updates calib_sc_vec
void DoBeamWeights::ComputeCalib(size_t frame_id, size_t sc_id,
                                 arma::cx_fvec& calib_sc_vec) {
  const size_t frames_to_complete = cfg_->RecipCalFrameCnt();
  if (cfg_->Frame().IsRecCalEnabled() && (frame_id >= frames_to_complete)) {
    const size_t cal_slot_current = cfg_->RecipCalIndex(frame_id);
    const bool frame_update = ((frame_id % frames_to_complete) == 0);

    // Use the previous window which has a full set of calibration results
    const size_t cal_slot_complete =
        cfg_->ModifyRecCalIndex(cal_slot_current, -1);

    // update moving sum
    arma::cx_fmat cur_calib_dl_msum_mat(
        reinterpret_cast<arma::cx_float*>(
            calib_dl_msum_buffer_[cal_slot_complete]),
        cfg_->BfAntNum(), cfg_->OfdmDataNum(), false);
    arma::cx_fmat cur_calib_ul_msum_mat(
        reinterpret_cast<arma::cx_float*>(
            calib_ul_msum_buffer_[cal_slot_complete]),
        cfg_->BfAntNum(), cfg_->OfdmDataNum(), false);

    arma::cx_fmat calib_mat(
        reinterpret_cast<arma::cx_float*>(calib_buffer_[cal_slot_complete]),
        cfg_->BfAntNum(), cfg_->OfdmDataNum(), false);

    // Update the moving sum
    if (frame_update) {
      if (sc_id == 0) {
        AGORA_LOG_TRACE(
            "DoBeamWeights[%d]: (Frame %zu, sc_id %zu), ComputeCalib "
            "updating "
            "calib at slot %zu : prev %zu, old %zu\n",
            tid_, frame_id, sc_id, cal_slot_complete, cal_slot_prev,
            cal_slot_old);
      }
      // Add the most recently completed value
      const arma::cx_fmat cur_calib_dl_mat(
          reinterpret_cast<arma::cx_float*>(
              calib_dl_buffer_[cal_slot_complete]),
          cfg_->OfdmDataNum(), cfg_->BfAntNum(), false);
      const arma::cx_fmat cur_calib_ul_mat(
          reinterpret_cast<arma::cx_float*>(
              calib_ul_buffer_[cal_slot_complete]),
          cfg_->OfdmDataNum(), cfg_->BfAntNum(), false);

      if (cfg_->SmoothCalib()) {
        // oldest frame data in buffer but could be partially written with newest values
        // using the second oldest....
        const size_t cal_slot_old =
            cfg_->ModifyRecCalIndex(cal_slot_current, +1);

        const arma::cx_fmat old_calib_dl_mat(
            reinterpret_cast<arma::cx_float*>(calib_dl_buffer_[cal_slot_old]),
            cfg_->OfdmDataNum(), cfg_->BfAntNum(), false);
        const arma::cx_fmat old_calib_ul_mat(
            reinterpret_cast<arma::cx_float*>(calib_ul_buffer_[cal_slot_old]),
            cfg_->OfdmDataNum(), cfg_->BfAntNum(), false);

        const size_t cal_slot_prev =
            cfg_->ModifyRecCalIndex(cal_slot_complete, -1);
        const arma::cx_fmat prev_calib_dl_msum_mat(
            reinterpret_cast<arma::cx_float*>(
                calib_dl_msum_buffer_[cal_slot_prev]),
            cfg_->BfAntNum(), cfg_->OfdmDataNum(), false);
        const arma::cx_fmat prev_calib_ul_msum_mat(
            reinterpret_cast<arma::cx_float*>(
                calib_ul_msum_buffer_[cal_slot_prev]),
            cfg_->BfAntNum(), cfg_->OfdmDataNum(), false);

        // Add new value to old rolling sum.  Then subtract out the oldest.

        cur_calib_dl_msum_mat.col(sc_id) =
            prev_calib_dl_msum_mat.col(sc_id) +
            (cur_calib_dl_mat.row(sc_id) - old_calib_dl_mat.row(sc_id)).st();
        cur_calib_ul_msum_mat.col(sc_id) =
            prev_calib_ul_msum_mat.col(sc_id) +
            (cur_calib_ul_mat.row(sc_id) - old_calib_ul_mat.row(sc_id)).st();
        calib_mat.col(sc_id) =
            cur_calib_ul_msum_mat.col(sc_id) / cur_calib_dl_msum_mat.col(sc_id);
      } else {
        calib_mat.col(sc_id) =
            (cur_calib_ul_mat.row(sc_id) / cur_calib_dl_mat.row(sc_id)).st();
      }
      if (kEnableMatLog) {
        phy_stats_->UpdateCalibMat(frame_id, sc_id, calib_mat.col(sc_id));
      }
    }
    calib_sc_vec = calib_mat.col(sc_id);
  }
  // Otherwise calib_sc_vec = identity from init
}

// Gather data of one symbol from partially-transposed buffer
// produced by dofft
static inline void PartialTransposeGather(size_t cur_sc_id, float* src,
                                          float*& dst, size_t bs_ant_num) {
  // The SIMD and non-SIMD methods are equivalent.

#ifdef __AVX512F__
  static constexpr size_t kAntNumPerSimd = 8;
#else
  static constexpr size_t kAntNumPerSimd = 4;
#endif

  size_t ant_start = 0;
  if (kUseSIMDGather && (bs_ant_num >= kAntNumPerSimd)) {
    const size_t transpose_block_id = cur_sc_id / kTransposeBlockSize;
    const size_t sc_inblock_idx = cur_sc_id % kTransposeBlockSize;
    const size_t offset_in_src_buffer =
        transpose_block_id * bs_ant_num * kTransposeBlockSize + sc_inblock_idx;

    src = src + offset_in_src_buffer * 2;
#ifdef __AVX512F__
    __m512i index = _mm512_setr_epi32(
        0, 1, kTransposeBlockSize * 2, kTransposeBlockSize * 2 + 1,
        kTransposeBlockSize * 4, kTransposeBlockSize * 4 + 1,
        kTransposeBlockSize * 6, kTransposeBlockSize * 6 + 1,
        kTransposeBlockSize * 8, kTransposeBlockSize * 8 + 1,
        kTransposeBlockSize * 10, kTransposeBlockSize * 10 + 1,
        kTransposeBlockSize * 12, kTransposeBlockSize * 12 + 1,
        kTransposeBlockSize * 14, kTransposeBlockSize * 14 + 1);
    for (size_t ant_idx = 0; ant_idx < bs_ant_num; ant_idx += kAntNumPerSimd) {
      // fetch 4 complex floats for 4 ants
      __m512 t = (kTransposeBlockSize == 1)
                     ? _mm512_load_ps(src)
                     : _mm512_i32gather_ps(index, src, 4);
      _mm512_storeu_ps(dst, t);
      src += kAntNumPerSimd * kTransposeBlockSize * 2;
      dst += kAntNumPerSimd * 2;
    }
#else
    __m256i index = _mm256_setr_epi32(
        0, 1, kTransposeBlockSize * 2, kTransposeBlockSize * 2 + 1,
        kTransposeBlockSize * 4, kTransposeBlockSize * 4 + 1,
        kTransposeBlockSize * 6, kTransposeBlockSize * 6 + 1);
    for (size_t ant_idx = 0; ant_idx < bs_ant_num; ant_idx += kAntNumPerSimd) {
      // fetch 4 complex floats for 4 ants
      __m256 t = _mm256_i32gather_ps(src, index, 4);
      _mm256_storeu_ps(dst, t);
      src += kAntNumPerSimd * kTransposeBlockSize * 2;
      dst += kAntNumPerSimd * 2;
    }
#endif
    // Set the of the remaining antennas to use non-SIMD gather
    ant_start = bs_ant_num - (bs_ant_num % kAntNumPerSimd);
  }
  if (ant_start < bs_ant_num) {
    const size_t pt_base_offset =
        (cur_sc_id / kTransposeBlockSize) * (kTransposeBlockSize * bs_ant_num);
    auto* cx_src = reinterpret_cast<complex_float*>(src);
    complex_float* cx_dst = (complex_float*)dst + ant_start;
    for (size_t ant_i = ant_start; ant_i < bs_ant_num; ant_i++) {
      *cx_dst = cx_src[pt_base_offset + (ant_i * kTransposeBlockSize) +
                       (cur_sc_id % kTransposeBlockSize)];
      cx_dst++;
    }
  }
}

// Gather data of one symbol from partially-transposed buffer
// produced by dofft
static inline void TransposeGather(size_t cur_sc_id, float* src, float*& dst,
                                   size_t bs_ant_num, size_t ofdm_data_num) {
  auto* cx_src = reinterpret_cast<complex_float*>(src);
  auto* cx_dst = reinterpret_cast<complex_float*>(dst);
  for (size_t ant_i = 0; ant_i < bs_ant_num; ant_i++) {
    *cx_dst = cx_src[ant_i * ofdm_data_num + cur_sc_id];
    cx_dst++;
  }
}

void DoBeamWeights::ComputeBeams(size_t tag) {
  //Request was generated from gen_tag_t::FrmSc
  const size_t frame_id = gen_tag_t(tag).frame_id_;
  const size_t base_sc_id = gen_tag_t(tag).sc_id_;
  const size_t frame_slot = frame_id % kFrameWnd;
  if (kDebugPrintInTask) {
    std::printf("In doZF thread %d: frame: %zu, base subcarrier: %zu\n", tid_,
                frame_id, base_sc_id);
  }

  // Process BeamBlockSize (or less) number of carriers
  // cfg_->OfdmDataNum() is the total number of usable subcarriers
  // First sc in the next block
  const size_t last_sc_id =
      base_sc_id +
      std::min(cfg_->BeamBlockSize(), cfg_->OfdmDataNum() - base_sc_id);
  // printf("BeamBlockSize: %zu, OfdmDataNum: %zu,"
  //        " base_sc_id: %zu, last_sc_id: %zu\n",
  //        cfg_->BeamBlockSize(), cfg_->OfdmDataNum(), base_sc_id, last_sc_id);


  // Note: no subcarrirer grouping or partial transpose for special case.
  // Reduce to scalar, vectorized operation in special case (1x1 ant config),
  // uplink, zeroforcing
  if (cfg_->SmallMimoAcc() && // enables special case acceleration
      cfg_->BsAntNum() == 1 && cfg_->UeAntNum() == 1 && // conditions
      cfg_->SpatialStreamsNum() == 1 &&
      cfg_->BeamformingAlgo() == CommsLib::BeamformingAlgorithm::kZF &&
      cfg_->Frame().NumDLSyms() == 0 &&
      num_ext_ref_ == 0 &&
      kUseInverseForZF) {
    
    const size_t start_tsc1 = GetTime::WorkerRdtsc();

    const size_t sc_vec_len = cfg_->OfdmDataNum();
    const size_t ue_idx = 0; // If UeAntNum() == 1, only one UE exists.

   // Equivalent to: arma::inv_sympd(mat_csi.t() * mat_csi) * mat_csi.t();
#if defined(__AVX512F__) && defined(AVX512_MATOP)
    // Gather CSI
    complex_float* ptr_src = csi_buffers_[frame_slot][ue_idx];

    // Prepare UL beam matrix. Linearly distribute the memory.
    complex_float* ptr_ul_beam = ul_beam_matrices_[frame_slot][base_sc_id];

    const size_t start_tsc2 = GetTime::WorkerRdtsc();
    duration_stat_->task_duration_[1] += start_tsc2 - start_tsc1;

    // A = [a], B = [1/a] = A^(-1)
    for (size_t i = 0; i < sc_vec_len; i += kSCsPerCacheline) {
      __m512 a = _mm512_loadu_ps(ptr_src + i);
      __m512 inv_a = CommsLib::M512ComplexCf32Reciprocal(a);
      _mm512_storeu_ps(ptr_ul_beam + i, inv_a);
    }
#else
    // Gather CSI
    complex_float* cx_src = &csi_buffers_[frame_slot][ue_idx][base_sc_id];
    arma::cx_fvec csi_vec((arma::cx_float*)cx_src, sc_vec_len, false);

    // Prepare UL beam matrix.
    complex_float* ul_beam_mem = ul_beam_matrices_[frame_slot][base_sc_id];
    arma::cx_fvec ul_beam_vec((arma::cx_float*)ul_beam_mem, sc_vec_len, false);

    const size_t start_tsc2 = GetTime::WorkerRdtsc();
    duration_stat_->task_duration_[1] += start_tsc2 - start_tsc1;

    // Compute beam weights (zero-forcing)
    ul_beam_vec = (1/(arma::square(arma::real(csi_vec)) + 
                      arma::square(arma::imag(csi_vec))))
                  % arma::conj(csi_vec);
#endif

    duration_stat_->task_duration_[2] += GetTime::WorkerRdtsc() - start_tsc2;
    duration_stat_->task_count_++;
    duration_stat_->task_duration_[0] += GetTime::WorkerRdtsc() - start_tsc1;
    return;
  } else if (cfg_->SmallMimoAcc() && // enables special case acceleration
             cfg_->BsAntNum() == 2 && cfg_->UeAntNum() == 2 && // conditions
             cfg_->SpatialStreamsNum() == 2 &&
             cfg_->BeamformingAlgo() == CommsLib::BeamformingAlgorithm::kZF &&
             cfg_->Frame().NumDLSyms() == 0 &&
             num_ext_ref_ == 0 &&
             kUseInverseForZF) {

    const size_t sc_vec_len = cfg_->OfdmDataNum();

    const size_t start_tsc1 = GetTime::WorkerRdtsc();

    // Equivalent to: arma::inv_sympd(mat_csi.t() * mat_csi) * mat_csi.t();
#if defined(__AVX512F__) && defined(AVX512_MATOP)

    // Gather CSI = [csi_a, csi_b; csi_c, csi_d]
    complex_float* ptr_a = csi_buffers_[frame_slot][0];
    complex_float* ptr_b = csi_buffers_[frame_slot][1];
    complex_float* ptr_c = ptr_a + sc_vec_len;
    complex_float* ptr_d = ptr_b + sc_vec_len;

    // Prepare UL beam matrix. Linearly distribute the memory.
    complex_float* ul_beam_mem = ul_beam_matrices_[frame_slot][0];
    complex_float* ul_beam_a = ul_beam_mem;
    complex_float* ul_beam_b = ul_beam_mem + sc_vec_len;
    complex_float* ul_beam_c = ul_beam_mem + 2 * sc_vec_len;
    complex_float* ul_beam_d = ul_beam_mem + 3 * sc_vec_len;

    // arma::cx_frowvec vec_det = arma::zeros<arma::cx_frowvec>(sc_vec_len);
    // complex_float* ptr_det =
    //   reinterpret_cast<complex_float*>(vec_det.memptr());

    const size_t start_tsc2 = GetTime::WorkerRdtsc();
    duration_stat_->task_duration_[1] += start_tsc2 - start_tsc1;

    // A = [ a b ], B = [a' b'] = [d  -b] / (a*d - b*c) = A^(-1)
    //     [ c d ]      [c' d']   [-c  a]
    for (size_t i = 0; i < sc_vec_len; i += kSCsPerCacheline) {
      __m512 a = _mm512_loadu_ps(ptr_a + i);
      __m512 b = _mm512_loadu_ps(ptr_b + i);
      __m512 c = _mm512_loadu_ps(ptr_c + i);
      __m512 d = _mm512_loadu_ps(ptr_d + i);

      // det = a*d - b*c
      __m512 term1 = CommsLib::M512ComplexCf32Mult(a, d, false);
      __m512 term2 = CommsLib::M512ComplexCf32Mult(b, c, false);
      __m512 det = _mm512_sub_ps(term1, term2);

      // lack of division of complex numbers.
      // use reciprocal for division since multiplication is faster
      det = CommsLib::M512ComplexCf32Reciprocal(det);

      // check if the channel matrix is invertible,
      // float lowest > 1e-38, normal range > 1e-8
      if (unlikely(CommsLib::M512ComplexCf32NearZeros(det, 1e-10))) {
        AGORA_LOG_WARN("Channel matrix seems not invertible\n");
      }
      // _mm512_storeu_ps(ptr_det + i, det);

      a = CommsLib::M512ComplexCf32Mult(a, det, false);
      b = CommsLib::M512ComplexCf32Mult(b, det, false);
      c = CommsLib::M512ComplexCf32Mult(c, det, false);
      d = CommsLib::M512ComplexCf32Mult(d, det, false);

      __m512 neg = _mm512_set1_ps(-1.0f);
      b = _mm512_mul_ps(b, neg);
      c = _mm512_mul_ps(c, neg);
      _mm512_storeu_ps(ul_beam_a + i, d);
      _mm512_storeu_ps(ul_beam_b + i, b);
      _mm512_storeu_ps(ul_beam_c + i, c);
      _mm512_storeu_ps(ul_beam_d + i, a);
    }
#elif defined(ARMA_VEC_MATOP)
    // Gather CSI = [csi_a, csi_b; csi_c, csi_d]
    auto* cx_src =
      reinterpret_cast<complex_float*>(csi_buffers_[frame_slot][0]);
    arma::cx_fvec vec_csi_a = arma::cx_fvec(
      (arma::cx_float*)(cx_src + 0 * cfg_->OfdmDataNum()), sc_vec_len, false);
    arma::cx_fvec vec_csi_c = arma::cx_fvec(
      (arma::cx_float*)(cx_src + 1 * cfg_->OfdmDataNum()), sc_vec_len, false);
    
    cx_src = reinterpret_cast<complex_float*>(csi_buffers_[frame_slot][1]);
    arma::cx_fvec vec_csi_b = arma::cx_fvec(
      (arma::cx_float*)(cx_src + 0 * cfg_->OfdmDataNum()), sc_vec_len, false);
    arma::cx_fvec vec_csi_d = arma::cx_fvec(
      (arma::cx_float*)(cx_src + 1 * cfg_->OfdmDataNum()), sc_vec_len, false);

    // Prepare UL beam matrix. Let Armadillo help handle the memory.
    // The format here is identical to the input of equalizer.
    arma::cx_float* ul_beam_ptr = reinterpret_cast<arma::cx_float*>(
      ul_beam_matrices_[frame_slot][cfg_->GetBeamScId(base_sc_id)]);
    // The following vector view uses the same format as AVX512 memory layout.
    arma::cx_fvec vec_ul_beam(
        (arma::cx_float*)ul_beam_ptr,
        sc_vec_len * cfg_->SpatialStreamsNum() * cfg_->BsAntNum(),
        false);

    const size_t start_tsc2 = GetTime::WorkerRdtsc();
    duration_stat_->task_duration_[1] += start_tsc2 - start_tsc1;

    // A = [ a b ], B = [a' b'] = [d  -b] / (a*d - b*c) = A^(-1)
    //     [ c d ]      [c' d']   [-c  a]

    // det = a*d - b*c
    arma::cx_fvec vec_det = (vec_csi_a % vec_csi_d) - (vec_csi_b % vec_csi_c);

    // check if the channel matrix is invertible,
    // float lowest > 1e-38, normal range > 1e-8
    if (unlikely(vec_det.is_zero(1e-10))) {
      AGORA_LOG_WARN("Channel matrix seems not invertible\n");
    }

    // use reciprocal for division since multiplication is faster
    vec_det = 1.0 / vec_det;

    size_t shift = sc_vec_len;
    vec_ul_beam.subvec(0, shift - 1) = vec_csi_d % vec_det;
    vec_ul_beam.subvec(shift, 2 * shift - 1) = -vec_csi_b % vec_det;
    vec_ul_beam.subvec(2 * shift, 3 * shift - 1) = -vec_csi_c % vec_det;
    vec_ul_beam.subvec(3 * shift, 4 * shift - 1) = vec_csi_a % vec_det;

#elif defined(ARMA_CUBE_MATOP)
    arma::cx_fcube cub_csi(cfg_->BsAntNum(), cfg_->SpatialStreamsNum(),
                           sc_vec_len);
    // Default: Handle each subcarrier one by one
    // Gather CSI = [csi_a, csi_b; csi_c, csi_d]
    // Handle each subcarrier in the block (base_sc_id : last_sc_id -1)
    for (size_t cur_sc_id = base_sc_id; cur_sc_id < last_sc_id; cur_sc_id++) {
      // Gather CSI matrices of each pilot from partially-transposed CSIs.
      arma::uvec ue_list = mac_sched_->ScheduledUeList(frame_id, cur_sc_id);
      for (size_t selected_ue_idx = 0;
          selected_ue_idx < cfg_->SpatialStreamsNum(); selected_ue_idx++) {
        size_t ue_idx = ue_list.at(selected_ue_idx);
        auto* dst_csi_ptr = reinterpret_cast<float*>(
            csi_gather_buffer_ + cfg_->BsAntNum() * selected_ue_idx);
        if (kUsePartialTrans) {
          PartialTransposeGather(
              cur_sc_id,
              reinterpret_cast<float*>(csi_buffers_[frame_slot][ue_idx]),
              dst_csi_ptr, cfg_->BsAntNum());
        } else {
          TransposeGather(
              cur_sc_id,
              reinterpret_cast<float*>(csi_buffers_[frame_slot][ue_idx]),
              dst_csi_ptr, cfg_->BsAntNum(), cfg_->OfdmDataNum());
        }
      }
      arma::cx_fmat mat_csi_temp((arma::cx_float*)csi_gather_buffer_,
                                 cfg_->BsAntNum(),
                                 cfg_->SpatialStreamsNum(), false);
      cub_csi.slice(cur_sc_id) = mat_csi_temp;
    }

    // Prepare UL beam matrix. Let Armadillo help handle the memory.
    // The format here is identical to the input of equalizer. Note 
    // that the memory layout for Armadillo (cube) & AVX512 impl are different.
    arma::cx_float* ul_beam_ptr = reinterpret_cast<arma::cx_float*>(
      ul_beam_matrices_[frame_slot][cfg_->GetBeamScId(base_sc_id)]);
    arma::cx_fcube cub_ul_beam(ul_beam_ptr, cfg_->SpatialStreamsNum(),
                               cfg_->BsAntNum(), sc_vec_len, false);

    const size_t start_tsc2 = GetTime::WorkerRdtsc();
    duration_stat_->task_duration_[1] += start_tsc2 - start_tsc1;

    // A = [ a b ], B = [a' b'] = [d  -b] / (a*d - b*c) = A^(-1)
    //     [ c d ]      [c' d']   [-c  a]

    // det = a*d - b*c
    arma::cx_fcube cub_det = (cub_csi.tube(0, 0) % cub_csi.tube(1, 1)) -
                             (cub_csi.tube(0, 1) % cub_csi.tube(1, 0));

    // check if the channel matrix is invertible,
    // float lowest > 1e-38, normal range > 1e-8
    if (unlikely(cub_det.is_zero(1e-10))) {
      AGORA_LOG_WARN("Channel matrix seems not invertible\n");
    }

    // use reciprocal for division since multiplication is faster
    cub_det = 1.0 / cub_det;

    cub_ul_beam.tube(0, 0) =  cub_csi.tube(1, 1) % cub_det;
    cub_ul_beam.tube(0, 1) = -cub_csi.tube(0, 1) % cub_det;
    cub_ul_beam.tube(1, 0) = -cub_csi.tube(1, 0) % cub_det;
    cub_ul_beam.tube(1, 1) =  cub_csi.tube(0, 0) % cub_det;
#else
    // Gather CSI = [csi_a, csi_b; csi_c, csi_d]
    auto* cx_src =
      reinterpret_cast<complex_float*>(csi_buffers_[frame_slot][0]);
    arma::cx_fvec vec_csi_a = arma::cx_fvec(
      (arma::cx_float*)(cx_src + 0 * cfg_->OfdmDataNum()), sc_vec_len, false);
    arma::cx_fvec vec_csi_c = arma::cx_fvec(
      (arma::cx_float*)(cx_src + 1 * cfg_->OfdmDataNum()), sc_vec_len, false);
    
    cx_src = reinterpret_cast<complex_float*>(csi_buffers_[frame_slot][1]);
    arma::cx_fvec vec_csi_b = arma::cx_fvec(
      (arma::cx_float*)(cx_src + 0 * cfg_->OfdmDataNum()), sc_vec_len, false);
    arma::cx_fvec vec_csi_d = arma::cx_fvec(
      (arma::cx_float*)(cx_src + 1 * cfg_->OfdmDataNum()), sc_vec_len, false);

    // Prepare UL beam matrix. Let Armadillo help handle the memory.
    // The format here is identical to the input of equalizer.
    // Note that the memory layout for Armadillo & AVX512 impl are different.
    arma::cx_float* ul_beam_ptr = reinterpret_cast<arma::cx_float*>(
      ul_beam_matrices_[frame_slot][cfg_->GetBeamScId(base_sc_id)]);
    arma::cx_fcube cub_ul_beam(ul_beam_ptr, cfg_->SpatialStreamsNum(),
                               cfg_->BsAntNum(), sc_vec_len, false);
    // The following vector view uses the same format as AVX512 memory layout.
    // arma::cx_fvec vec_ul_beam(
    //     (arma::cx_float*)ul_beam_ptr,
    //     sc_vec_len * cfg_->SpatialStreamsNum() * cfg_->BsAntNum(),
    //     false);

    const size_t start_tsc2 = GetTime::WorkerRdtsc();
    duration_stat_->task_duration_[1] += start_tsc2 - start_tsc1;

    // A = [ a b ], B = [a' b'] = [d  -b] / (a*d - b*c) = A^(-1)
    //     [ c d ]      [c' d']   [-c  a]
    arma::cx_fcube cub_csi(cfg_->BsAntNum(), cfg_->SpatialStreamsNum(),
                           sc_vec_len);
    cub_csi.tube(0, 0) = vec_csi_a;
    cub_csi.tube(0, 1) = vec_csi_b;
    cub_csi.tube(1, 0) = vec_csi_c;
    cub_csi.tube(1, 1) = vec_csi_d;

    // det = a*d - b*c
    arma::cx_fcube cub_det = (cub_csi.tube(0, 0) % cub_csi.tube(1, 1)) -
                             (cub_csi.tube(0, 1) % cub_csi.tube(1, 0));

    // check if the channel matrix is invertible,
    // float lowest > 1e-38, normal range > 1e-8
    if (unlikely(cub_det.is_zero(1e-10))) {
      AGORA_LOG_WARN("Channel matrix seems not invertible\n");
    }

    // use reciprocal for division since multiplication is faster
    cub_det = 1.0 / cub_det;

    cub_ul_beam.tube(0, 0) =  cub_csi.tube(1, 1) % cub_det;
    cub_ul_beam.tube(0, 1) = -cub_csi.tube(0, 1) % cub_det;
    cub_ul_beam.tube(1, 0) = -cub_csi.tube(1, 0) % cub_det;
    cub_ul_beam.tube(1, 1) =  cub_csi.tube(0, 0) % cub_det;

    // arma::cx_fvec ul_beam_a = cub_ul_beam.tube(0, 0);
    // arma::cx_fvec ul_beam_b = cub_ul_beam.tube(0, 1);
    // arma::cx_fvec ul_beam_c = cub_ul_beam.tube(1, 0);
    // arma::cx_fvec ul_beam_d = cub_ul_beam.tube(1, 1);
    // vec_ul_beam.subvec(0, sc_vec_len - 1) = ul_beam_a;
    // vec_ul_beam.subvec(sc_vec_len, 2 * sc_vec_len - 1) = ul_beam_b;
    // vec_ul_beam.subvec(2 * sc_vec_len, 3 * sc_vec_len - 1) = ul_beam_c;
    // vec_ul_beam.subvec(3 * sc_vec_len, 4 * sc_vec_len - 1) = ul_beam_d;
#endif

    duration_stat_->task_duration_[2] += GetTime::WorkerRdtsc() - start_tsc2;
    duration_stat_->task_count_++;
    duration_stat_->task_duration_[0] += GetTime::WorkerRdtsc() - start_tsc1;
    return;
  } else if (cfg_->SmallMimoAcc() && // enables special case acceleration
             cfg_->BsAntNum() == 4 && cfg_->UeAntNum() == 4 && // conditions
             cfg_->SpatialStreamsNum() == 4 &&
             cfg_->BeamformingAlgo() == CommsLib::BeamformingAlgorithm::kZF &&
             cfg_->Frame().NumDLSyms() == 0 &&
             num_ext_ref_ == 0 &&
             kUseInverseForZF) {

    const size_t sc_vec_len = cfg_->OfdmDataNum();

    const size_t start_tsc1 = GetTime::WorkerRdtsc();

#if defined(__AVX512F__) && defined(AVX512_MATOP)
    // Gather CSI
    complex_float* ptr_0_0 = csi_buffers_[frame_slot][0];
    complex_float* ptr_0_1 = csi_buffers_[frame_slot][1];
    complex_float* ptr_0_2 = csi_buffers_[frame_slot][2];
    complex_float* ptr_0_3 = csi_buffers_[frame_slot][3];
    complex_float* ptr_1_0 = ptr_0_0 + sc_vec_len;
    complex_float* ptr_1_1 = ptr_0_1 + sc_vec_len;
    complex_float* ptr_1_2 = ptr_0_2 + sc_vec_len;
    complex_float* ptr_1_3 = ptr_0_3 + sc_vec_len;
    complex_float* ptr_2_0 = ptr_1_0 + sc_vec_len;
    complex_float* ptr_2_1 = ptr_1_1 + sc_vec_len;
    complex_float* ptr_2_2 = ptr_1_2 + sc_vec_len;
    complex_float* ptr_2_3 = ptr_1_3 + sc_vec_len;
    complex_float* ptr_3_0 = ptr_2_0 + sc_vec_len;
    complex_float* ptr_3_1 = ptr_2_1 + sc_vec_len;
    complex_float* ptr_3_2 = ptr_2_2 + sc_vec_len;
    complex_float* ptr_3_3 = ptr_2_3 + sc_vec_len;

    // Prepare UL beam matrix. Linearly distribute the memory.
    complex_float* ul_beam_mem = ul_beam_matrices_[frame_slot][0];
    complex_float* ul_beam_0_0 = ul_beam_mem;
    complex_float* ul_beam_0_1 = ul_beam_0_0 + sc_vec_len;
    complex_float* ul_beam_0_2 = ul_beam_0_1 + sc_vec_len;
    complex_float* ul_beam_0_3 = ul_beam_0_2 + sc_vec_len;
    complex_float* ul_beam_1_0 = ul_beam_0_3 + sc_vec_len;
    complex_float* ul_beam_1_1 = ul_beam_1_0 + sc_vec_len;
    complex_float* ul_beam_1_2 = ul_beam_1_1 + sc_vec_len;
    complex_float* ul_beam_1_3 = ul_beam_1_2 + sc_vec_len;
    complex_float* ul_beam_2_0 = ul_beam_1_3 + sc_vec_len;
    complex_float* ul_beam_2_1 = ul_beam_2_0 + sc_vec_len;
    complex_float* ul_beam_2_2 = ul_beam_2_1 + sc_vec_len;
    complex_float* ul_beam_2_3 = ul_beam_2_2 + sc_vec_len;
    complex_float* ul_beam_3_0 = ul_beam_2_3 + sc_vec_len;
    complex_float* ul_beam_3_1 = ul_beam_3_0 + sc_vec_len;
    complex_float* ul_beam_3_2 = ul_beam_3_1 + sc_vec_len;
    complex_float* ul_beam_3_3 = ul_beam_3_2 + sc_vec_len;

    // arma::cx_frowvec vec_det = arma::zeros<arma::cx_frowvec>(sc_vec_len);
    // complex_float* ptr_det =
    //   reinterpret_cast<complex_float*>(vec_det.memptr());

    const size_t start_tsc2 = GetTime::WorkerRdtsc();
    duration_stat_->task_duration_[1] += start_tsc2 - start_tsc1;

    for (size_t i = 0; i < sc_vec_len; i += kSCsPerCacheline) {
      __m512 csi_0_0 = _mm512_loadu_ps(ptr_0_0 + i);
      __m512 csi_0_1 = _mm512_loadu_ps(ptr_0_1 + i);
      __m512 csi_0_2 = _mm512_loadu_ps(ptr_0_2 + i);
      __m512 csi_0_3 = _mm512_loadu_ps(ptr_0_3 + i);
      __m512 csi_1_0 = _mm512_loadu_ps(ptr_1_0 + i);
      __m512 csi_1_1 = _mm512_loadu_ps(ptr_1_1 + i);
      __m512 csi_1_2 = _mm512_loadu_ps(ptr_1_2 + i);
      __m512 csi_1_3 = _mm512_loadu_ps(ptr_1_3 + i);
      __m512 csi_2_0 = _mm512_loadu_ps(ptr_2_0 + i);
      __m512 csi_2_1 = _mm512_loadu_ps(ptr_2_1 + i);
      __m512 csi_2_2 = _mm512_loadu_ps(ptr_2_2 + i);
      __m512 csi_2_3 = _mm512_loadu_ps(ptr_2_3 + i);
      __m512 csi_3_0 = _mm512_loadu_ps(ptr_3_0 + i);
      __m512 csi_3_1 = _mm512_loadu_ps(ptr_3_1 + i);
      __m512 csi_3_2 = _mm512_loadu_ps(ptr_3_2 + i);
      __m512 csi_3_3 = _mm512_loadu_ps(ptr_3_3 + i);

      // Prepare operands to avoid repeated computation
      __m512 prod_1_0_x_2_1 = CommsLib::M512ComplexCf32Mult(csi_1_0, csi_2_1, false);
      __m512 prod_1_0_x_2_2 = CommsLib::M512ComplexCf32Mult(csi_1_0, csi_2_2, false);
      __m512 prod_1_0_x_2_3 = CommsLib::M512ComplexCf32Mult(csi_1_0, csi_2_3, false);
      __m512 prod_1_0_x_3_1 = CommsLib::M512ComplexCf32Mult(csi_1_0, csi_3_1, false);
      __m512 prod_1_0_x_3_2 = CommsLib::M512ComplexCf32Mult(csi_1_0, csi_3_2, false);
      __m512 prod_1_0_x_3_3 = CommsLib::M512ComplexCf32Mult(csi_1_0, csi_3_3, false);

      __m512 prod_1_1_x_2_0 = CommsLib::M512ComplexCf32Mult(csi_1_1, csi_2_0, false);
      __m512 prod_1_1_x_2_2 = CommsLib::M512ComplexCf32Mult(csi_1_1, csi_2_2, false);
      __m512 prod_1_1_x_2_3 = CommsLib::M512ComplexCf32Mult(csi_1_1, csi_2_3, false);
      __m512 prod_1_1_x_3_0 = CommsLib::M512ComplexCf32Mult(csi_1_1, csi_3_0, false);
      __m512 prod_1_1_x_3_2 = CommsLib::M512ComplexCf32Mult(csi_1_1, csi_3_2, false);
      __m512 prod_1_1_x_3_3 = CommsLib::M512ComplexCf32Mult(csi_1_1, csi_3_3, false);

      __m512 prod_1_2_x_2_0 = CommsLib::M512ComplexCf32Mult(csi_1_2, csi_2_0, false);
      __m512 prod_1_2_x_2_1 = CommsLib::M512ComplexCf32Mult(csi_1_2, csi_2_1, false);
      __m512 prod_1_2_x_2_3 = CommsLib::M512ComplexCf32Mult(csi_1_2, csi_2_3, false);
      __m512 prod_1_2_x_3_0 = CommsLib::M512ComplexCf32Mult(csi_1_2, csi_3_0, false);
      __m512 prod_1_2_x_3_1 = CommsLib::M512ComplexCf32Mult(csi_1_2, csi_3_1, false);
      __m512 prod_1_2_x_3_3 = CommsLib::M512ComplexCf32Mult(csi_1_2, csi_3_3, false);

      __m512 prod_1_3_x_2_0 = CommsLib::M512ComplexCf32Mult(csi_1_3, csi_2_0, false);
      __m512 prod_1_3_x_2_1 = CommsLib::M512ComplexCf32Mult(csi_1_3, csi_2_1, false);
      __m512 prod_1_3_x_2_2 = CommsLib::M512ComplexCf32Mult(csi_1_3, csi_2_2, false);
      __m512 prod_1_3_x_3_0 = CommsLib::M512ComplexCf32Mult(csi_1_3, csi_3_0, false);
      __m512 prod_1_3_x_3_1 = CommsLib::M512ComplexCf32Mult(csi_1_3, csi_3_1, false);
      __m512 prod_1_3_x_3_2 = CommsLib::M512ComplexCf32Mult(csi_1_3, csi_3_2, false);

      __m512 prod_2_0_x_3_1 = CommsLib::M512ComplexCf32Mult(csi_2_0, csi_3_1, false);
      __m512 prod_2_0_x_3_2 = CommsLib::M512ComplexCf32Mult(csi_2_0, csi_3_2, false);
      __m512 prod_2_0_x_3_3 = CommsLib::M512ComplexCf32Mult(csi_2_0, csi_3_3, false);

      __m512 prod_2_1_x_3_0 = CommsLib::M512ComplexCf32Mult(csi_2_1, csi_3_0, false);
      __m512 prod_2_1_x_3_2 = CommsLib::M512ComplexCf32Mult(csi_2_1, csi_3_2, false);
      __m512 prod_2_1_x_3_3 = CommsLib::M512ComplexCf32Mult(csi_2_1, csi_3_3, false);

      __m512 prod_2_2_x_3_0 = CommsLib::M512ComplexCf32Mult(csi_2_2, csi_3_0, false);
      __m512 prod_2_2_x_3_1 = CommsLib::M512ComplexCf32Mult(csi_2_2, csi_3_1, false);
      __m512 prod_2_2_x_3_3 = CommsLib::M512ComplexCf32Mult(csi_2_2, csi_3_3, false);

      __m512 prod_2_3_x_3_0 = CommsLib::M512ComplexCf32Mult(csi_2_3, csi_3_0, false);
      __m512 prod_2_3_x_3_1 = CommsLib::M512ComplexCf32Mult(csi_2_3, csi_3_1, false);
      __m512 prod_2_3_x_3_2 = CommsLib::M512ComplexCf32Mult(csi_2_3, csi_3_2, false);

      // Calculate determinant, please refer to arma impl for formula
      __m512 term1 = _mm512_sub_ps(prod_2_2_x_3_3, prod_2_3_x_3_2);
      __m512 term2 = _mm512_sub_ps(prod_2_3_x_3_1, prod_2_1_x_3_3);
      __m512 term3 = _mm512_sub_ps(prod_2_1_x_3_2, prod_2_2_x_3_1);
      term1 = CommsLib::M512ComplexCf32Mult(csi_1_1, term1, false);
      term2 = CommsLib::M512ComplexCf32Mult(csi_1_2, term2, false);
      term3 = CommsLib::M512ComplexCf32Mult(csi_1_3, term3, false);
      __m512 term = _mm512_add_ps(term1, _mm512_add_ps(term2, term3));
      term = CommsLib::M512ComplexCf32Mult(csi_0_0, term, false);
      __m512 det = term;

      term1 = _mm512_sub_ps(prod_2_2_x_3_3, prod_2_3_x_3_2);
      term2 = _mm512_sub_ps(prod_2_3_x_3_1, prod_2_1_x_3_3);
      term3 = _mm512_sub_ps(prod_2_1_x_3_2, prod_2_2_x_3_1);
      term1 = CommsLib::M512ComplexCf32Mult(csi_0_1, term1, false);
      term2 = CommsLib::M512ComplexCf32Mult(csi_0_2, term2, false);
      term3 = CommsLib::M512ComplexCf32Mult(csi_0_3, term3, false);
      term = _mm512_add_ps(term1, _mm512_add_ps(term2, term3));
      term = CommsLib::M512ComplexCf32Mult(csi_1_0, term, false);
      det = _mm512_sub_ps(det, term);

      term1 = _mm512_sub_ps(prod_1_2_x_3_3, prod_1_3_x_3_2);
      term2 = _mm512_sub_ps(prod_1_3_x_3_1, prod_1_1_x_3_3);
      term3 = _mm512_sub_ps(prod_1_1_x_3_2, prod_1_2_x_3_1);
      term1 = CommsLib::M512ComplexCf32Mult(csi_0_1, term1, false);
      term2 = CommsLib::M512ComplexCf32Mult(csi_0_2, term2, false);
      term3 = CommsLib::M512ComplexCf32Mult(csi_0_3, term3, false);
      term = _mm512_add_ps(term1, _mm512_add_ps(term2, term3));
      term = CommsLib::M512ComplexCf32Mult(csi_2_0, term, false);
      det = _mm512_add_ps(det, term);

      term1 = _mm512_sub_ps(prod_1_2_x_2_3, prod_1_3_x_2_2);
      term2 = _mm512_sub_ps(prod_1_3_x_2_1, prod_1_1_x_2_3);
      term3 = _mm512_sub_ps(prod_1_1_x_2_2, prod_1_2_x_2_1);
      term1 = CommsLib::M512ComplexCf32Mult(csi_0_1, term1, false);
      term2 = CommsLib::M512ComplexCf32Mult(csi_0_2, term2, false);
      term3 = CommsLib::M512ComplexCf32Mult(csi_0_3, term3, false);
      term = _mm512_add_ps(term1, _mm512_add_ps(term2, term3));
      term = CommsLib::M512ComplexCf32Mult(csi_3_0, term, false);
      det = _mm512_sub_ps(det, term);

      // check if the channel matrix is invertible,
      // float lowest > 1e-38, normal range > 1e-8
      if (unlikely(CommsLib::M512ComplexCf32NearZeros(det, 1e-10))) {
        AGORA_LOG_WARN("Channel matrix seems not invertible\n");
      }
      // _mm512_storeu_ps(ptr_det + i, det);

      // lack of division of complex numbers.
      // use reciprocal for division since multiplication is faster
      det = CommsLib::M512ComplexCf32Reciprocal(det);

      term1 = _mm512_sub_ps(prod_2_2_x_3_3, prod_2_3_x_3_2);
      term2 = _mm512_sub_ps(prod_2_3_x_3_1, prod_2_1_x_3_3);
      term3 = _mm512_sub_ps(prod_2_1_x_3_2, prod_2_2_x_3_1);
      term1 = CommsLib::M512ComplexCf32Mult(csi_1_1, term1, false);
      term2 = CommsLib::M512ComplexCf32Mult(csi_1_2, term2, false);
      term3 = CommsLib::M512ComplexCf32Mult(csi_1_3, term3, false);
      __m512 adj_0_0 = _mm512_add_ps(term1, _mm512_add_ps(term2, term3));

      term1 = _mm512_sub_ps(prod_2_3_x_3_2, prod_2_2_x_3_3);
      term2 = _mm512_sub_ps(prod_2_1_x_3_3, prod_2_3_x_3_1);
      term3 = _mm512_sub_ps(prod_2_2_x_3_1, prod_2_1_x_3_2);
      term1 = CommsLib::M512ComplexCf32Mult(csi_0_1, term1, false);
      term2 = CommsLib::M512ComplexCf32Mult(csi_0_2, term2, false);
      term3 = CommsLib::M512ComplexCf32Mult(csi_0_3, term3, false);
      __m512 adj_0_1 = _mm512_add_ps(term1, _mm512_add_ps(term2, term3));

      term1 = _mm512_sub_ps(prod_1_2_x_3_3, prod_1_3_x_3_2);
      term2 = _mm512_sub_ps(prod_1_3_x_3_1, prod_1_1_x_3_3);
      term3 = _mm512_sub_ps(prod_1_1_x_3_2, prod_1_2_x_3_1);
      term1 = CommsLib::M512ComplexCf32Mult(csi_0_1, term1, false);
      term2 = CommsLib::M512ComplexCf32Mult(csi_0_2, term2, false);
      term3 = CommsLib::M512ComplexCf32Mult(csi_0_3, term3, false);
      __m512 adj_0_2 = _mm512_add_ps(term1, _mm512_add_ps(term2, term3));

      term1 = _mm512_sub_ps(prod_1_3_x_2_2, prod_1_2_x_2_3);
      term2 = _mm512_sub_ps(prod_1_1_x_2_3, prod_1_3_x_2_1);
      term3 = _mm512_sub_ps(prod_1_2_x_2_1, prod_1_1_x_2_2);
      term1 = CommsLib::M512ComplexCf32Mult(csi_0_1, term1, false);
      term2 = CommsLib::M512ComplexCf32Mult(csi_0_2, term2, false);
      term3 = CommsLib::M512ComplexCf32Mult(csi_0_3, term3, false);
      __m512 adj_0_3 = _mm512_add_ps(term1, _mm512_add_ps(term2, term3));

      term1 = _mm512_sub_ps(prod_2_3_x_3_2, prod_2_2_x_3_3);
      term2 = _mm512_sub_ps(prod_2_0_x_3_3, prod_2_3_x_3_0);
      term3 = _mm512_sub_ps(prod_2_2_x_3_0, prod_2_0_x_3_2);
      term1 = CommsLib::M512ComplexCf32Mult(csi_1_0, term1, false);
      term2 = CommsLib::M512ComplexCf32Mult(csi_1_2, term2, false);
      term3 = CommsLib::M512ComplexCf32Mult(csi_1_3, term3, false);
      __m512 adj_1_0 = _mm512_add_ps(term1, _mm512_add_ps(term2, term3));

      term1 = _mm512_sub_ps(prod_2_2_x_3_3, prod_2_3_x_3_2);
      term2 = _mm512_sub_ps(prod_2_3_x_3_0, prod_2_0_x_3_3);
      term3 = _mm512_sub_ps(prod_2_0_x_3_2, prod_2_2_x_3_0);
      term1 = CommsLib::M512ComplexCf32Mult(csi_0_0, term1, false);
      term2 = CommsLib::M512ComplexCf32Mult(csi_0_2, term2, false);
      term3 = CommsLib::M512ComplexCf32Mult(csi_0_3, term3, false);
      __m512 adj_1_1 = _mm512_add_ps(term1, _mm512_add_ps(term2, term3));

      term1 = _mm512_sub_ps(prod_1_3_x_3_2, prod_1_2_x_3_3);
      term2 = _mm512_sub_ps(prod_1_0_x_3_3, prod_1_3_x_3_0);
      term3 = _mm512_sub_ps(prod_1_2_x_3_0, prod_1_0_x_3_2);
      term1 = CommsLib::M512ComplexCf32Mult(csi_0_0, term1, false);
      term2 = CommsLib::M512ComplexCf32Mult(csi_0_2, term2, false);
      term3 = CommsLib::M512ComplexCf32Mult(csi_0_3, term3, false);
      __m512 adj_1_2 = _mm512_add_ps(term1, _mm512_add_ps(term2, term3));

      term1 = _mm512_sub_ps(prod_1_2_x_2_3, prod_1_3_x_2_2);
      term2 = _mm512_sub_ps(prod_1_3_x_2_0, prod_1_0_x_2_3);
      term3 = _mm512_sub_ps(prod_1_0_x_2_2, prod_1_2_x_2_0);
      term1 = CommsLib::M512ComplexCf32Mult(csi_0_0, term1, false);
      term2 = CommsLib::M512ComplexCf32Mult(csi_0_2, term2, false);
      term3 = CommsLib::M512ComplexCf32Mult(csi_0_3, term3, false);
      __m512 adj_1_3 = _mm512_add_ps(term1, _mm512_add_ps(term2, term3));

      term1 = _mm512_sub_ps(prod_2_1_x_3_3, prod_2_3_x_3_1);
      term2 = _mm512_sub_ps(prod_2_3_x_3_0, prod_2_0_x_3_3);
      term3 = _mm512_sub_ps(prod_2_0_x_3_1, prod_2_1_x_3_0);
      term1 = CommsLib::M512ComplexCf32Mult(csi_1_0, term1, false);
      term2 = CommsLib::M512ComplexCf32Mult(csi_1_1, term2, false);
      term3 = CommsLib::M512ComplexCf32Mult(csi_1_3, term3, false);
      __m512 adj_2_0 = _mm512_add_ps(term1, _mm512_add_ps(term2, term3));

      term1 = _mm512_sub_ps(prod_2_3_x_3_1, prod_2_1_x_3_3);
      term2 = _mm512_sub_ps(prod_2_0_x_3_3, prod_2_3_x_3_0);
      term3 = _mm512_sub_ps(prod_2_1_x_3_0, prod_2_0_x_3_1);
      term1 = CommsLib::M512ComplexCf32Mult(csi_0_0, term1, false);
      term2 = CommsLib::M512ComplexCf32Mult(csi_0_1, term2, false);
      term3 = CommsLib::M512ComplexCf32Mult(csi_0_3, term3, false);
      __m512 adj_2_1 = _mm512_add_ps(term1, _mm512_add_ps(term2, term3));

      term1 = _mm512_sub_ps(prod_1_1_x_3_3, prod_1_3_x_3_1);
      term2 = _mm512_sub_ps(prod_1_3_x_3_0, prod_1_0_x_3_3);
      term3 = _mm512_sub_ps(prod_1_0_x_3_1, prod_1_1_x_3_0);
      term1 = CommsLib::M512ComplexCf32Mult(csi_0_0, term1, false);
      term2 = CommsLib::M512ComplexCf32Mult(csi_0_1, term2, false);
      term3 = CommsLib::M512ComplexCf32Mult(csi_0_3, term3, false);
      __m512 adj_2_2 = _mm512_add_ps(term1, _mm512_add_ps(term2, term3));

      term1 = _mm512_sub_ps(prod_1_3_x_2_1, prod_1_1_x_2_3);
      term2 = _mm512_sub_ps(prod_1_0_x_2_3, prod_1_3_x_2_0);
      term3 = _mm512_sub_ps(prod_1_1_x_2_0, prod_1_0_x_2_1);
      term1 = CommsLib::M512ComplexCf32Mult(csi_0_0, term1, false);
      term2 = CommsLib::M512ComplexCf32Mult(csi_0_1, term2, false);
      term3 = CommsLib::M512ComplexCf32Mult(csi_0_3, term3, false);
      __m512 adj_2_3 = _mm512_add_ps(term1, _mm512_add_ps(term2, term3));

      term1 = _mm512_sub_ps(prod_2_2_x_3_1, prod_2_1_x_3_2);
      term2 = _mm512_sub_ps(prod_2_0_x_3_2, prod_2_2_x_3_0);
      term3 = _mm512_sub_ps(prod_2_1_x_3_0, prod_2_0_x_3_1);
      term1 = CommsLib::M512ComplexCf32Mult(csi_1_0, term1, false);
      term2 = CommsLib::M512ComplexCf32Mult(csi_1_1, term2, false);
      term3 = CommsLib::M512ComplexCf32Mult(csi_1_2, term3, false);
      __m512 adj_3_0 = _mm512_add_ps(term1, _mm512_add_ps(term2, term3));

      term1 = _mm512_sub_ps(prod_2_1_x_3_2, prod_2_2_x_3_1);
      term2 = _mm512_sub_ps(prod_2_2_x_3_0, prod_2_0_x_3_2);
      term3 = _mm512_sub_ps(prod_2_0_x_3_1, prod_2_1_x_3_0);
      term1 = CommsLib::M512ComplexCf32Mult(csi_0_0, term1, false);
      term2 = CommsLib::M512ComplexCf32Mult(csi_0_1, term2, false);
      term3 = CommsLib::M512ComplexCf32Mult(csi_0_2, term3, false);
      __m512 adj_3_1 = _mm512_add_ps(term1, _mm512_add_ps(term2, term3));

      term1 = _mm512_sub_ps(prod_1_2_x_3_1, prod_1_1_x_3_2);
      term2 = _mm512_sub_ps(prod_1_0_x_3_2, prod_1_2_x_3_0);
      term3 = _mm512_sub_ps(prod_1_1_x_3_0, prod_1_0_x_3_1);
      term1 = CommsLib::M512ComplexCf32Mult(csi_0_0, term1, false);
      term2 = CommsLib::M512ComplexCf32Mult(csi_0_1, term2, false);
      term3 = CommsLib::M512ComplexCf32Mult(csi_0_2, term3, false);
      __m512 adj_3_2 = _mm512_add_ps(term1, _mm512_add_ps(term2, term3));

      term1 = _mm512_sub_ps(prod_1_1_x_2_2, prod_1_2_x_2_1);
      term2 = _mm512_sub_ps(prod_1_2_x_2_0, prod_1_0_x_2_2);
      term3 = _mm512_sub_ps(prod_1_0_x_2_1, prod_1_1_x_2_0);
      term1 = CommsLib::M512ComplexCf32Mult(csi_0_0, term1, false);
      term2 = CommsLib::M512ComplexCf32Mult(csi_0_1, term2, false);
      term3 = CommsLib::M512ComplexCf32Mult(csi_0_2, term3, false);
      __m512 adj_3_3 = _mm512_add_ps(term1, _mm512_add_ps(term2, term3));

      adj_0_0 = CommsLib::M512ComplexCf32Mult(adj_0_0, det, false);
      adj_0_1 = CommsLib::M512ComplexCf32Mult(adj_0_1, det, false);
      adj_0_2 = CommsLib::M512ComplexCf32Mult(adj_0_2, det, false);
      adj_0_3 = CommsLib::M512ComplexCf32Mult(adj_0_3, det, false);
      adj_1_0 = CommsLib::M512ComplexCf32Mult(adj_1_0, det, false);
      adj_1_1 = CommsLib::M512ComplexCf32Mult(adj_1_1, det, false);
      adj_1_2 = CommsLib::M512ComplexCf32Mult(adj_1_2, det, false);
      adj_1_3 = CommsLib::M512ComplexCf32Mult(adj_1_3, det, false);
      adj_2_0 = CommsLib::M512ComplexCf32Mult(adj_2_0, det, false);
      adj_2_1 = CommsLib::M512ComplexCf32Mult(adj_2_1, det, false);
      adj_2_2 = CommsLib::M512ComplexCf32Mult(adj_2_2, det, false);
      adj_2_3 = CommsLib::M512ComplexCf32Mult(adj_2_3, det, false);
      adj_3_0 = CommsLib::M512ComplexCf32Mult(adj_3_0, det, false);
      adj_3_1 = CommsLib::M512ComplexCf32Mult(adj_3_1, det, false);
      adj_3_2 = CommsLib::M512ComplexCf32Mult(adj_3_2, det, false);
      adj_3_3 = CommsLib::M512ComplexCf32Mult(adj_3_3, det, false);

      // store the result
      _mm512_storeu_ps(ul_beam_0_0 + i, adj_0_0);
      _mm512_storeu_ps(ul_beam_0_1 + i, adj_0_1);
      _mm512_storeu_ps(ul_beam_0_2 + i, adj_0_2);
      _mm512_storeu_ps(ul_beam_0_3 + i, adj_0_3);
      _mm512_storeu_ps(ul_beam_1_0 + i, adj_1_0);
      _mm512_storeu_ps(ul_beam_1_1 + i, adj_1_1);
      _mm512_storeu_ps(ul_beam_1_2 + i, adj_1_2);
      _mm512_storeu_ps(ul_beam_1_3 + i, adj_1_3);
      _mm512_storeu_ps(ul_beam_2_0 + i, adj_2_0);
      _mm512_storeu_ps(ul_beam_2_1 + i, adj_2_1);
      _mm512_storeu_ps(ul_beam_2_2 + i, adj_2_2);
      _mm512_storeu_ps(ul_beam_2_3 + i, adj_2_3);
      _mm512_storeu_ps(ul_beam_3_0 + i, adj_3_0);
      _mm512_storeu_ps(ul_beam_3_1 + i, adj_3_1);
      _mm512_storeu_ps(ul_beam_3_2 + i, adj_3_2);
      _mm512_storeu_ps(ul_beam_3_3 + i, adj_3_3);
    }
#elif defined(ARMA_VEC_MATOP)
    // Prepare CSI matrix. Read in vectors.
    auto* cx_src =
      reinterpret_cast<complex_float*>(csi_buffers_[frame_slot][0]);
    arma::cx_fvec vec_csi_0_0 = arma::cx_fvec(
      (arma::cx_float*)(cx_src + 0 * cfg_->OfdmDataNum()), sc_vec_len, false);
    arma::cx_fvec vec_csi_1_0 = arma::cx_fvec(
      (arma::cx_float*)(cx_src + 1 * cfg_->OfdmDataNum()), sc_vec_len, false);
    arma::cx_fvec vec_csi_2_0 = arma::cx_fvec(
      (arma::cx_float*)(cx_src + 2 * cfg_->OfdmDataNum()), sc_vec_len, false);
    arma::cx_fvec vec_csi_3_0 = arma::cx_fvec(
      (arma::cx_float*)(cx_src + 3 * cfg_->OfdmDataNum()), sc_vec_len, false);

    cx_src = reinterpret_cast<complex_float*>(csi_buffers_[frame_slot][1]);
    arma::cx_fvec vec_csi_0_1 = arma::cx_fvec(
      (arma::cx_float*)(cx_src + 0 * cfg_->OfdmDataNum()), sc_vec_len, false);
    arma::cx_fvec vec_csi_1_1 = arma::cx_fvec(
      (arma::cx_float*)(cx_src + 1 * cfg_->OfdmDataNum()), sc_vec_len, false);
    arma::cx_fvec vec_csi_2_1 = arma::cx_fvec(
      (arma::cx_float*)(cx_src + 2 * cfg_->OfdmDataNum()), sc_vec_len, false);
    arma::cx_fvec vec_csi_3_1 = arma::cx_fvec(
      (arma::cx_float*)(cx_src + 3 * cfg_->OfdmDataNum()), sc_vec_len, false);

    cx_src = reinterpret_cast<complex_float*>(csi_buffers_[frame_slot][2]);
    arma::cx_fvec vec_csi_0_2 = arma::cx_fvec(
      (arma::cx_float*)(cx_src + 0 * cfg_->OfdmDataNum()), sc_vec_len, false);
    arma::cx_fvec vec_csi_1_2 = arma::cx_fvec(
      (arma::cx_float*)(cx_src + 1 * cfg_->OfdmDataNum()), sc_vec_len, false);
    arma::cx_fvec vec_csi_2_2 = arma::cx_fvec(
      (arma::cx_float*)(cx_src + 2 * cfg_->OfdmDataNum()), sc_vec_len, false);
    arma::cx_fvec vec_csi_3_2 = arma::cx_fvec(
      (arma::cx_float*)(cx_src + 3 * cfg_->OfdmDataNum()), sc_vec_len, false);

    cx_src = reinterpret_cast<complex_float*>(csi_buffers_[frame_slot][3]);
    arma::cx_fvec vec_csi_0_3 = arma::cx_fvec(
      (arma::cx_float*)(cx_src + 0 * cfg_->OfdmDataNum()), sc_vec_len, false);
    arma::cx_fvec vec_csi_1_3 = arma::cx_fvec(
      (arma::cx_float*)(cx_src + 1 * cfg_->OfdmDataNum()), sc_vec_len, false);
    arma::cx_fvec vec_csi_2_3 = arma::cx_fvec(
      (arma::cx_float*)(cx_src + 2 * cfg_->OfdmDataNum()), sc_vec_len, false);
    arma::cx_fvec vec_csi_3_3 = arma::cx_fvec(
      (arma::cx_float*)(cx_src + 3 * cfg_->OfdmDataNum()), sc_vec_len, false);

    const size_t start_tsc2 = GetTime::WorkerRdtsc();
    duration_stat_->task_duration_[1] += start_tsc2 - start_tsc1;

    // Calculate the determinant
    arma::cx_fvec vec_det =
      vec_csi_0_0 %
       (vec_csi_1_1 % (vec_csi_2_2 % vec_csi_3_3 - vec_csi_2_3 % vec_csi_3_2) +
        vec_csi_1_2 % (vec_csi_2_3 % vec_csi_3_1 - vec_csi_2_1 % vec_csi_3_3) +
        vec_csi_1_3 % (vec_csi_2_1 % vec_csi_3_2 - vec_csi_2_2 % vec_csi_3_1)) -
      vec_csi_1_0 %
       (vec_csi_0_1 % (vec_csi_2_2 % vec_csi_3_3 - vec_csi_2_3 % vec_csi_3_2) +
        vec_csi_0_2 % (vec_csi_2_3 % vec_csi_3_1 - vec_csi_2_1 % vec_csi_3_3) +
        vec_csi_0_3 % (vec_csi_2_1 % vec_csi_3_2 - vec_csi_2_2 % vec_csi_3_1)) +
      vec_csi_2_0 %
       (vec_csi_0_1 % (vec_csi_1_2 % vec_csi_3_3 - vec_csi_1_3 % vec_csi_3_2) +
        vec_csi_0_2 % (vec_csi_1_3 % vec_csi_3_1 - vec_csi_1_1 % vec_csi_3_3) +
        vec_csi_0_3 % (vec_csi_1_1 % vec_csi_3_2 - vec_csi_1_2 % vec_csi_3_1)) -
      vec_csi_3_0 %
       (vec_csi_0_1 % (vec_csi_1_2 % vec_csi_2_3 - vec_csi_1_3 % vec_csi_2_2) +
        vec_csi_0_2 % (vec_csi_1_3 % vec_csi_2_1 - vec_csi_1_1 % vec_csi_2_3) +
        vec_csi_0_3 % (vec_csi_1_1 % vec_csi_2_2 - vec_csi_1_2 % vec_csi_2_1));

    if (unlikely(arma::any(arma::abs(vec_det) < 1e-10))) {
      AGORA_LOG_WARN("Channel matrix seems not invertible\n");
    }

    // use reciprocal for division since multiplication is faster
    vec_det = 1.0 / vec_det;

    arma::cx_fvec vec_adj_0_0 =
      vec_csi_1_1 % (vec_csi_2_2 % vec_csi_3_3 - vec_csi_2_3 % vec_csi_3_2) +
      vec_csi_1_2 % (vec_csi_2_3 % vec_csi_3_1 - vec_csi_2_1 % vec_csi_3_3) +
      vec_csi_1_3 % (vec_csi_2_1 % vec_csi_3_2 - vec_csi_2_2 % vec_csi_3_1);
    arma::cx_fvec vec_adj_0_1 =
      vec_csi_0_1 % (vec_csi_2_3 % vec_csi_3_2 - vec_csi_2_2 % vec_csi_3_3) +
      vec_csi_0_2 % (vec_csi_2_1 % vec_csi_3_3 - vec_csi_2_3 % vec_csi_3_1) +
      vec_csi_0_3 % (vec_csi_2_2 % vec_csi_3_1 - vec_csi_2_1 % vec_csi_3_2);
    arma::cx_fvec vec_adj_0_2 =
      vec_csi_0_1 % (vec_csi_1_2 % vec_csi_3_3 - vec_csi_1_3 % vec_csi_3_2) +
      vec_csi_0_2 % (vec_csi_1_3 % vec_csi_3_1 - vec_csi_1_1 % vec_csi_3_3) +
      vec_csi_0_3 % (vec_csi_1_1 % vec_csi_3_2 - vec_csi_1_2 % vec_csi_3_1);
    arma::cx_fvec vec_adj_0_3 =
      vec_csi_0_1 % (vec_csi_1_3 % vec_csi_2_2 - vec_csi_1_2 % vec_csi_2_3) +
      vec_csi_0_2 % (vec_csi_1_1 % vec_csi_2_3 - vec_csi_1_3 % vec_csi_2_1) +
      vec_csi_0_3 % (vec_csi_1_2 % vec_csi_2_1 - vec_csi_1_1 % vec_csi_2_2);

    arma::cx_fvec vec_adj_1_0 =
      vec_csi_1_0 % (vec_csi_2_3 % vec_csi_3_2 - vec_csi_2_2 % vec_csi_3_3) +
      vec_csi_1_2 % (vec_csi_2_0 % vec_csi_3_3 - vec_csi_2_3 % vec_csi_3_0) +
      vec_csi_1_3 % (vec_csi_2_2 % vec_csi_3_0 - vec_csi_2_0 % vec_csi_3_2);
    arma::cx_fvec vec_adj_1_1 =
      vec_csi_0_0 % (vec_csi_2_2 % vec_csi_3_3 - vec_csi_2_3 % vec_csi_3_2) +
      vec_csi_0_2 % (vec_csi_2_3 % vec_csi_3_0 - vec_csi_2_0 % vec_csi_3_3) +
      vec_csi_0_3 % (vec_csi_2_0 % vec_csi_3_2 - vec_csi_2_2 % vec_csi_3_0);
    arma::cx_fvec vec_adj_1_2 =
      vec_csi_0_0 % (vec_csi_1_3 % vec_csi_3_2 - vec_csi_1_2 % vec_csi_3_3) +
      vec_csi_0_2 % (vec_csi_1_0 % vec_csi_3_3 - vec_csi_1_3 % vec_csi_3_0) +
      vec_csi_0_3 % (vec_csi_1_2 % vec_csi_3_0 - vec_csi_1_0 % vec_csi_3_2);
    arma::cx_fvec vec_adj_1_3 =
      vec_csi_0_0 % (vec_csi_1_2 % vec_csi_2_3 - vec_csi_1_3 % vec_csi_2_2) +
      vec_csi_0_2 % (vec_csi_1_3 % vec_csi_2_0 - vec_csi_1_0 % vec_csi_2_3) +
      vec_csi_0_3 % (vec_csi_1_0 % vec_csi_2_2 - vec_csi_1_2 % vec_csi_2_0);

    arma::cx_fvec vec_adj_2_0 =
      vec_csi_1_0 % (vec_csi_2_1 % vec_csi_3_3 - vec_csi_2_3 % vec_csi_3_1) +
      vec_csi_1_1 % (vec_csi_2_3 % vec_csi_3_0 - vec_csi_2_0 % vec_csi_3_3) +
      vec_csi_1_3 % (vec_csi_2_0 % vec_csi_3_1 - vec_csi_2_1 % vec_csi_3_0);
    arma::cx_fvec vec_adj_2_1 =
      vec_csi_0_0 % (vec_csi_2_3 % vec_csi_3_1 - vec_csi_2_1 % vec_csi_3_3) +
      vec_csi_0_1 % (vec_csi_2_0 % vec_csi_3_3 - vec_csi_2_3 % vec_csi_3_0) +
      vec_csi_0_3 % (vec_csi_2_1 % vec_csi_3_0 - vec_csi_2_0 % vec_csi_3_1);
    arma::cx_fvec vec_adj_2_2 =
      vec_csi_0_0 % (vec_csi_1_1 % vec_csi_3_3 - vec_csi_1_3 % vec_csi_3_1) +
      vec_csi_0_1 % (vec_csi_1_3 % vec_csi_3_0 - vec_csi_1_0 % vec_csi_3_3) +
      vec_csi_0_3 % (vec_csi_1_0 % vec_csi_3_1 - vec_csi_1_1 % vec_csi_3_0);
    arma::cx_fvec vec_adj_2_3 =
      vec_csi_0_0 % (vec_csi_1_3 % vec_csi_2_1 - vec_csi_1_1 % vec_csi_2_3) +
      vec_csi_0_1 % (vec_csi_1_0 % vec_csi_2_3 - vec_csi_1_3 % vec_csi_2_0) +
      vec_csi_0_3 % (vec_csi_1_1 % vec_csi_2_0 - vec_csi_1_0 % vec_csi_2_1);

    arma::cx_fvec vec_adj_3_0 =
      vec_csi_1_0 % (vec_csi_2_2 % vec_csi_3_1 - vec_csi_2_1 % vec_csi_3_2) +
      vec_csi_1_1 % (vec_csi_2_0 % vec_csi_3_2 - vec_csi_2_2 % vec_csi_3_0) +
      vec_csi_1_2 % (vec_csi_2_1 % vec_csi_3_0 - vec_csi_2_0 % vec_csi_3_1);
    arma::cx_fvec vec_adj_3_1 =
      vec_csi_0_0 % (vec_csi_2_1 % vec_csi_3_2 - vec_csi_2_2 % vec_csi_3_1) +
      vec_csi_0_1 % (vec_csi_2_2 % vec_csi_3_0 - vec_csi_2_0 % vec_csi_3_2) +
      vec_csi_0_2 % (vec_csi_2_0 % vec_csi_3_1 - vec_csi_2_1 % vec_csi_3_0);
    arma::cx_fvec vec_adj_3_2 =
      vec_csi_0_0 % (vec_csi_1_2 % vec_csi_3_1 - vec_csi_1_1 % vec_csi_3_2) +
      vec_csi_0_1 % (vec_csi_1_0 % vec_csi_3_2 - vec_csi_1_2 % vec_csi_3_0) +
      vec_csi_0_2 % (vec_csi_1_1 % vec_csi_3_0 - vec_csi_1_0 % vec_csi_3_1);
    arma::cx_fvec vec_adj_3_3 =
      vec_csi_0_0 % (vec_csi_1_1 % vec_csi_2_2 - vec_csi_1_2 % vec_csi_2_1) +
      vec_csi_0_1 % (vec_csi_1_2 % vec_csi_2_0 - vec_csi_1_0 % vec_csi_2_2) +
      vec_csi_0_2 % (vec_csi_1_0 % vec_csi_2_1 - vec_csi_1_1 % vec_csi_2_0);

    // Prepare UL beam matrix. Let Armadillo help handle the memory.
    // The format here is identical to the input of equalizer.
    // Note that the memory layout for Armadillo & AVX512 impl are different.
    arma::cx_float* ul_beam_ptr = reinterpret_cast<arma::cx_float*>(
      ul_beam_matrices_[frame_slot][cfg_->GetBeamScId(base_sc_id)]);
    // Use beam matrix layout align with the one used by AVX512 equalizer.
    arma::cx_fvec vec_ul_beam(
        (arma::cx_float*)ul_beam_ptr,
        sc_vec_len * cfg_->SpatialStreamsNum() * cfg_->BsAntNum(),
        false);
    size_t shift = sc_vec_len;

    vec_ul_beam.subvec(0, shift - 1) = vec_adj_0_0 % vec_det;
    vec_ul_beam.subvec(shift, 2 * shift - 1) = vec_adj_0_1 % vec_det;
    vec_ul_beam.subvec(2 * shift, 3 * shift - 1) = vec_adj_0_2 % vec_det;
    vec_ul_beam.subvec(3 * shift, 4 * shift - 1) = vec_adj_0_3 % vec_det;
    vec_ul_beam.subvec(4 * shift, 5 * shift - 1) = vec_adj_1_0 % vec_det;
    vec_ul_beam.subvec(5 * shift, 6 * shift - 1) = vec_adj_1_1 % vec_det;
    vec_ul_beam.subvec(6 * shift, 7 * shift - 1) = vec_adj_1_2 % vec_det;
    vec_ul_beam.subvec(7 * shift, 8 * shift - 1) = vec_adj_1_3 % vec_det;
    vec_ul_beam.subvec(8 * shift, 9 * shift - 1) = vec_adj_2_0 % vec_det;
    vec_ul_beam.subvec(9 * shift, 10 * shift - 1) = vec_adj_2_1 % vec_det;
    vec_ul_beam.subvec(10 * shift, 11 * shift - 1) = vec_adj_2_2 % vec_det;
    vec_ul_beam.subvec(11 * shift, 12 * shift - 1) = vec_adj_2_3 % vec_det;
    vec_ul_beam.subvec(12 * shift, 13 * shift - 1) = vec_adj_3_0 % vec_det;
    vec_ul_beam.subvec(13 * shift, 14 * shift - 1) = vec_adj_3_1 % vec_det;
    vec_ul_beam.subvec(14 * shift, 15 * shift - 1) = vec_adj_3_2 % vec_det;
    vec_ul_beam.subvec(15 * shift, 16 * shift - 1) = vec_adj_3_3 % vec_det;
#elif defined(ARMA_CUBE_MATOP)
    arma::cx_fcube cub_csi(cfg_->BsAntNum(), cfg_->SpatialStreamsNum(),
                           sc_vec_len);
    // Default: Handle each subcarrier one by one
    // Gather CSI = [csi_a, csi_b; csi_c, csi_d]
    // Handle each subcarrier in the block (base_sc_id : last_sc_id -1)
    for (size_t cur_sc_id = base_sc_id; cur_sc_id < last_sc_id; cur_sc_id++) {
      // Gather CSI matrices of each pilot from partially-transposed CSIs.
      arma::uvec ue_list = mac_sched_->ScheduledUeList(frame_id, cur_sc_id);
      for (size_t selected_ue_idx = 0;
          selected_ue_idx < cfg_->SpatialStreamsNum(); selected_ue_idx++) {
        size_t ue_idx = ue_list.at(selected_ue_idx);
        auto* dst_csi_ptr = reinterpret_cast<float*>(
            csi_gather_buffer_ + cfg_->BsAntNum() * selected_ue_idx);
        if (kUsePartialTrans) {
          PartialTransposeGather(
              cur_sc_id,
              reinterpret_cast<float*>(csi_buffers_[frame_slot][ue_idx]),
              dst_csi_ptr, cfg_->BsAntNum());
        } else {
          TransposeGather(
              cur_sc_id,
              reinterpret_cast<float*>(csi_buffers_[frame_slot][ue_idx]),
              dst_csi_ptr, cfg_->BsAntNum(), cfg_->OfdmDataNum());
        }
      }
      arma::cx_fmat mat_csi_temp((arma::cx_float*)csi_gather_buffer_,
                                 cfg_->BsAntNum(),
                                 cfg_->SpatialStreamsNum(), false);
      cub_csi.slice(cur_sc_id) = mat_csi_temp;
    }

    // Prepare UL beam matrix. Let Armadillo help handle the memory.
    // The format here is identical to the input of equalizer. Note 
    // that the memory layout for Armadillo (cube) & AVX512 impl are different.
    arma::cx_float* ul_beam_ptr = reinterpret_cast<arma::cx_float*>(
      ul_beam_matrices_[frame_slot][cfg_->GetBeamScId(base_sc_id)]);
    arma::cx_fcube cub_ul_beam(ul_beam_ptr, cfg_->SpatialStreamsNum(),
                               cfg_->BsAntNum(), sc_vec_len, false);

    const size_t start_tsc2 = GetTime::WorkerRdtsc();
    duration_stat_->task_duration_[1] += start_tsc2 - start_tsc1;

    // Calculate the determinant
    arma::cx_fcube cub_det =
      cub_csi.tube(0, 0) %
         (cub_csi.tube(1, 1) % (cub_csi.tube(2, 2) % cub_csi.tube(3, 3) -
                                cub_csi.tube(2, 3) % cub_csi.tube(3, 2)) +
          cub_csi.tube(1, 2) % (cub_csi.tube(2, 3) % cub_csi.tube(3, 1) -
                                cub_csi.tube(2, 1) % cub_csi.tube(3, 3)) +
          cub_csi.tube(1, 3) % (cub_csi.tube(2, 1) % cub_csi.tube(3, 2) -
                                cub_csi.tube(2, 2) % cub_csi.tube(3, 1))) -
      cub_csi.tube(1, 0) %
         (cub_csi.tube(0, 1) % (cub_csi.tube(2, 2) % cub_csi.tube(3, 3) -
                                cub_csi.tube(2, 3) % cub_csi.tube(3, 2)) +
          cub_csi.tube(0, 2) % (cub_csi.tube(2, 3) % cub_csi.tube(3, 1) -
                                cub_csi.tube(2, 1) % cub_csi.tube(3, 3)) +
          cub_csi.tube(0, 3) % (cub_csi.tube(2, 1) % cub_csi.tube(3, 2) -
                                cub_csi.tube(2, 2) % cub_csi.tube(3, 1))) +
      cub_csi.tube(2, 0) %
         (cub_csi.tube(0, 1) % (cub_csi.tube(1, 2) % cub_csi.tube(3, 3) -
                                cub_csi.tube(1, 3) % cub_csi.tube(3, 2)) +
          cub_csi.tube(0, 2) % (cub_csi.tube(1, 3) % cub_csi.tube(3, 1) -
                                cub_csi.tube(1, 1) % cub_csi.tube(3, 3)) +
          cub_csi.tube(0, 3) % (cub_csi.tube(1, 1) % cub_csi.tube(3, 2) -
                                cub_csi.tube(1, 2) % cub_csi.tube(3, 1))) -
      cub_csi.tube(3, 0) %
         (cub_csi.tube(0, 1) % (cub_csi.tube(1, 2) % cub_csi.tube(2, 3) -
                                cub_csi.tube(1, 3) % cub_csi.tube(2, 2)) +
          cub_csi.tube(0, 2) % (cub_csi.tube(1, 3) % cub_csi.tube(2, 1) -
                                cub_csi.tube(1, 1) % cub_csi.tube(2, 3)) +
          cub_csi.tube(0, 3) % (cub_csi.tube(1, 1) % cub_csi.tube(2, 2) -
                                cub_csi.tube(1, 2) % cub_csi.tube(2, 1)));

    // check if the channel matrix is invertible,
    // float lowest > 1e-38, normal range > 1e-8
    if (unlikely(cub_det.is_zero(1e-10))) {
      AGORA_LOG_WARN("Channel matrix seems not invertible\n");
    }

    // use reciprocal for division since multiplication is faster
    cub_det = 1.0 / cub_det;

    // Calculate the adjoint matrix
    cub_ul_beam.tube(0, 0) =
      cub_csi.tube(1, 1) % (cub_csi.tube(2, 2) % cub_csi.tube(3, 3) -
                            cub_csi.tube(2, 3) % cub_csi.tube(3, 2)) +
      cub_csi.tube(1, 2) % (cub_csi.tube(2, 3) % cub_csi.tube(3, 1) -
                            cub_csi.tube(2, 1) % cub_csi.tube(3, 3)) +
      cub_csi.tube(1, 3) % (cub_csi.tube(2, 1) % cub_csi.tube(3, 2) -
                            cub_csi.tube(2, 2) % cub_csi.tube(3, 1));
    cub_ul_beam.tube(0, 1) =
      cub_csi.tube(0, 1) % (cub_csi.tube(2, 3) % cub_csi.tube(3, 2) -
                            cub_csi.tube(2, 2) % cub_csi.tube(3, 3)) +
      cub_csi.tube(0, 2) % (cub_csi.tube(2, 1) % cub_csi.tube(3, 3) -
                            cub_csi.tube(2, 3) % cub_csi.tube(3, 1)) +
      cub_csi.tube(0, 3) % (cub_csi.tube(2, 2) % cub_csi.tube(3, 1) -
                            cub_csi.tube(2, 1) % cub_csi.tube(3, 2));
    cub_ul_beam.tube(0, 2) =
      cub_csi.tube(0, 1) % (cub_csi.tube(1, 2) % cub_csi.tube(3, 3) -
                            cub_csi.tube(1, 3) % cub_csi.tube(3, 2)) +
      cub_csi.tube(0, 2) % (cub_csi.tube(1, 3) % cub_csi.tube(3, 1) -
                            cub_csi.tube(1, 1) % cub_csi.tube(3, 3)) +
      cub_csi.tube(0, 3) % (cub_csi.tube(1, 1) % cub_csi.tube(3, 2) -
                            cub_csi.tube(1, 2) % cub_csi.tube(3, 1));
    cub_ul_beam.tube(0, 3) =
      cub_csi.tube(0, 1) % (cub_csi.tube(1, 3) % cub_csi.tube(2, 2) -
                            cub_csi.tube(1, 2) % cub_csi.tube(2, 3)) +
      cub_csi.tube(0, 2) % (cub_csi.tube(1, 1) % cub_csi.tube(2, 3) -
                            cub_csi.tube(1, 3) % cub_csi.tube(2, 1)) +
      cub_csi.tube(0, 3) % (cub_csi.tube(1, 2) % cub_csi.tube(2, 1) -
                            cub_csi.tube(1, 1) % cub_csi.tube(2, 2));
    cub_ul_beam.tube(1, 0) =
      cub_csi.tube(1, 0) % (cub_csi.tube(2, 3) % cub_csi.tube(3, 2) -
                            cub_csi.tube(2, 2) % cub_csi.tube(3, 3)) +
      cub_csi.tube(1, 2) % (cub_csi.tube(2, 0) % cub_csi.tube(3, 3) -
                            cub_csi.tube(2, 3) % cub_csi.tube(3, 0)) +
      cub_csi.tube(1, 3) % (cub_csi.tube(2, 2) % cub_csi.tube(3, 0) -
                            cub_csi.tube(2, 0) % cub_csi.tube(3, 2));
    cub_ul_beam.tube(1, 1) =
      cub_csi.tube(0, 0) % (cub_csi.tube(2, 2) % cub_csi.tube(3, 3) -
                            cub_csi.tube(2, 3) % cub_csi.tube(3, 2)) +
      cub_csi.tube(0, 2) % (cub_csi.tube(2, 3) % cub_csi.tube(3, 0) -
                            cub_csi.tube(2, 0) % cub_csi.tube(3, 3)) +
      cub_csi.tube(0, 3) % (cub_csi.tube(2, 0) % cub_csi.tube(3, 2) -
                            cub_csi.tube(2, 2) % cub_csi.tube(3, 0));
    cub_ul_beam.tube(1, 2) =
      cub_csi.tube(0, 0) % (cub_csi.tube(1, 3) % cub_csi.tube(3, 2) -
                            cub_csi.tube(1, 2) % cub_csi.tube(3, 3)) +
      cub_csi.tube(0, 2) % (cub_csi.tube(1, 0) % cub_csi.tube(3, 3) -
                            cub_csi.tube(1, 3) % cub_csi.tube(3, 0)) +
      cub_csi.tube(0, 3) % (cub_csi.tube(1, 2) % cub_csi.tube(3, 0) -
                            cub_csi.tube(1, 0) % cub_csi.tube(3, 2));
    cub_ul_beam.tube(1, 3) =
      cub_csi.tube(0, 0) % (cub_csi.tube(1, 2) % cub_csi.tube(2, 3) -
                            cub_csi.tube(1, 3) % cub_csi.tube(2, 2)) +
      cub_csi.tube(0, 2) % (cub_csi.tube(1, 3) % cub_csi.tube(2, 0) -
                            cub_csi.tube(1, 0) % cub_csi.tube(2, 3)) +
      cub_csi.tube(0, 3) % (cub_csi.tube(1, 0) % cub_csi.tube(2, 2) -
                            cub_csi.tube(1, 2) % cub_csi.tube(2, 0));
    cub_ul_beam.tube(2, 0) =
      cub_csi.tube(1, 0) % (cub_csi.tube(2, 1) % cub_csi.tube(3, 3) -
                            cub_csi.tube(2, 3) % cub_csi.tube(3, 1)) +
      cub_csi.tube(1, 1) % (cub_csi.tube(2, 3) % cub_csi.tube(3, 0) -
                            cub_csi.tube(2, 0) % cub_csi.tube(3, 3)) +
      cub_csi.tube(1, 3) % (cub_csi.tube(2, 0) % cub_csi.tube(3, 1) -
                            cub_csi.tube(2, 1) % cub_csi.tube(3, 0));
    cub_ul_beam.tube(2, 1) =
      cub_csi.tube(0, 0) % (cub_csi.tube(2, 3) % cub_csi.tube(3, 1) -
                            cub_csi.tube(2, 1) % cub_csi.tube(3, 3)) +
      cub_csi.tube(0, 1) % (cub_csi.tube(2, 0) % cub_csi.tube(3, 3) -
                            cub_csi.tube(2, 3) % cub_csi.tube(3, 0)) +
      cub_csi.tube(0, 3) % (cub_csi.tube(2, 1) % cub_csi.tube(3, 0) -
                            cub_csi.tube(2, 0) % cub_csi.tube(3, 1));
    cub_ul_beam.tube(2, 2) =
      cub_csi.tube(0, 0) % (cub_csi.tube(1, 1) % cub_csi.tube(3, 3) -
                            cub_csi.tube(1, 3) % cub_csi.tube(3, 1)) +
      cub_csi.tube(0, 1) % (cub_csi.tube(1, 3) % cub_csi.tube(3, 0) -
                            cub_csi.tube(1, 0) % cub_csi.tube(3, 3)) +
      cub_csi.tube(0, 3) % (cub_csi.tube(1, 0) % cub_csi.tube(3, 1) -
                            cub_csi.tube(1, 1) % cub_csi.tube(3, 0));
    cub_ul_beam.tube(2, 3) =
      cub_csi.tube(0, 0) % (cub_csi.tube(1, 3) % cub_csi.tube(2, 1) -
                            cub_csi.tube(1, 1) % cub_csi.tube(2, 3)) +
      cub_csi.tube(0, 1) % (cub_csi.tube(1, 0) % cub_csi.tube(2, 3) -
                            cub_csi.tube(1, 3) % cub_csi.tube(2, 0)) +
      cub_csi.tube(0, 3) % (cub_csi.tube(1, 1) % cub_csi.tube(2, 0) -
                            cub_csi.tube(1, 0) % cub_csi.tube(2, 1));
    cub_ul_beam.tube(3, 0) =
      cub_csi.tube(1, 0) % (cub_csi.tube(2, 2) % cub_csi.tube(3, 1) -
                            cub_csi.tube(2, 1) % cub_csi.tube(3, 2)) +
      cub_csi.tube(1, 1) % (cub_csi.tube(2, 0) % cub_csi.tube(3, 2) -
                            cub_csi.tube(2, 2) % cub_csi.tube(3, 0)) +
      cub_csi.tube(1, 2) % (cub_csi.tube(2, 1) % cub_csi.tube(3, 0) -
                            cub_csi.tube(2, 0) % cub_csi.tube(3, 1));
    cub_ul_beam.tube(3, 1) =
      cub_csi.tube(0, 0) % (cub_csi.tube(2, 1) % cub_csi.tube(3, 2) -
                            cub_csi.tube(2, 2) % cub_csi.tube(3, 1)) +
      cub_csi.tube(0, 1) % (cub_csi.tube(2, 2) % cub_csi.tube(3, 0) -
                            cub_csi.tube(2, 0) % cub_csi.tube(3, 2)) +
      cub_csi.tube(0, 2) % (cub_csi.tube(2, 0) % cub_csi.tube(3, 1) -
                            cub_csi.tube(2, 1) % cub_csi.tube(3, 0));
    cub_ul_beam.tube(3, 2) =
      cub_csi.tube(0, 0) % (cub_csi.tube(1, 2) % cub_csi.tube(3, 1) -
                            cub_csi.tube(1, 1) % cub_csi.tube(3, 2)) +
      cub_csi.tube(0, 1) % (cub_csi.tube(1, 0) % cub_csi.tube(3, 2) -
                            cub_csi.tube(1, 2) % cub_csi.tube(3, 0)) +
      cub_csi.tube(0, 2) % (cub_csi.tube(1, 1) % cub_csi.tube(3, 0) -
                            cub_csi.tube(1, 0) % cub_csi.tube(3, 1));
    cub_ul_beam.tube(3, 3) =
      cub_csi.tube(0, 0) % (cub_csi.tube(1, 1) % cub_csi.tube(2, 2) -
                            cub_csi.tube(1, 2) % cub_csi.tube(2, 1)) +
      cub_csi.tube(0, 1) % (cub_csi.tube(1, 2) % cub_csi.tube(2, 0) -
                            cub_csi.tube(1, 0) % cub_csi.tube(2, 2)) +
      cub_csi.tube(0, 2) % (cub_csi.tube(1, 0) % cub_csi.tube(2, 1) -
                            cub_csi.tube(1, 1) % cub_csi.tube(2, 0));

    // Multiply the adjoint matrix with the reciprocal of the determinant
    cub_ul_beam.tube(0, 0) = cub_ul_beam.tube(0, 0) % cub_det;
    cub_ul_beam.tube(0, 1) = cub_ul_beam.tube(0, 1) % cub_det;
    cub_ul_beam.tube(0, 2) = cub_ul_beam.tube(0, 2) % cub_det;
    cub_ul_beam.tube(0, 3) = cub_ul_beam.tube(0, 3) % cub_det;
    cub_ul_beam.tube(1, 0) = cub_ul_beam.tube(1, 0) % cub_det;
    cub_ul_beam.tube(1, 1) = cub_ul_beam.tube(1, 1) % cub_det;
    cub_ul_beam.tube(1, 2) = cub_ul_beam.tube(1, 2) % cub_det;
    cub_ul_beam.tube(1, 3) = cub_ul_beam.tube(1, 3) % cub_det;
    cub_ul_beam.tube(2, 0) = cub_ul_beam.tube(2, 0) % cub_det;
    cub_ul_beam.tube(2, 1) = cub_ul_beam.tube(2, 1) % cub_det;
    cub_ul_beam.tube(2, 2) = cub_ul_beam.tube(2, 2) % cub_det;
    cub_ul_beam.tube(2, 3) = cub_ul_beam.tube(2, 3) % cub_det;
    cub_ul_beam.tube(3, 0) = cub_ul_beam.tube(3, 0) % cub_det;
    cub_ul_beam.tube(3, 1) = cub_ul_beam.tube(3, 1) % cub_det;
    cub_ul_beam.tube(3, 2) = cub_ul_beam.tube(3, 2) % cub_det;
    cub_ul_beam.tube(3, 3) = cub_ul_beam.tube(3, 3) % cub_det;
#else
    // Prepare CSI matrix. Read in vectors.
    auto* cx_src =
      reinterpret_cast<complex_float*>(csi_buffers_[frame_slot][0]);
    arma::cx_fvec vec_csi_0_0 = arma::cx_fvec(
      (arma::cx_float*)(cx_src + 0 * cfg_->OfdmDataNum()), sc_vec_len, false);
    arma::cx_fvec vec_csi_1_0 = arma::cx_fvec(
      (arma::cx_float*)(cx_src + 1 * cfg_->OfdmDataNum()), sc_vec_len, false);
    arma::cx_fvec vec_csi_2_0 = arma::cx_fvec(
      (arma::cx_float*)(cx_src + 2 * cfg_->OfdmDataNum()), sc_vec_len, false);
    arma::cx_fvec vec_csi_3_0 = arma::cx_fvec(
      (arma::cx_float*)(cx_src + 3 * cfg_->OfdmDataNum()), sc_vec_len, false);

    cx_src = reinterpret_cast<complex_float*>(csi_buffers_[frame_slot][1]);
    arma::cx_fvec vec_csi_0_1 = arma::cx_fvec(
      (arma::cx_float*)(cx_src + 0 * cfg_->OfdmDataNum()), sc_vec_len, false);
    arma::cx_fvec vec_csi_1_1 = arma::cx_fvec(
      (arma::cx_float*)(cx_src + 1 * cfg_->OfdmDataNum()), sc_vec_len, false);
    arma::cx_fvec vec_csi_2_1 = arma::cx_fvec(
      (arma::cx_float*)(cx_src + 2 * cfg_->OfdmDataNum()), sc_vec_len, false);
    arma::cx_fvec vec_csi_3_1 = arma::cx_fvec(
      (arma::cx_float*)(cx_src + 3 * cfg_->OfdmDataNum()), sc_vec_len, false);

    cx_src = reinterpret_cast<complex_float*>(csi_buffers_[frame_slot][2]);
    arma::cx_fvec vec_csi_0_2 = arma::cx_fvec(
      (arma::cx_float*)(cx_src + 0 * cfg_->OfdmDataNum()), sc_vec_len, false);
    arma::cx_fvec vec_csi_1_2 = arma::cx_fvec(
      (arma::cx_float*)(cx_src + 1 * cfg_->OfdmDataNum()), sc_vec_len, false);
    arma::cx_fvec vec_csi_2_2 = arma::cx_fvec(
      (arma::cx_float*)(cx_src + 2 * cfg_->OfdmDataNum()), sc_vec_len, false);
    arma::cx_fvec vec_csi_3_2 = arma::cx_fvec(
      (arma::cx_float*)(cx_src + 3 * cfg_->OfdmDataNum()), sc_vec_len, false);

    cx_src = reinterpret_cast<complex_float*>(csi_buffers_[frame_slot][3]);
    arma::cx_fvec vec_csi_0_3 = arma::cx_fvec(
      (arma::cx_float*)(cx_src + 0 * cfg_->OfdmDataNum()), sc_vec_len, false);
    arma::cx_fvec vec_csi_1_3 = arma::cx_fvec(
      (arma::cx_float*)(cx_src + 1 * cfg_->OfdmDataNum()), sc_vec_len, false);
    arma::cx_fvec vec_csi_2_3 = arma::cx_fvec(
      (arma::cx_float*)(cx_src + 2 * cfg_->OfdmDataNum()), sc_vec_len, false);
    arma::cx_fvec vec_csi_3_3 = arma::cx_fvec(
      (arma::cx_float*)(cx_src + 3 * cfg_->OfdmDataNum()), sc_vec_len, false);

    const size_t start_tsc2 = GetTime::WorkerRdtsc();
    duration_stat_->task_duration_[1] += start_tsc2 - start_tsc1;

    // Calculate the determinant
    arma::cx_fvec vec_det =
      vec_csi_0_0 %
       (vec_csi_1_1 % (vec_csi_2_2 % vec_csi_3_3 - vec_csi_2_3 % vec_csi_3_2) +
        vec_csi_1_2 % (vec_csi_2_3 % vec_csi_3_1 - vec_csi_2_1 % vec_csi_3_3) +
        vec_csi_1_3 % (vec_csi_2_1 % vec_csi_3_2 - vec_csi_2_2 % vec_csi_3_1)) -
      vec_csi_1_0 %
       (vec_csi_0_1 % (vec_csi_2_2 % vec_csi_3_3 - vec_csi_2_3 % vec_csi_3_2) +
        vec_csi_0_2 % (vec_csi_2_3 % vec_csi_3_1 - vec_csi_2_1 % vec_csi_3_3) +
        vec_csi_0_3 % (vec_csi_2_1 % vec_csi_3_2 - vec_csi_2_2 % vec_csi_3_1)) +
      vec_csi_2_0 %
       (vec_csi_0_1 % (vec_csi_1_2 % vec_csi_3_3 - vec_csi_1_3 % vec_csi_3_2) +
        vec_csi_0_2 % (vec_csi_1_3 % vec_csi_3_1 - vec_csi_1_1 % vec_csi_3_3) +
        vec_csi_0_3 % (vec_csi_1_1 % vec_csi_3_2 - vec_csi_1_2 % vec_csi_3_1)) -
      vec_csi_3_0 %
       (vec_csi_0_1 % (vec_csi_1_2 % vec_csi_2_3 - vec_csi_1_3 % vec_csi_2_2) +
        vec_csi_0_2 % (vec_csi_1_3 % vec_csi_2_1 - vec_csi_1_1 % vec_csi_2_3) +
        vec_csi_0_3 % (vec_csi_1_1 % vec_csi_2_2 - vec_csi_1_2 % vec_csi_2_1));

    if (unlikely(arma::any(arma::abs(vec_det) < 1e-10))) {
      AGORA_LOG_WARN("Channel matrix seems not invertible\n");
    }

    // use reciprocal for division since multiplication is faster
    vec_det = 1.0 / vec_det;

    arma::cx_fvec vec_adj_0_0 =
      vec_csi_1_1 % (vec_csi_2_2 % vec_csi_3_3 - vec_csi_2_3 % vec_csi_3_2) +
      vec_csi_1_2 % (vec_csi_2_3 % vec_csi_3_1 - vec_csi_2_1 % vec_csi_3_3) +
      vec_csi_1_3 % (vec_csi_2_1 % vec_csi_3_2 - vec_csi_2_2 % vec_csi_3_1);
    arma::cx_fvec vec_adj_0_1 =
      vec_csi_0_1 % (vec_csi_2_3 % vec_csi_3_2 - vec_csi_2_2 % vec_csi_3_3) +
      vec_csi_0_2 % (vec_csi_2_1 % vec_csi_3_3 - vec_csi_2_3 % vec_csi_3_1) +
      vec_csi_0_3 % (vec_csi_2_2 % vec_csi_3_1 - vec_csi_2_1 % vec_csi_3_2);
    arma::cx_fvec vec_adj_0_2 =
      vec_csi_0_1 % (vec_csi_1_2 % vec_csi_3_3 - vec_csi_1_3 % vec_csi_3_2) +
      vec_csi_0_2 % (vec_csi_1_3 % vec_csi_3_1 - vec_csi_1_1 % vec_csi_3_3) +
      vec_csi_0_3 % (vec_csi_1_1 % vec_csi_3_2 - vec_csi_1_2 % vec_csi_3_1);
    arma::cx_fvec vec_adj_0_3 =
      vec_csi_0_1 % (vec_csi_1_3 % vec_csi_2_2 - vec_csi_1_2 % vec_csi_2_3) +
      vec_csi_0_2 % (vec_csi_1_1 % vec_csi_2_3 - vec_csi_1_3 % vec_csi_2_1) +
      vec_csi_0_3 % (vec_csi_1_2 % vec_csi_2_1 - vec_csi_1_1 % vec_csi_2_2);

    arma::cx_fvec vec_adj_1_0 =
      vec_csi_1_0 % (vec_csi_2_3 % vec_csi_3_2 - vec_csi_2_2 % vec_csi_3_3) +
      vec_csi_1_2 % (vec_csi_2_0 % vec_csi_3_3 - vec_csi_2_3 % vec_csi_3_0) +
      vec_csi_1_3 % (vec_csi_2_2 % vec_csi_3_0 - vec_csi_2_0 % vec_csi_3_2);
    arma::cx_fvec vec_adj_1_1 =
      vec_csi_0_0 % (vec_csi_2_2 % vec_csi_3_3 - vec_csi_2_3 % vec_csi_3_2) +
      vec_csi_0_2 % (vec_csi_2_3 % vec_csi_3_0 - vec_csi_2_0 % vec_csi_3_3) +
      vec_csi_0_3 % (vec_csi_2_0 % vec_csi_3_2 - vec_csi_2_2 % vec_csi_3_0);
    arma::cx_fvec vec_adj_1_2 =
      vec_csi_0_0 % (vec_csi_1_3 % vec_csi_3_2 - vec_csi_1_2 % vec_csi_3_3) +
      vec_csi_0_2 % (vec_csi_1_0 % vec_csi_3_3 - vec_csi_1_3 % vec_csi_3_0) +
      vec_csi_0_3 % (vec_csi_1_2 % vec_csi_3_0 - vec_csi_1_0 % vec_csi_3_2);
    arma::cx_fvec vec_adj_1_3 =
      vec_csi_0_0 % (vec_csi_1_2 % vec_csi_2_3 - vec_csi_1_3 % vec_csi_2_2) +
      vec_csi_0_2 % (vec_csi_1_3 % vec_csi_2_0 - vec_csi_1_0 % vec_csi_2_3) +
      vec_csi_0_3 % (vec_csi_1_0 % vec_csi_2_2 - vec_csi_1_2 % vec_csi_2_0);

    arma::cx_fvec vec_adj_2_0 =
      vec_csi_1_0 % (vec_csi_2_1 % vec_csi_3_3 - vec_csi_2_3 % vec_csi_3_1) +
      vec_csi_1_1 % (vec_csi_2_3 % vec_csi_3_0 - vec_csi_2_0 % vec_csi_3_3) +
      vec_csi_1_3 % (vec_csi_2_0 % vec_csi_3_1 - vec_csi_2_1 % vec_csi_3_0);
    arma::cx_fvec vec_adj_2_1 =
      vec_csi_0_0 % (vec_csi_2_3 % vec_csi_3_1 - vec_csi_2_1 % vec_csi_3_3) +
      vec_csi_0_1 % (vec_csi_2_0 % vec_csi_3_3 - vec_csi_2_3 % vec_csi_3_0) +
      vec_csi_0_3 % (vec_csi_2_1 % vec_csi_3_0 - vec_csi_2_0 % vec_csi_3_1);
    arma::cx_fvec vec_adj_2_2 =
      vec_csi_0_0 % (vec_csi_1_1 % vec_csi_3_3 - vec_csi_1_3 % vec_csi_3_1) +
      vec_csi_0_1 % (vec_csi_1_3 % vec_csi_3_0 - vec_csi_1_0 % vec_csi_3_3) +
      vec_csi_0_3 % (vec_csi_1_0 % vec_csi_3_1 - vec_csi_1_1 % vec_csi_3_0);
    arma::cx_fvec vec_adj_2_3 =
      vec_csi_0_0 % (vec_csi_1_3 % vec_csi_2_1 - vec_csi_1_1 % vec_csi_2_3) +
      vec_csi_0_1 % (vec_csi_1_0 % vec_csi_2_3 - vec_csi_1_3 % vec_csi_2_0) +
      vec_csi_0_3 % (vec_csi_1_1 % vec_csi_2_0 - vec_csi_1_0 % vec_csi_2_1);

    arma::cx_fvec vec_adj_3_0 =
      vec_csi_1_0 % (vec_csi_2_2 % vec_csi_3_1 - vec_csi_2_1 % vec_csi_3_2) +
      vec_csi_1_1 % (vec_csi_2_0 % vec_csi_3_2 - vec_csi_2_2 % vec_csi_3_0) +
      vec_csi_1_2 % (vec_csi_2_1 % vec_csi_3_0 - vec_csi_2_0 % vec_csi_3_1);
    arma::cx_fvec vec_adj_3_1 =
      vec_csi_0_0 % (vec_csi_2_1 % vec_csi_3_2 - vec_csi_2_2 % vec_csi_3_1) +
      vec_csi_0_1 % (vec_csi_2_2 % vec_csi_3_0 - vec_csi_2_0 % vec_csi_3_2) +
      vec_csi_0_2 % (vec_csi_2_0 % vec_csi_3_1 - vec_csi_2_1 % vec_csi_3_0);
    arma::cx_fvec vec_adj_3_2 =
      vec_csi_0_0 % (vec_csi_1_2 % vec_csi_3_1 - vec_csi_1_1 % vec_csi_3_2) +
      vec_csi_0_1 % (vec_csi_1_0 % vec_csi_3_2 - vec_csi_1_2 % vec_csi_3_0) +
      vec_csi_0_2 % (vec_csi_1_1 % vec_csi_3_0 - vec_csi_1_0 % vec_csi_3_1);
    arma::cx_fvec vec_adj_3_3 =
      vec_csi_0_0 % (vec_csi_1_1 % vec_csi_2_2 - vec_csi_1_2 % vec_csi_2_1) +
      vec_csi_0_1 % (vec_csi_1_2 % vec_csi_2_0 - vec_csi_1_0 % vec_csi_2_2) +
      vec_csi_0_2 % (vec_csi_1_0 % vec_csi_2_1 - vec_csi_1_1 % vec_csi_2_0);

    // Prepare UL beam matrix. Let Armadillo help handle the memory.
    // The format here is identical to the input of equalizer.
    // Note that the memory layout for Armadillo & AVX512 impl are different.
    arma::cx_float* ul_beam_ptr = reinterpret_cast<arma::cx_float*>(
      ul_beam_matrices_[frame_slot][cfg_->GetBeamScId(base_sc_id)]);
    arma::cx_fcube cub_ul_beam(ul_beam_ptr, cfg_->SpatialStreamsNum(),
                               cfg_->BsAntNum(), sc_vec_len, false);

    cub_ul_beam.tube(0, 0) = vec_adj_0_0 % vec_det;
    cub_ul_beam.tube(0, 1) = vec_adj_0_1 % vec_det;
    cub_ul_beam.tube(0, 2) = vec_adj_0_2 % vec_det;
    cub_ul_beam.tube(0, 3) = vec_adj_0_3 % vec_det;
    cub_ul_beam.tube(1, 0) = vec_adj_1_0 % vec_det;
    cub_ul_beam.tube(1, 1) = vec_adj_1_1 % vec_det;
    cub_ul_beam.tube(1, 2) = vec_adj_1_2 % vec_det;
    cub_ul_beam.tube(1, 3) = vec_adj_1_3 % vec_det;
    cub_ul_beam.tube(2, 0) = vec_adj_2_0 % vec_det;
    cub_ul_beam.tube(2, 1) = vec_adj_2_1 % vec_det;
    cub_ul_beam.tube(2, 2) = vec_adj_2_2 % vec_det;
    cub_ul_beam.tube(2, 3) = vec_adj_2_3 % vec_det;
    cub_ul_beam.tube(3, 0) = vec_adj_3_0 % vec_det;
    cub_ul_beam.tube(3, 1) = vec_adj_3_1 % vec_det;
    cub_ul_beam.tube(3, 2) = vec_adj_3_2 % vec_det;
    cub_ul_beam.tube(3, 3) = vec_adj_3_3 % vec_det;

    // // Transform default beam matrix layout to the one used by AVX512 equalizer.
    // arma::cx_fvec vec_ul_beam(
    //     (arma::cx_float*)ul_beam_ptr,
    //     sc_vec_len * cfg_->SpatialStreamsNum() * cfg_->BsAntNum(),
    //     false);
    // size_t shift = sc_vec_len;

    // vec_ul_beam.subvec(0, shift - 1) = vec_adj_0_0 % vec_det;
    // vec_ul_beam.subvec(shift, 2 * shift - 1) = vec_adj_0_1 % vec_det;
    // vec_ul_beam.subvec(2 * shift, 3 * shift - 1) = vec_adj_0_2 % vec_det;
    // vec_ul_beam.subvec(3 * shift, 4 * shift - 1) = vec_adj_0_3 % vec_det;
    // vec_ul_beam.subvec(4 * shift, 5 * shift - 1) = vec_adj_1_0 % vec_det;
    // vec_ul_beam.subvec(5 * shift, 6 * shift - 1) = vec_adj_1_1 % vec_det;
    // vec_ul_beam.subvec(6 * shift, 7 * shift - 1) = vec_adj_1_2 % vec_det;
    // vec_ul_beam.subvec(7 * shift, 8 * shift - 1) = vec_adj_1_3 % vec_det;
    // vec_ul_beam.subvec(8 * shift, 9 * shift - 1) = vec_adj_2_0 % vec_det;
    // vec_ul_beam.subvec(9 * shift, 10 * shift - 1) = vec_adj_2_1 % vec_det;
    // vec_ul_beam.subvec(10 * shift, 11 * shift - 1) = vec_adj_2_2 % vec_det;
    // vec_ul_beam.subvec(11 * shift, 12 * shift - 1) = vec_adj_2_3 % vec_det;
    // vec_ul_beam.subvec(12 * shift, 13 * shift - 1) = vec_adj_3_0 % vec_det;
    // vec_ul_beam.subvec(13 * shift, 14 * shift - 1) = vec_adj_3_1 % vec_det;
    // vec_ul_beam.subvec(14 * shift, 15 * shift - 1) = vec_adj_3_2 % vec_det;
    // vec_ul_beam.subvec(15 * shift, 16 * shift - 1) = vec_adj_3_3 % vec_det;
#endif

    duration_stat_->task_duration_[2] += GetTime::WorkerRdtsc() - start_tsc2;
    duration_stat_->task_count_++;
    duration_stat_->task_duration_[0] += GetTime::WorkerRdtsc() - start_tsc1;

    return;
  } // end special 1x1, 2x2 or 4x4 cases

  // Default: Handle each subcarrier one by one
  size_t sc_inc = 1;
  size_t start_sc = base_sc_id;

  // When grouping sc, we can skip all sc except sc % PilotScGroupSize == 0
  if (cfg_->GroupPilotSc()) {
    // When grouping sc only process the first sc in each group
    sc_inc = cfg_->PilotScGroupSize();
    const size_t rem = start_sc % cfg_->PilotScGroupSize();
    if (rem != 0) {
      //Start at the next multiple of PilotScGroupSize
      start_sc += (cfg_->PilotScGroupSize() - rem);
    }
  }

  // Handle each subcarrier in the block (base_sc_id : last_sc_id -1)
  for (size_t cur_sc_id = start_sc; cur_sc_id < last_sc_id;
       cur_sc_id = cur_sc_id + sc_inc) {
    arma::cx_fvec& cal_sc_vec = *calib_sc_vec_ptr_;
    const size_t start_tsc1 = GetTime::WorkerRdtsc();

    // Gather CSI matrices of each pilot from partially-transposed CSIs.
    arma::uvec ue_list = mac_sched_->ScheduledUeList(frame_id, cur_sc_id);
    for (size_t selected_ue_idx = 0;
         selected_ue_idx < cfg_->SpatialStreamsNum(); selected_ue_idx++) {
      size_t ue_idx = ue_list.at(selected_ue_idx);
      auto* dst_csi_ptr = reinterpret_cast<float*>(
          csi_gather_buffer_ + cfg_->BsAntNum() * selected_ue_idx);
      if (kUsePartialTrans) {
        PartialTransposeGather(
            cur_sc_id,
            reinterpret_cast<float*>(csi_buffers_[frame_slot][ue_idx]),
            dst_csi_ptr, cfg_->BsAntNum());
      } else {
        TransposeGather(
            cur_sc_id,
            reinterpret_cast<float*>(csi_buffers_[frame_slot][ue_idx]),
            dst_csi_ptr, cfg_->BsAntNum(), cfg_->OfdmDataNum());
      }
    }

    const size_t start_tsc2 = GetTime::WorkerRdtsc();
    duration_stat_->task_duration_[1] += start_tsc2 - start_tsc1;

    arma::cx_fmat mat_csi((arma::cx_float*)csi_gather_buffer_, cfg_->BsAntNum(),
                          cfg_->SpatialStreamsNum(), false);

    if (cfg_->Frame().NumDLSyms() > 0) {
      ComputeCalib(frame_id, cur_sc_id, cal_sc_vec);
    }
    if (num_ext_ref_ > 0) {
      mat_csi.shed_rows(ext_ref_id_);
    }

    const size_t start_tsc3 = GetTime::WorkerRdtsc();
    duration_stat_->task_duration_[2] += start_tsc3 - start_tsc2;

    float noise = 0;
    if (cfg_->BeamformingAlgo() == CommsLib::BeamformingAlgorithm::kMMSE) {
      noise = phy_stats_->GetNoise(frame_id, ue_list);
    }
    ComputePrecoder(frame_id, cur_sc_id, mat_csi, cal_sc_vec, noise,
                    ul_beam_matrices_[frame_slot][cur_sc_id],
                    dl_beam_matrices_[frame_slot][cur_sc_id]);

    duration_stat_->task_duration_[3] += GetTime::WorkerRdtsc() - start_tsc3;
    duration_stat_->task_count_++;
    duration_stat_->task_duration_[0] += GetTime::WorkerRdtsc() - start_tsc1;
  }
}
