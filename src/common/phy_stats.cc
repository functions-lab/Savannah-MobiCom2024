/**
 * @file phy_stats.cc
 * @brief Declaration file for the PhyStats class.
 */
#include "phy_stats.h"

#include <cfloat>
#include <cmath>

#include "logger.h"

PhyStats::PhyStats(Config* const cfg, Direction dir)
    : config_(cfg),
      dir_(dir),
      logger_plt_snr_(CsvLog::kPltSnr, cfg, dir, true),
      logger_plt_rssi_(CsvLog::kPltRssi, cfg, dir, true),
      logger_plt_noise_(CsvLog::kPltNoise, cfg, dir, true),
      logger_bf_snr_(CsvLog::kBfSnr, cfg, dir),
      logger_bf_rssi_(CsvLog::kBfRssi, cfg, dir),
      logger_bf_noise_(CsvLog::kBfNoise, cfg, dir),
      logger_evm_(CsvLog::kEvm, cfg, dir),
      logger_evm_sc_(CsvLog::kEvmSc, cfg, dir),
      logger_evm_snr_(CsvLog::kEvmSnr, cfg, dir),
      logger_ber_(CsvLog::kBer, cfg, dir),
      logger_ser_(CsvLog::kSer, cfg, dir),
      logger_csi_(CsvLog::kCsi, cfg, dir),
      logger_calib_(CsvLog::kCalib, cfg, dir),
      logger_ul_csi_(CsvLog::kUlCsi, cfg, dir),
      logger_dl_csi_(CsvLog::kDlCsi, cfg, dir),
      logger_ul_beam_(CsvLog::kUlBeam, cfg, dir),
      logger_dl_beam_(CsvLog::kDlBeam, cfg, dir) {
  if (dir_ == Direction::kDownlink) {
    num_rx_symbols_ = cfg->Frame().NumDLSyms();
    num_rxdata_symbols_ = cfg->Frame().NumDlDataSyms();
  } else {
    num_rx_symbols_ = cfg->Frame().NumULSyms();
    num_rxdata_symbols_ = cfg->Frame().NumUlDataSyms();
  }
  const size_t task_buffer_symbol_num = num_rx_symbols_ * kFrameWnd;

  decoded_bits_count_.Calloc(cfg->UeAntNum(), task_buffer_symbol_num,
                             Agora_memory::Alignment_t::kAlign64);
  bit_error_count_.Calloc(cfg->UeAntNum(), task_buffer_symbol_num,
                          Agora_memory::Alignment_t::kAlign64);
  frame_decoded_bits_.Calloc(cfg->UeAntNum(), kFrameWnd,
                             Agora_memory::Alignment_t::kAlign64);
  frame_bit_errors_.Calloc(cfg->UeAntNum(), kFrameWnd,
                           Agora_memory::Alignment_t::kAlign64);

  decoded_blocks_count_.Calloc(cfg->UeAntNum(), task_buffer_symbol_num,
                               Agora_memory::Alignment_t::kAlign64);
  block_error_count_.Calloc(cfg->UeAntNum(), task_buffer_symbol_num,
                            Agora_memory::Alignment_t::kAlign64);
  frame_symbol_errors_.Calloc(cfg->UeAntNum(), kFrameWnd,
                              Agora_memory::Alignment_t::kAlign64);
  frame_decoded_symbols_.Calloc(cfg->UeAntNum(), kFrameWnd,
                                Agora_memory::Alignment_t::kAlign64);

  uncoded_bits_count_.Calloc(cfg->UeAntNum(), task_buffer_symbol_num,
                             Agora_memory::Alignment_t::kAlign64);
  uncoded_bit_error_count_.Calloc(cfg->UeAntNum(), task_buffer_symbol_num,
                                  Agora_memory::Alignment_t::kAlign64);

  evm_buffer_.Calloc(kFrameWnd, cfg->UeAntNum(),
                     Agora_memory::Alignment_t::kAlign64);
  evm_sc_buffer_.Calloc(kFrameWnd, cfg->UeAntNum() * cfg->OfdmDataNum(),
                        Agora_memory::Alignment_t::kAlign64);

  if (num_rxdata_symbols_ > 0) {
    gt_cube_ = arma::cx_fcube(cfg->UeAntNum(), cfg->OfdmDataNum(),
                              num_rxdata_symbols_);
    for (size_t i = 0; i < num_rxdata_symbols_; i++) {
      auto* iq_f_ptr = reinterpret_cast<arma::cx_float*>(
          (dir_ == Direction::kDownlink)
              ? cfg->DlIqF()[cfg->Frame().ClientDlPilotSymbols() + i]
              : cfg->UlIqF()[cfg->Frame().ClientUlPilotSymbols() + i]);
      arma::cx_fmat iq_f_mat(iq_f_ptr, cfg->OfdmDataNum(), cfg->UeAntNum(),
                             false);
      gt_cube_.slice(i) = iq_f_mat.st();
    }
  }
  dl_pilot_snr_.Calloc(kFrameWnd,
                       cfg->UeAntNum() * cfg->Frame().ClientDlPilotSymbols(),
                       Agora_memory::Alignment_t::kAlign64);
  dl_pilot_rssi_.Calloc(kFrameWnd,
                        cfg->UeAntNum() * cfg->Frame().ClientDlPilotSymbols(),
                        Agora_memory::Alignment_t::kAlign64);
  dl_pilot_noise_.Calloc(kFrameWnd,
                         cfg->UeAntNum() * cfg->Frame().ClientDlPilotSymbols(),
                         Agora_memory::Alignment_t::kAlign64);
  pilot_snr_.Calloc(kFrameWnd, cfg->UeAntNum() * cfg->BsAntNum(),
                    Agora_memory::Alignment_t::kAlign64);
  pilot_rssi_.Calloc(kFrameWnd, cfg->UeAntNum() * cfg->BsAntNum(),
                     Agora_memory::Alignment_t::kAlign64);
  pilot_noise_.Calloc(kFrameWnd, cfg->UeAntNum() * cfg->BsAntNum(),
                      Agora_memory::Alignment_t::kAlign64);
  calib_pilot_snr_.Calloc(kFrameWnd, 2 * cfg->BsAntNum(),
                          Agora_memory::Alignment_t::kAlign64);
  csi_cond_.Calloc(kFrameWnd, cfg->OfdmDataNum(),
                   Agora_memory::Alignment_t::kAlign64);
}

PhyStats::~PhyStats() {
  decoded_bits_count_.Free();
  bit_error_count_.Free();

  frame_decoded_bits_.Free();
  frame_bit_errors_.Free();

  decoded_blocks_count_.Free();
  block_error_count_.Free();

  frame_symbol_errors_.Free();
  frame_decoded_symbols_.Free();

  uncoded_bits_count_.Free();
  uncoded_bit_error_count_.Free();

  evm_buffer_.Free();
  evm_sc_buffer_.Free();
  csi_cond_.Free();

  calib_pilot_snr_.Free();
  pilot_snr_.Free();
  pilot_rssi_.Free();
  pilot_noise_.Free();
  dl_pilot_snr_.Free();
  dl_pilot_rssi_.Free();
  dl_pilot_noise_.Free();
}

void PhyStats::PrintPhyStats() {
  const size_t task_buffer_symbol_num = num_rx_symbols_ * kFrameWnd;
  std::string tx_type;
  if (dir_ == Direction::kDownlink) {
    tx_type = "Downlink";
  } else {
    tx_type = "Uplink";
  }

  if (num_rx_symbols_ > 0) {
    for (size_t ue_id = 0; ue_id < this->config_->UeAntNum(); ue_id++) {
      size_t total_decoded_bits(0);
      size_t total_bit_errors(0);
      size_t total_decoded_blocks(0);
      size_t total_block_errors(0);

      for (size_t i = 0u; i < task_buffer_symbol_num; i++) {
        total_decoded_bits += decoded_bits_count_[ue_id][i];
        total_bit_errors += bit_error_count_[ue_id][i];
        total_decoded_blocks += decoded_blocks_count_[ue_id][i];
        total_block_errors += block_error_count_[ue_id][i];
      }

      AGORA_LOG_INFO(
          "UE %zu: %s block errors (BLER) "
          "%zu/%zu (%f)\n",
          ue_id, tx_type.c_str(),
          total_block_errors, total_decoded_blocks,
          static_cast<float>(total_block_errors) /
              static_cast<float>(total_decoded_blocks));
    }
  }
}

void PhyStats::PrintEvmStats(size_t frame_id, const arma::uvec& ue_list) {
  arma::fmat evm_buf(evm_buffer_[frame_id % kFrameWnd], config_->UeAntNum(), 1,
                     false);
  arma::fmat evm_mat =
      evm_buf.st() / (config_->OfdmDataNum() * num_rxdata_symbols_);

  [[maybe_unused]] std::stringstream ss;
  ss << "Frame " << frame_id << ", Scheduled User(s): \n  " << ue_list.st()
     << "  EVM " << (100.0f * evm_mat(ue_list).st()) << "  SNR "
     << (-10.0f * arma::log10(evm_mat(ue_list).st()));
  AGORA_LOG_INFO("%s\n", ss.str().c_str());
}

float PhyStats::GetEvmSnr(size_t frame_id, size_t ue_id) {
  float evm = evm_buffer_[frame_id % kFrameWnd][ue_id];
  evm = evm / config_->OfdmDataNum();
  return (-10.0f * std::log10(evm));
}

void PhyStats::ClearEvmBuffer(size_t frame_id) {
  for (size_t i = 0; i < config_->UeAntNum(); i++) {
    evm_buffer_[frame_id % kFrameWnd][i] = 0.0f;
  }
}

void PhyStats::PrintDlSnrStats(size_t frame_id, const arma::uvec& ue_list) {
  [[maybe_unused]] std::stringstream ss;
  ss << "Frame " << frame_id << " DL Pilot SNR (dB) at " << std::fixed
     << std::setw(5) << std::setprecision(1);
  size_t dl_pilots_num = config_->Frame().ClientDlPilotSymbols();
  for (const auto& i : ue_list) {
    //for (size_t i = 0; i < config_->UeAntNum(); i++) {
    ss << "UE Antenna " << i << ": [ ";
    for (size_t j = 0; j < dl_pilots_num; j++) {
      float frame_snr =
          dl_pilot_snr_[frame_id % kFrameWnd][i * dl_pilots_num + j];
      ss << frame_snr << " ";
    }
    ss << "] ";
  }
  ss << std::endl;
  AGORA_LOG_INFO("%s", ss.str().c_str());
}

void PhyStats::PrintUlSnrStats(size_t frame_id) {
  [[maybe_unused]] std::stringstream ss;
  ss << "Frame " << frame_id
     << " Pilot Signal SNR (dB) Range at BS Antennas: " << std::fixed
     << std::setw(5) << std::setprecision(1);
  for (size_t i = 0; i < config_->UeAntNum(); i++) {
    float max_snr = FLT_MIN;
    float min_snr = FLT_MAX;
    size_t min_snr_id = 0;
    const float* frame_snr =
        &pilot_snr_[frame_id % kFrameWnd][i * config_->BsAntNum()];
    for (size_t j = 0; j < config_->BsAntNum(); j++) {
      const size_t radio_id = j / config_->NumChannels();
      const size_t cell_id = config_->CellId().at(radio_id);
      if (config_->ExternalRefNode(cell_id) == true &&
          radio_id == config_->RefRadio(cell_id)) {
        continue;
      }
      if (frame_snr[j] < min_snr) {
        min_snr = frame_snr[j];
        min_snr_id = j;
      }
      if (frame_snr[j] > max_snr) {
        max_snr = frame_snr[j];
      }
    }
    if (min_snr == FLT_MAX) {
      min_snr = -100;
    }
    if (max_snr == FLT_MIN) {
      max_snr = -100;
    }
    ss << "User " << i << ": [" << min_snr << "," << max_snr << "]"
       << " ";
    if (max_snr - min_snr > 20 && min_snr < 0) {
      ss << "(Possible bad antenna " << min_snr_id << ") ";
    }
  }
  ss << std::endl;
  AGORA_LOG_INFO("%s", ss.str().c_str());
}

void PhyStats::PrintCalibSnrStats(size_t frame_id) {
  [[maybe_unused]] std::stringstream ss;
  ss << "Cal Index " << frame_id
     << " Calibration Pilot Signal SNR (dB) Range at BS Antennas: "
     << std::fixed << std::setw(5) << std::setprecision(1);
  for (size_t i = 0; i < 2; i++) {
    float max_snr = FLT_MIN;
    float min_snr = FLT_MAX;
    const float* frame_snr =
        &calib_pilot_snr_[frame_id % kFrameWnd][i * config_->BsAntNum()];
    for (size_t j = 0; j < config_->BsAntNum(); j++) {
      const size_t radio_id = j / config_->NumChannels();
      const size_t cell_id = config_->CellId().at(radio_id);
      if (config_->ExternalRefNode(cell_id) == true &&
          radio_id == config_->RefRadio(cell_id)) {
        continue;
      }
      if (frame_snr[j] < min_snr) {
        min_snr = frame_snr[j];
      }
      if (frame_snr[j] > max_snr) {
        max_snr = frame_snr[j];
      }
    }
    if (min_snr == FLT_MAX) {
      min_snr = -100;
    }
    if (max_snr == FLT_MIN) {
      max_snr = -100;
    }
    if (i == 0) {
      ss << "Downlink ";
    } else {
      ss << "Uplink ";
    }
    ss << ": [" << min_snr << "," << max_snr << "] ";
  }
  ss << std::endl;
  AGORA_LOG_INFO("%s", ss.str().c_str());
}

void PhyStats::RecordPilotSnr(size_t frame_id) {
  if (kEnableCsvLog) {
    std::stringstream ss_snr;
    std::stringstream ss_rssi;
    std::stringstream ss_noise;
    ss_snr << frame_id;
    ss_rssi << frame_id;
    ss_noise << frame_id;
    const size_t frame_slot = frame_id % kFrameWnd;
    for (size_t i = 0; i < config_->UeAntNum(); i++) {
      for (size_t j = 0; j < config_->BsAntNum(); j++) {
        const size_t idx_offset = i * config_->BsAntNum() + j;
        ss_snr << "," << pilot_snr_[frame_slot][idx_offset];
        ss_rssi << "," << pilot_rssi_[frame_slot][idx_offset];
        ss_noise << "," << pilot_noise_[frame_slot][idx_offset];
      }
    }
    logger_plt_snr_.Write(ss_snr.str());
    logger_plt_rssi_.Write(ss_rssi.str());
    logger_plt_noise_.Write(ss_noise.str());
  }
}

void PhyStats::RecordCsiCond(size_t frame_id, size_t num_rec_sc) {
  if (kEnableCsvLog) {
    std::stringstream ss;
    ss << frame_id;
    const size_t sc_step = config_->OfdmDataNum() / num_rec_sc;
    const size_t sc_offset = sc_step / 2;
    for (size_t sc_rec = 0; sc_rec < num_rec_sc; sc_rec++) {
      const size_t sc_id = sc_rec * sc_step + sc_offset;
      ss << "," << (csi_cond_[frame_id % kFrameWnd][sc_id]);
    }
    logger_csi_.Write(ss.str());
  }
}

void PhyStats::RecordEvm(size_t frame_id, size_t num_rec_sc,
                         const arma::uvec& ue_map) {
  if (kEnableCsvLog) {
    std::stringstream ss_evm;
    std::stringstream ss_evm_sc;
    ss_evm << frame_id;
    ss_evm_sc << frame_id;
    const size_t num_frame_data = config_->OfdmDataNum() * num_rxdata_symbols_;
    for (size_t ue_id = 0; ue_id < config_->UeAntNum(); ue_id++) {
      float evm_pcnt =
          ((evm_buffer_[frame_id % kFrameWnd][ue_id] / num_frame_data) *
           100.0f);
      ss_evm << ","
             << ((ue_map.at(ue_id) != 0) ? std::to_string(evm_pcnt) : "ns");
    }
    const size_t sc_step = config_->OfdmDataNum() / num_rec_sc;
    const size_t sc_offset = sc_step / 2;
    for (size_t ue_id = 0; ue_id < config_->UeAntNum(); ue_id++) {
      for (size_t sc_rec = 0; sc_rec < num_rec_sc; sc_rec++) {
        const size_t sc_id = sc_rec * sc_step + sc_offset;
        const size_t ue_offset = ue_id * config_->OfdmDataNum();
        ss_evm_sc << ","
                  << (evm_sc_buffer_[frame_id % kFrameWnd][ue_offset + sc_id] *
                      100.0f);
      }
    }
    logger_evm_.Write(ss_evm.str());
    logger_evm_sc_.Write(ss_evm_sc.str());
  }
}

void PhyStats::RecordEvmSnr(size_t frame_id, const arma::uvec& ue_map) {
  if (kEnableCsvLog) {
    std::stringstream ss;
    ss << frame_id;
    const size_t num_frame_data = config_->OfdmDataNum() * num_rxdata_symbols_;
    for (size_t i = 0; i < config_->UeAntNum(); i++) {
      float evm_snr_db =
          (-10.0f *
           std::log10(evm_buffer_[frame_id % kFrameWnd][i] / num_frame_data));
      ss << "," << ((ue_map.at(i) != 0) ? std::to_string(evm_snr_db) : "ns");
    }
    logger_evm_snr_.Write(ss.str());
  }
}

void PhyStats::RecordDlPilotSnr(size_t frame_id, const arma::uvec& ue_map) {
  if (kEnableCsvLog) {
    const size_t dl_pilots_num = config_->Frame().ClientDlPilotSymbols();
    if (dl_pilots_num > 0) {
      std::stringstream ss_snr;
      std::stringstream ss_rssi;
      std::stringstream ss_noise;
      ss_snr << frame_id;
      ss_rssi << frame_id;
      ss_noise << frame_id;
      const size_t frame_slot = frame_id % kFrameWnd;
      for (size_t i = 0; i < config_->UeAntNum(); i++) {
        for (size_t j = 0; j < dl_pilots_num; j++) {
          const size_t idx_offset = i * dl_pilots_num + j;
          ss_snr << ","
                 << (ue_map.at(i) != 0
                         ? std::to_string(dl_pilot_snr_[frame_slot][idx_offset])
                         : "ns");
          ss_rssi << ","
                  << (ue_map.at(i) != 0
                          ? std::to_string(
                                dl_pilot_rssi_[frame_slot][idx_offset])
                          : "ns");
          ss_noise << ","
                   << (ue_map.at(i) != 0
                           ? std::to_string(
                                 dl_pilot_noise_[frame_slot][idx_offset])
                           : "ns");
        }
      }
      logger_bf_snr_.Write(ss_snr.str());
      logger_bf_rssi_.Write(ss_rssi.str());
      logger_bf_noise_.Write(ss_noise.str());
    }
  }
}

void PhyStats::RecordDlCsi(size_t frame_id, size_t num_rec_sc,
                           const Table<complex_float>& csi_buffer,
                           const arma::uvec& ue_list) {
  if (kEnableCsvLog) {
    const size_t csi_offset_base = (frame_id % kFrameWnd) * config_->UeAntNum();
    std::stringstream ss;
    ss << frame_id;
    for (long long unsigned ue_id : ue_list) {
      const auto* csi_buffer_ptr = reinterpret_cast<const arma::cx_float*>(
          csi_buffer.At(csi_offset_base + ue_id));
      const size_t sc_step = config_->OfdmDataNum() / num_rec_sc;
      const size_t sc_offset = sc_step / 2;
      for (size_t sc_rec = 0; sc_rec < num_rec_sc; sc_rec++) {
        const size_t sc_id = sc_rec * sc_step + sc_offset;
        ss << "," << std::abs(csi_buffer_ptr[sc_id]);
      }
    }
    logger_csi_.Write(ss.str());
  }
}

void PhyStats::RecordBer(size_t frame_id, const arma::uvec& ue_map) {
  if (kEnableCsvLog) {
    std::stringstream ss;
    ss << frame_id;
    const size_t frame_slot = frame_id % kFrameWnd;
    for (size_t i = 0; i < config_->UeAntNum(); i++) {
      size_t& error_bits = frame_bit_errors_[i][frame_slot];
      size_t& total_bits = frame_decoded_bits_[i][frame_slot];
      if (ue_map.at(i) != 0) {
        ss << ","
           << (static_cast<float>(error_bits) / static_cast<float>(total_bits));
      } else {
        ss << ", ns";
      }
      error_bits = 0;
      total_bits = 0;
    }
    logger_ber_.Write(ss.str());
  }
}

void PhyStats::RecordSer(size_t frame_id, const arma::uvec& ue_map) {
  if (kEnableCsvLog) {
    std::stringstream ss;
    ss << frame_id;
    const size_t frame_slot = frame_id % kFrameWnd;
    for (size_t i = 0; i < config_->UeAntNum(); i++) {
      size_t& error_symbols = frame_symbol_errors_[i][frame_slot];
      size_t& total_symbols = frame_decoded_symbols_[i][frame_slot];
      if (ue_map.at(i) != 0) {
        ss << ","
           << (static_cast<float>(error_symbols) /
               static_cast<float>(total_symbols));
      } else {
        ss << ", ns";
      }
      error_symbols = 0;
      total_symbols = 0;
    }
    logger_ser_.Write(ss.str());
  }
}

void PhyStats::UpdateCalibPilotSnr(size_t frame_id, size_t calib_sym_id,
                                   size_t ant_id, complex_float* fft_data) {
  const arma::cx_fmat fft_mat(reinterpret_cast<arma::cx_float*>(fft_data),
                              config_->OfdmCaNum(), 1, false);
  arma::fmat fft_abs_mat = arma::abs(fft_mat);
  arma::fmat fft_abs_mag = fft_abs_mat % fft_abs_mat;
  const float rssi = arma::as_scalar(arma::sum(fft_abs_mag));
  const float noise_per_sc1 = arma::as_scalar(
      arma::mean(fft_abs_mag.rows(0, config_->OfdmDataStart() - 1)));
  const float noise_per_sc2 = arma::as_scalar(arma::mean(
      fft_abs_mag.rows(config_->OfdmDataStop(), config_->OfdmCaNum() - 1)));
  const float noise =
      config_->OfdmCaNum() * (noise_per_sc1 + noise_per_sc2) / 2;
  const float snr = (rssi - noise) / noise;
  calib_pilot_snr_[frame_id % kFrameWnd][calib_sym_id * config_->BsAntNum() +
                                         ant_id] = (10.0f * std::log10(snr));
}

void PhyStats::UpdatePilotSnr(size_t frame_id, size_t ue_id, size_t ant_id,
                              complex_float* fft_data) {
  const arma::cx_fvec fft_vec(reinterpret_cast<arma::cx_float*>(fft_data),
                              config_->OfdmCaNum(), false);
  arma::fvec fft_abs_vec = arma::abs(fft_vec);
  arma::fvec fft_abs_mag = fft_abs_vec % fft_abs_vec;
  const float rssi_per_sc = arma::mean(fft_abs_mag(config_->PilotUeSc(ue_id)));
  fft_abs_mag.shed_rows(config_->OfdmDataStart(), config_->OfdmDataStop() - 1);
  const float noise_per_sc = arma::mean(fft_abs_mag);
  const float snr = (rssi_per_sc - noise_per_sc) / noise_per_sc;
  const size_t frame_slot = frame_id % kFrameWnd;
  const size_t idx_offset = ue_id * config_->BsAntNum() + ant_id;
  pilot_snr_[frame_slot][idx_offset] = 10.0f * std::log10(snr);
  pilot_rssi_[frame_slot][idx_offset] = rssi_per_sc;
  pilot_noise_[frame_slot][idx_offset] = noise_per_sc;
}

void PhyStats::UpdateDlPilotSnr(size_t frame_id, size_t symbol_id,
                                size_t ant_id, complex_float* fft_data) {
  const arma::cx_fvec fft_vec(reinterpret_cast<arma::cx_float*>(fft_data),
                              config_->OfdmCaNum(), false);
  arma::fvec fft_abs_vec = arma::abs(fft_vec);
  arma::fvec fft_abs_mag = fft_abs_vec % fft_abs_vec;
  const float rssi_per_sc = arma::mean(fft_abs_mag);
  fft_abs_mag.shed_rows(config_->OfdmDataStart(), config_->OfdmDataStop() - 1);
  const float noise_per_sc = arma::mean(fft_abs_mag);
  const float snr = (rssi_per_sc - noise_per_sc) / noise_per_sc;
  const size_t frame_slot = frame_id % kFrameWnd;
  const size_t idx_offset =
      ant_id * config_->Frame().ClientDlPilotSymbols() + symbol_id;
  dl_pilot_snr_[frame_slot][idx_offset] = 10.0f * std::log10(snr);
  dl_pilot_rssi_[frame_slot][idx_offset] = rssi_per_sc;
  dl_pilot_noise_[frame_slot][idx_offset] = noise_per_sc;
}

void PhyStats::PrintBeamStats(size_t frame_id) {
  const size_t frame_slot = frame_id % kFrameWnd;
  [[maybe_unused]] std::stringstream ss;
  ss << "Frame " << frame_id
     << " Beamweight matrix inverse condition number range: " << std::fixed
     << std::setw(5) << std::setprecision(2);
  arma::fvec cond_vec(csi_cond_[frame_slot], config_->OfdmDataNum(), false);
  float max_cond = 0;
  float min_cond = 1;
  for (size_t j = 0; j < config_->OfdmDataNum(); j++) {
    if (cond_vec.at(j) < min_cond) {
      min_cond = cond_vec.at(j);
    }
    if (cond_vec.at(j) > max_cond) {
      max_cond = cond_vec.at(j);
    }
  }
  ss << "[" << min_cond << "," << max_cond
     << "], Mean: " << arma::mean(cond_vec);
  ss << std::endl;
  AGORA_LOG_INFO("%s", ss.str().c_str());
}

void PhyStats::UpdateCsiCond(size_t frame_id, size_t sc_id, float cond) {
  csi_cond_[frame_id % kFrameWnd][sc_id] = cond;
}

void PhyStats::UpdateEvm(size_t frame_id, size_t data_symbol_id, size_t sc_id,
                         const arma::cx_fvec& eq_vec,
                         const arma::uvec& ue_list) {
  arma::cx_fvec tx_data = gt_cube_.slice(data_symbol_id).col(sc_id);
  arma::fvec evm_vec = arma::square(arma::abs(eq_vec - tx_data(ue_list)));
  evm_sc_buffer_[frame_id % kFrameWnd][sc_id] = arma::mean(evm_vec);
  arma::fvec evm_buf(evm_buffer_[frame_id % kFrameWnd], config_->UeAntNum(),
                     false);
  evm_buf(ue_list) += evm_vec;
}

void PhyStats::UpdateEvm(size_t frame_id, size_t data_symbol_id, size_t sc_id,
                         size_t tx_ue_id, size_t rx_ue_id, arma::cx_float eq) {
  const float evm =
      std::norm(eq - gt_cube_.slice(data_symbol_id)(tx_ue_id, sc_id));
  evm_buffer_[frame_id % kFrameWnd][rx_ue_id] += evm;
  evm_sc_buffer_[frame_id % kFrameWnd]
                [rx_ue_id * config_->OfdmDataNum() + sc_id] = evm;
}

void PhyStats::UpdateBitErrors(size_t ue_id, size_t offset, size_t frame_slot,
                               uint8_t tx_byte, uint8_t rx_byte) {
  static constexpr size_t kBitsInByte = 8;
  AGORA_LOG_TRACE("Updating bit errors: User %zu Offset  %zu Tx %d Rx %d\n",
                  ue_id, offset, tx_byte, rx_byte);
  uint8_t xor_byte(tx_byte ^ rx_byte);
  size_t bit_errors = 0;
  for (size_t j = 0; j < kBitsInByte; j++) {
    bit_errors += (xor_byte & 1);
    xor_byte >>= 1;
  }
  bit_error_count_[ue_id][offset] += bit_errors;
  frame_bit_errors_[ue_id][frame_slot] += bit_errors;
}

void PhyStats::UpdateDecodedBits(size_t ue_id, size_t offset, size_t frame_slot,
                                 size_t new_bits_num) {
  decoded_bits_count_[ue_id][offset] += new_bits_num;
  frame_decoded_bits_[ue_id][frame_slot] += new_bits_num;
}

void PhyStats::UpdateBlockErrors(size_t ue_id, size_t offset, size_t frame_slot,
                                 size_t block_error_count) {
  block_error_count_[ue_id][offset] +=
      static_cast<unsigned long>(block_error_count > 0);
  frame_symbol_errors_[ue_id][frame_slot] += block_error_count;
}

void PhyStats::IncrementDecodedBlocks(size_t ue_id, size_t offset,
                                      size_t frame_slot) {
  decoded_blocks_count_[ue_id][offset]++;
  frame_decoded_symbols_[ue_id][frame_slot] += config_->GetOFDMDataNum();
}

void PhyStats::UpdateUncodedBitErrors(size_t ue_id, size_t offset,
                                      size_t mod_bit_size, uint8_t tx_byte,
                                      uint8_t rx_byte) {
  uint8_t xor_byte(tx_byte ^ rx_byte);
  size_t bit_errors = 0;
  for (size_t j = 0; j < mod_bit_size; j++) {
    bit_errors += xor_byte & 1;
    xor_byte >>= 1;
  }
  uncoded_bit_error_count_[ue_id][offset] += bit_errors;
}

void PhyStats::UpdateUncodedBits(size_t ue_id, size_t offset,
                                 size_t new_bits_num) {
  uncoded_bits_count_[ue_id][offset] += new_bits_num;
}

void PhyStats::UpdateUlCsi(size_t frame_id, size_t sc_id,
                           const arma::cx_fmat& mat_in) {
  logger_ul_csi_.UpdateMatBuf(frame_id, sc_id, mat_in);
}

void PhyStats::UpdateDlCsi(size_t frame_id, size_t sc_id,
                           const arma::cx_fmat& mat_in) {
  logger_dl_csi_.UpdateMatBuf(frame_id, sc_id, mat_in);
}

void PhyStats::UpdateUlBeam(size_t frame_id, size_t sc_id,
                            const arma::cx_fmat& mat_in) {
  logger_ul_beam_.UpdateMatBuf(frame_id, sc_id, mat_in);
}

void PhyStats::UpdateDlBeam(size_t frame_id, size_t sc_id,
                            const arma::cx_fmat& mat_in) {
  logger_dl_beam_.UpdateMatBuf(frame_id, sc_id, mat_in);
}

void PhyStats::UpdateCalibMat(size_t frame_id, size_t sc_id,
                              const arma::cx_fvec& vec_in) {
  logger_calib_.UpdateMatBuf(frame_id, sc_id, vec_in);
}

float PhyStats::GetNoise(size_t frame_id, const arma::uvec& ue_list) {
  arma::fvec noise_vec(pilot_noise_[frame_id % kFrameWnd],
                       config_->BsAntNum() * config_->UeAntNum(), false);

  return arma::as_scalar(arma::mean(noise_vec(ue_list)));
}
