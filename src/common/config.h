// Copyright (c) 2018-2020, Rice University
// RENEW OPEN SOURCE LICENSE: http://renew-wireless.org/license

/**
 * @file config.h
 * @brief Declaration file for the configuration class which importants
 * json configuration values into class variables
 */

#ifndef CONFIG_H_
#define CONFIG_H_

#include <atomic>
#include <cstddef>
#include <string>
#include <vector>

#include "armadillo"
#include "common_typedef_sdk.h"
#include "framestats.h"
#include "ldpc_config.h"
#include "memory_manage.h"
#include "nlohmann/json.hpp"
#include "symbols.h"
#include "utils.h"

class Config {
 public:
  static constexpr bool kDebugRecipCal = false;
  // Constructor
  explicit Config(std::string jsonfilename);
  ~Config();

  inline void Running(bool value) { this->running_.store(value); }
  inline bool Running() const { return this->running_.load(); }
  inline size_t BsAntNum() const { return this->bs_ant_num_; }
  inline void BsAntNum(size_t n_bs_ant) { this->bs_ant_num_ = n_bs_ant; }

  // Inline accessors (basic types)
  inline size_t BfAntNum() const { return this->bf_ant_num_; }
  inline size_t UeNum() const { return this->ue_num_; }
  inline size_t UeAntNum() const { return this->ue_ant_num_; }
  inline size_t UeAntOffset() const { return this->ue_ant_offset_; }
  inline size_t UeAntTotal() const { return this->ue_ant_total_; }

  inline size_t OfdmCaNum() const { return this->ofdm_ca_num_; }
  inline size_t CpLen() const { return this->cp_len_; }
  inline size_t OfdmDataNum() const { return this->ofdm_data_num_; }
  inline size_t OfdmDataStart() const { return this->ofdm_data_start_; }

  inline size_t OfdmDataStop() const { return this->ofdm_data_stop_; }
  inline size_t OfdmPilotSpacing() const { return this->ofdm_pilot_spacing_; }

  inline bool HwFramer() const { return this->hw_framer_; }
  inline bool UeHwFramer() const { return this->ue_hw_framer_; }
  inline size_t UeResyncPeriod() const { return this->ue_resync_period_; }
  inline double FreqGhz() const { return this->freq_ghz_; };
  inline double Freq() const { return this->freq_; }
  inline double Rate() const { return this->rate_; }
  inline double Nco() const { return this->nco_; }

  inline double RadioRfFreq() const { return this->radio_rf_freq_; }
  inline double BwFilter() const { return this->bw_filter_; }
  inline bool SingleGain() const { return this->single_gain_; }
  inline double TxGainA() const { return this->tx_gain_a_; }
  inline double RxGainA() const { return this->rx_gain_a_; }
  inline double TxGainB() const { return this->tx_gain_b_; }
  inline double RxGainB() const { return this->rx_gain_b_; }
  inline double CalibTxGainA() const { return this->calib_tx_gain_a_; }
  inline double CalibTxGainB() const { return this->calib_tx_gain_b_; }
  inline double ClientTxGainA(size_t id) const {
    return this->client_tx_gain_a_.at(id);
  }
  inline double ClientRxGainA(size_t id) const {
    return this->client_rx_gain_a_.at(id);
  }
  inline double ClientTxGainB(size_t id) const {
    return this->client_tx_gain_b_.at(id);
  }
  inline double ClientRxGainB(size_t id) const {
    return this->client_rx_gain_b_.at(id);
  }
  inline const std::vector<double>& ClientTxGainA() const {
    return this->client_tx_gain_a_;
  }
  inline const std::vector<double>& ClientRxGainA() const {
    return this->client_rx_gain_a_;
  }
  inline const std::vector<double>& ClientTxGainB() const {
    return this->client_tx_gain_b_;
  }
  inline const std::vector<double>& ClientRxGainB() const {
    return this->client_rx_gain_b_;
  }
  inline size_t NumCells() const { return this->num_cells_; }
  inline size_t NumRadios() const { return this->num_radios_; }
  inline size_t InitCalibRepeat() const { return this->init_calib_repeat_; }

  inline size_t NumChannels() const { return this->num_channels_; }
  inline size_t NumUeChannels() const { return this->num_ue_channels_; }
  inline size_t RefAnt(size_t id) const { return this->ref_ant_.at(id); }
  inline size_t RefRadio(size_t id) const { return this->ref_radio_.at(id); }
  inline size_t BeaconAnt() const { return this->beacon_ant_; }
  inline size_t BeaconLen() const { return this->beacon_len_; }

  inline bool SmoothCalib() const { return this->smooth_calib_; }
  inline bool Beamsweep() const { return this->beamsweep_; }
  inline bool SampleCalEn() const { return this->sample_cal_en_; }
  inline bool ImbalanceCalEn() const { return this->imbalance_cal_en_; }
  inline size_t BeamformingAlgo() const { return this->beamforming_algo_; }
  inline std::string Beamforming() const { return this->beamforming_str_; }
  inline size_t SpatialStreamsNum() const { return this->num_spatial_streams_; }
  inline bool ExternalRefNode(size_t id) const {
    return this->external_ref_node_.at(id);
  }
  inline std::string Channel() const { return this->channel_; }
  inline std::string UeChannel() const { return this->ue_channel_; }

  // Groups for Downlink Recip Cal
  // Returns antenna number for rec cal dl symbol
  // Assumes that there are the same number of dl cal symbols in each frame
  inline size_t RecipCalDlAnt(size_t frame_id, size_t dl_cal_symbol) const {
    assert(GetSymbolType(dl_cal_symbol) == SymbolType::kCalDL);
    const size_t dl_cal_offset = (frame_id * frame_.NumDLCalSyms()) +
                                 frame_.GetDLCalSymbolIdx(dl_cal_symbol);

    const size_t tx_ant = dl_cal_offset % bf_ant_num_;

    if (kDebugRecipCal) {
      std::printf("RecipCalDlAnt (Frame %zu, Symbol %zu) tx antenna %zu\n",
                  frame_id, dl_cal_symbol, tx_ant);
    }
    return (tx_ant);
  };

  inline size_t ModifyRecCalIndex(size_t previous_index,
                                  int mod_value = 0) const {
    return (previous_index + mod_value) % kFrameWnd;
  }

  inline size_t RecipCalIndex(size_t frame_id) const {
    const size_t frame_cal_idx = frame_id / RecipCalFrameCnt();
    return ModifyRecCalIndex(frame_cal_idx);
  }

  // Returns the cal index if ant tx dl cal pilots this frame
  // SIZE_MAX otherwise
  inline size_t RecipCalUlRxIndex(size_t frame_id, size_t ant) const {
    const size_t num_frames_for_full_cal = RecipCalFrameCnt();
    const size_t num_cal_per_idx = frame_.NumDLCalSyms();
    const size_t cal_offset = (frame_id % num_frames_for_full_cal);
    const size_t tx_ant_start = cal_offset * num_cal_per_idx;
    const size_t tx_ant_end = tx_ant_start + (num_cal_per_idx - 1);

    size_t cal_ind;
    if ((ant >= tx_ant_start) && (ant <= tx_ant_end)) {
      cal_ind = RecipCalIndex(frame_id);
    } else {
      cal_ind = SIZE_MAX;
    }

    if (kDebugRecipCal) {
      std::printf(
          "RecipCalUlRxIndex (Frame %zu, Antenna %zu) index %zu - Start %zu, "
          "End %zu, full %zu\n",
          frame_id, ant, cal_ind, tx_ant_start, tx_ant_end,
          num_frames_for_full_cal);
    }
    return (cal_ind);
  };

  // Returns the number of frames to obtain a full set of RecipCal data
  // assumes that bf_ant_num_ % frame_.NumDLCalSyms() == 0
  inline size_t RecipCalFrameCnt() const {
    if ((frame_.IsRecCalEnabled() == false) || (frame_.NumDLCalSyms() == 0)) {
      return 0;
    } else {
      assert((bf_ant_num_ % frame_.NumDLCalSyms()) == 0);
      return bf_ant_num_ / frame_.NumDLCalSyms();
    }
  }

  inline size_t CoreOffset() const { return this->core_offset_; }
  inline size_t WorkerThreadNum() const { return this->worker_thread_num_; }
  inline size_t SocketThreadNum() const { return this->socket_thread_num_; }
  inline size_t UeCoreOffset() const { return this->ue_core_offset_; }
  inline size_t UeWorkerThreadNum() const {
    return this->ue_worker_thread_num_;
  }
  inline size_t UeSocketThreadNum() const {
    return this->ue_socket_thread_num_;
  }

  inline size_t FftThreadNum() const { return this->fft_thread_num_; }
  inline size_t DemulThreadNum() const { return this->demul_thread_num_; }
  inline size_t DecodeThreadNum() const { return this->decode_thread_num_; }
  inline size_t BeamThreadNum() const { return this->beam_thread_num_; }

  inline bool SmallMimoAcc() const { return this->small_mimo_acc_; }
  inline size_t DemulBlockSize() const { return this->demul_block_size_; }
  inline size_t DemulEventsPerSymbol() const {
    return this->demul_events_per_symbol_;
  }
  inline size_t BeamBlockSize() const { return this->beam_block_size_; }
  inline size_t BeamEventsPerSymbol() const {
    return this->beam_events_per_symbol_;
  }
  inline size_t FftBlockSize() const { return this->fft_block_size_; }

  inline size_t EncodeBlockSize() const { return this->encode_block_size_; }
  inline bool FreqOrthogonalPilot() const {
    return this->freq_orthogonal_pilot_;
  }
  inline bool GroupPilotSc() const { return this->group_pilot_sc_; }
  inline size_t PilotScGroupSize() const { return this->pilot_sc_group_size_; }
  inline size_t OfdmTxZeroPrefix() const { return this->ofdm_tx_zero_prefix_; }
  inline size_t OfdmTxZeroPostfix() const {
    return this->ofdm_tx_zero_postfix_;
  }
  inline size_t OfdmRxZeroPrefixBs() const {
    return this->ofdm_rx_zero_prefix_bs_;
  }

  inline size_t OfdmRxZeroPrefixCalUl() const {
    return this->ofdm_rx_zero_prefix_cal_ul_;
  }
  void OfdmRxZeroPrefixCalUl(size_t prefix) {
    this->ofdm_rx_zero_prefix_cal_ul_ = prefix;
  }
  inline size_t OfdmRxZeroPrefixCalDl() const {
    return this->ofdm_rx_zero_prefix_cal_dl_;
  }
  void OfdmRxZeroPrefixCalDl(const size_t prefix) {
    this->ofdm_rx_zero_prefix_cal_dl_ = prefix;
  }
  inline size_t OfdmRxZeroPrefixClient() const {
    return this->ofdm_rx_zero_prefix_client_;
  }
  inline size_t SampsPerSymbol() const { return this->samps_per_symbol_; }
  inline size_t SampsPerFrame() const {
    return this->frame_.NumTotalSyms() * this->samps_per_symbol_;
  }
  inline size_t PacketLength() const { return this->packet_length_; }

  inline float Scale() const { return this->scale_; }
  inline bool BigstationMode() const { return this->bigstation_mode_; }
  inline size_t DlPacketLength() const { return this->dl_packet_length_; }
  inline std::string Modulation(Direction dir) const {
    return dir == Direction::kUplink ? this->ul_modulation_
                                     : this->dl_modulation_;
  }

  inline size_t ModOrderBits(Direction dir) const {
    return dir == Direction::kUplink ? this->ul_mod_order_bits_
                                     : this->dl_mod_order_bits_;
  }
  inline size_t NumBytesPerCb(Direction dir) const {
    return dir == Direction::kUplink ? this->ul_num_bytes_per_cb_
                                     : this->dl_num_bytes_per_cb_;
  }
  inline size_t NumPaddingBytesPerCb(Direction dir) const {
    return dir == Direction::kUplink ? this->ul_num_padding_bytes_per_cb_
                                     : this->dl_num_padding_bytes_per_cb_;
  }
  inline size_t MacDataBytesNumPerframe(Direction dir) const {
    return dir == Direction::kUplink ? this->ul_mac_data_bytes_num_perframe_
                                     : this->dl_mac_data_bytes_num_perframe_;
  }
  inline size_t MacBytesNumPerframe(Direction dir) const {
    return dir == Direction::kUplink ? this->ul_mac_bytes_num_perframe_
                                     : this->dl_mac_bytes_num_perframe_;
  }

  inline size_t MacPacketLength(Direction dir) const {
    return dir == Direction::kUplink ? this->ul_mac_packet_length_
                                     : this->dl_mac_packet_length_;
  }
  inline size_t MacPayloadMaxLength(Direction dir) const {
    return dir == Direction::kUplink ? this->ul_mac_data_length_max_
                                     : this->dl_mac_data_length_max_;
  }
  inline size_t MacPacketsPerframe(Direction dir) const {
    return dir == Direction::kUplink ? this->ul_mac_packets_perframe_
                                     : this->dl_mac_packets_perframe_;
  }
  inline const LDPCconfig& LdpcConfig(Direction dir) const {
    return dir == Direction::kUplink ? this->ul_ldpc_config_
                                     : this->dl_ldpc_config_;
  }
  inline const LDPCconfig& BcLdpcConfig() const {
    return dl_bcast_ldpc_config_;
  }
  inline Table<complex_float>& ModTable(Direction dir) {
    return dir == Direction::kUplink ? this->ul_mod_table_
                                     : this->dl_mod_table_;
  }
  inline const nlohmann::json& MCSParams(Direction dir) const {
    return dir == Direction::kUplink ? this->ul_mcs_params_
                                     : this->dl_mcs_params_;
  }
  inline size_t SubcarrierPerCodeBlock(Direction dir) const {
    return this->LdpcConfig(dir).NumCbCodewLen() / this->ModOrderBits(dir);
  }
  inline size_t McsIndex(Direction dir) const {
    return dir == Direction::kUplink ? this->ul_mcs_index_
                                     : this->dl_mcs_index_;
  }

  inline bool ScrambleEnabled() const { return this->scramble_enabled_; }

  inline std::string UeServerAddr() const { return this->ue_server_addr_; }
  inline std::string BsServerAddr() const { return this->bs_server_addr_; }

  inline std::string UeRruAddr() const { return this->ue_rru_addr_; }
  inline std::string BsRruAddr() const { return this->bs_rru_addr_; }

  inline int BsServerPort() const { return this->bs_server_port_; }
  inline int BsRruPort() const { return this->bs_rru_port_; }
  inline int UeServerPort() const { return this->ue_server_port_; }
  inline int UeRruPort() const { return this->ue_rru_port_; }

  inline size_t FramesToTest() const { return this->frames_to_test_; }
  inline float NoiseLevel() const { return this->noise_level_; }
  inline bool FftInRru() const { return this->fft_in_rru_; }

  inline uint16_t DpdkNumPorts() const { return this->dpdk_num_ports_; }
  inline uint16_t DpdkPortOffset() const { return this->dpdk_port_offset_; }

  inline const std::string& DpdkMacAddrs() const {
    return this->dpdk_mac_addrs_;
  }

  inline size_t BsMacRxPort() const { return this->bs_mac_rx_port_; }
  inline size_t BsMacTxPort() const { return this->bs_mac_tx_port_; }

  inline size_t UeMacRxPort() const { return this->ue_mac_rx_port_; }
  inline size_t UeMacTxPort() const { return this->ue_mac_tx_port_; }

  inline const std::string& LogListenerAddr() const {
    return this->log_listener_addr_;
  }

  inline size_t LogListenerPort() const { return this->log_listener_port_; }

  inline size_t LogScNum() const { return this->log_sc_num_; }
  inline bool LogTimestamp() const { return this->log_timestamp_; }

  /* Inline accessors (complex types) */
  inline const std::vector<int>& ClTxAdvance() const {
    return this->cl_tx_advance_;
  }
  inline const std::vector<float>& ClCorrScale() const {
    return this->cl_corr_scale_;
  }

  inline const FrameStats& Frame() const { return this->frame_; }
  inline const std::vector<std::complex<float>>& PilotCf32() const {
    return this->pilot_cf32_;
  };
  inline const std::vector<std::complex<float>>& GoldCf32() const {
    return this->gold_cf32_;
  };
  inline const std::vector<uint32_t>& Coeffs() const { return this->coeffs_; };

  inline const std::vector<uint32_t>& Pilot() const { return this->pilot_; };
  inline const std::vector<uint32_t>& Beacon() const { return this->beacon_; };
  // inline const complex_float *pilots (void ) const { return this->pilots_; };
  inline const complex_float* PilotsSgn() const { return this->pilots_sgn_; };
  inline const std::vector<std::complex<float>>& CommonPilot() const {
    return this->common_pilot_;
  };
  inline const std::vector<std::string>& RadioId() const {
    return this->radio_id_;
  };
  inline const std::vector<std::string>& HubId() const {
    return this->hub_id_;
  };
  inline const std::vector<std::string>& UeRadioId() const {
    return this->ue_radio_id_;
  };
  inline const std::vector<std::string>& UeRadioName() const {
    return this->ue_radio_name_;
  };
  inline const std::vector<size_t>& CellId() const { return this->cell_id_; }

  // non-const (can modify)
  inline Table<complex_float>& UeSpecificPilot() {
    return this->ue_specific_pilot_;
  };
  inline Table<std::complex<int16_t>>& UeSpecificPilotT() {
    return this->ue_specific_pilot_t_;
  };
  inline std::vector<std::complex<int16_t>>& PilotCi16() {
    return this->pilot_ci16_;
  };
  inline std::vector<std::complex<int16_t>>& PilotUeCi16(size_t ue_id,
                                                         size_t pilot_idx) {
    return this->pilot_ue_ci16_.at(ue_id).at(pilot_idx);
  };
  inline const arma::uvec& PilotUeSc(size_t ue_id) const {
    return this->pilot_ue_sc_.at(ue_id);
  };
  inline std::vector<std::complex<int16_t>>& BeaconCi16() {
    return this->beacon_ci16_;
  };

  inline Table<int8_t>& DlBits() { return this->dl_bits_; }
  inline Table<int8_t>& UlBits() { return this->ul_bits_; }
  inline Table<int8_t>& DlModBits() { return this->dl_mod_bits_; }
  inline Table<int8_t>& UlModBits() { return this->ul_mod_bits_; }
  inline Table<complex_float>& UlIqF() { return this->ul_iq_f_; }
  inline Table<complex_float>& DlIqF() { return this->dl_iq_f_; }
  inline Table<std::complex<int16_t>>& UlIqT() { return this->ul_iq_t_; }
  inline Table<std::complex<int16_t>>& DlIqT() { return this->dl_iq_t_; }

  // Public functions
  void GenPilots();
  void GenData();
  void GenBroadcastSlots(std::vector<std::complex<int16_t>*>& bcast_iq_samps,
                         std::vector<size_t> ctrl_msg);
  size_t DecodeBroadcastSlots(const int16_t* const bcast_iq_samps);
  void UpdateUlMCS(const nlohmann::json& ul_mcs_params);
  void UpdateDlMCS(const nlohmann::json& dl_mcs_params);
  void UpdateCtrlMCS();

  /// TODO document and review
  size_t GetSymbolId(size_t input_id) const;

  bool IsBeacon(size_t /*frame_id*/, size_t /*symbol_id*/) const;
  bool IsPilot(size_t /*unused*/, size_t /*symbol_id*/) const;
  bool IsDlPilot(size_t /*unused*/, size_t /*symbol_id*/) const;
  bool IsCalDlPilot(size_t /*unused*/, size_t /*symbol_id*/) const;
  bool IsCalUlPilot(size_t /*unused*/, size_t /*symbol_id*/) const;
  bool IsDownlink(size_t /*frame_id*/, size_t /*symbol_id*/) const;
  bool IsDownlinkBroadcast(size_t /*frame_id*/, size_t /*symbol_id*/) const;
  bool IsUplink(size_t /*unused*/, size_t /*symbol_id*/) const;

  /* Public functions that do not meet coding standard format */
  /// Return the symbol type of this symbol in this frame
  SymbolType GetSymbolType(size_t symbol_id) const;

  /// Return total number of data symbols of all frames in a buffer
  /// that holds data of kFrameWnd frames
  inline size_t GetTotalDataSymbolIdx(size_t frame_id, size_t symbol_id) const {
    return ((frame_id % kFrameWnd) * this->frame_.NumDataSyms() + symbol_id);
  }

  /// Return total number of uplink data symbols of all frames in a buffer
  /// that holds data of kFrameWnd frames
  inline size_t GetTotalDataSymbolIdxUl(size_t frame_id,
                                        size_t symbol_idx_ul) const {
    return ((frame_id % kFrameWnd) * this->frame_.NumULSyms() + symbol_idx_ul);
  }

  /// Return total number of downlink data symbols of all frames in a buffer
  /// that holds data of kFrameWnd frames
  inline size_t GetTotalDataSymbolIdxDl(size_t frame_id,
                                        size_t symbol_idx_dl) const {
    return ((frame_id % kFrameWnd) * this->frame_.NumDLSyms() + symbol_idx_dl);
  }

  inline size_t GetTotalSymbolIdxDl(size_t frame_id, size_t symbol_id) {
    const size_t symbol_idx_dl =
        symbol_id < this->frame_.GetDLSymbol(0)
            ? this->frame_.GetDLControlSymbolIdx(symbol_id)
            : this->frame_.GetDLSymbolIdx(symbol_id) +
                  this->frame_.NumDlControlSyms();
    return (frame_id % kFrameWnd) *
               (this->frame_.NumDlControlSyms() + this->frame_.NumDLSyms()) +
           symbol_idx_dl;
  }

  //Returns Beacon+Dl symbol index
  inline size_t GetBeaconDlIdx(size_t symbol_id) const {
    size_t symbol_idx = SIZE_MAX;
    const auto type = GetSymbolType(symbol_id);
    if (type == SymbolType::kBeacon) {
      symbol_idx = Frame().GetBeaconSymbolIdx(symbol_id);
    } else if (type == SymbolType::kControl) {
      symbol_idx =
          Frame().GetDLControlSymbolIdx(symbol_id) + Frame().NumBeaconSyms();
    } else if (type == SymbolType::kDL) {
      symbol_idx = Frame().GetDLSymbolIdx(symbol_id) + Frame().NumDlBcastSyms();
    } else {
      throw std::runtime_error("Invalid BS Beacon or DL symbol id " +
                               std::to_string(symbol_id));
    }
    return symbol_idx;
  }

  //Returns Pilot+Ul symbol index
  inline size_t GetPilotUlIdx(size_t symbol_id) const {
    size_t symbol_idx = SIZE_MAX;
    const auto type = GetSymbolType(symbol_id);
    if (type == SymbolType::kPilot) {
      symbol_idx = Frame().GetPilotSymbolIdx(symbol_id);
    } else if (type == SymbolType::kUL) {
      symbol_idx = Frame().GetULSymbolIdx(symbol_id) + Frame().NumPilotSyms();
    } else {
      throw std::runtime_error("Invalid Ue Pilot or UL symbol id " +
                               std::to_string(symbol_id));
    }
    return symbol_idx;
  }

  /// Return the frame duration in seconds
  inline double GetFrameDurationSec() const {
    return ((this->frame_.NumTotalSyms() * this->samps_per_symbol_) /
            this->rate_);
  }

  /// Fetch the data buffer for this frame and symbol ID. The symbol must
  /// be an uplink symbol.
  inline complex_float* GetDataBuf(Table<complex_float>& data_buffers,
                                   size_t frame_id, size_t symbol_id) const {
    size_t frame_slot = frame_id % kFrameWnd;
    size_t symbol_offset = (frame_slot * this->frame_.NumULSyms()) +
                           this->frame_.GetULSymbolIdx(symbol_id);
    return data_buffers[symbol_offset];
  }

  /// Return the subcarrier ID which we should refer to for the beamweight
  /// matrices of subcarrier [sc_id].
  inline size_t GetBeamScId(size_t sc_id) const {
    return this->group_pilot_sc_ ? sc_id - (sc_id % this->pilot_sc_group_size_)
                                 : sc_id;
  }

  /// Get the calibration buffer for this frame and subcarrier ID
  inline complex_float* GetCalibBuffer(Table<complex_float>& calib_buffer,
                                       size_t frame_id, size_t sc_id) const {
    size_t frame_slot = frame_id % kFrameWnd;
    return &calib_buffer[frame_slot][sc_id * bs_ant_num_];
  }

  /// Get mac bits for this frame, symbol, user and code block ID
  inline int8_t* GetMacBits(Table<int8_t>& info_bits, Direction dir,
                            size_t frame_id, size_t symbol_id, size_t ue_id,
                            size_t cb_id) const {
    size_t mac_bytes_perframe;
    size_t num_bytes_per_cb;
    size_t mac_packet_length;
    if (dir == Direction::kDownlink) {
      mac_bytes_perframe = this->dl_mac_bytes_num_perframe_;
      num_bytes_per_cb = this->dl_num_bytes_per_cb_;
      mac_packet_length = this->dl_mac_packet_length_;
    } else {
      mac_bytes_perframe = ul_mac_bytes_num_perframe_;
      num_bytes_per_cb = this->ul_num_bytes_per_cb_;
      mac_packet_length = this->ul_mac_packet_length_;
    }
    return &info_bits[ue_id][(frame_id % kFrameWnd) * mac_bytes_perframe +
                             symbol_id * mac_packet_length +
                             cb_id * num_bytes_per_cb];
  }

  /// Get info bits for this symbol, user and code block ID
  inline int8_t* GetInfoBits(Table<int8_t>& info_bits, Direction dir,
                             size_t symbol_id, size_t ue_id,
                             size_t cb_id) const {
    size_t num_bytes_per_cb;
    size_t num_blocks_in_symbol;
    if (dir == Direction::kDownlink) {
      num_bytes_per_cb = this->dl_num_bytes_per_cb_;
      num_blocks_in_symbol = this->dl_ldpc_config_.NumBlocksInSymbol();
    } else {
      num_bytes_per_cb = this->ul_num_bytes_per_cb_;
      num_blocks_in_symbol = this->ul_ldpc_config_.NumBlocksInSymbol();
    }
    return &info_bits[symbol_id][Roundup<64>(num_bytes_per_cb) *
                                 (num_blocks_in_symbol * ue_id + cb_id)];
  }

  /// Get encoded_buffer for this frame, symbol, user and code block ID
  inline int8_t* GetModBitsBuf(Table<int8_t>& mod_bits_buffer, Direction dir,
                               size_t frame_id, size_t symbol_id, size_t ue_id,
                               size_t sc_id) const {
    size_t total_data_symbol_id;
    size_t ofdm_data_num;
    if (dir == Direction::kDownlink) {
      total_data_symbol_id = GetTotalDataSymbolIdxDl(frame_id, symbol_id);
      ofdm_data_num = GetOFDMDataNum();
    } else {
      total_data_symbol_id = GetTotalDataSymbolIdxUl(frame_id, symbol_id);
      ofdm_data_num = this->ofdm_data_num_;
    }

    return &mod_bits_buffer[total_data_symbol_id]
                           [Roundup<64>(ofdm_data_num) * ue_id + sc_id];
  }

  // Returns the number of pilot subcarriers in downlink symbols used for
  // phase tracking
  inline size_t GetOFDMPilotNum() const {
    return ofdm_data_num_ / ofdm_pilot_spacing_;
  }

  inline size_t GetOFDMDataNum() const {
    return ofdm_data_num_ - GetOFDMPilotNum();
  }

  inline size_t GetOFDMCtrlNum() const {
    return ofdm_data_num_ - 2 * GetOFDMPilotNum();
  }

  inline size_t GetOFDMDataIndex(size_t sc_id) const {
    return dl_symbol_data_id_.at(sc_id);
  }

  inline size_t GetOFDMCtrlIndex(size_t sc_id) const {
    return dl_symbol_ctrl_id_.at(sc_id);
  }

  inline bool IsDataSubcarrier(size_t sc_id) const {
    return dl_symbol_map_.at(sc_id) == SubcarrierType::kData;
  }
  inline bool IsControlSubcarrier(size_t sc_id) const {
    return control_symbol_map_.at(sc_id) == SubcarrierType::kData;
  }
  inline const std::string& ConfigFilename() const { return config_filename_; }
  inline const std::string& TraceFilename() const { return trace_file_; }
  inline const std::string& Timestamp() const { return timestamp_; }
  inline const std::vector<std::string>& UlTxFreqDataFiles() const {
    return ul_tx_f_data_files_;
  }

 private:
  void Print() const;
  nlohmann::json Parse(const nlohmann::json& in_json,
                       const std::string& json_handle);
  void DumpMcsInfo();

  /* Class constants */
  inline static const size_t kDefaultSymbolNumPerFrame = 70;
  inline static const size_t kDefaultFreqOrthPilotSymbolNum = 1;
  inline static const size_t kDefaultULSymPerFrame = 30;
  inline static const size_t kDefaultULSymStart = 9;
  inline static const size_t kDefaultDLSymPerFrame = 30;
  inline static const size_t kDefaultDLSymStart = 40;

  // Number of code blocks per OFDM symbol
  // Temporarily set to 1
  // TODO: This number should independent of OFDM symbols
  static constexpr size_t kCbPerSymbol = 1;

  /* Private class variables */
  const double freq_ghz_;  // RDTSC frequency in GHz

  size_t bs_ant_num_;  // Total number of BS antennas
  size_t bf_ant_num_;  // Number of antennas used in beamforming

  // The count of ues an instance is responsable for
  size_t ue_num_;
  // The count of ue antennas an instance is responsable for
  size_t ue_ant_num_;

  // Total number of us antennas in this experiment including the ones
  // instantiated on other runs/machines.
  size_t ue_ant_total_;
  // The offset into the number total ue antennas this instantiation is
  // responsible for.
  size_t ue_ant_offset_;

  // The total number of OFDM subcarriers, which is a power of two
  size_t ofdm_ca_num_;
  size_t num_client_ul_pilot;

  // The number of cyclic prefix IQ samples. These are taken from the tail of
  // the time-domain OFDM samples and prepended to the beginning.
  size_t cp_len_;

  // The number of OFDM subcarriers that are non-zero in the frequency domain
  size_t ofdm_data_num_;

  // The index of the first non-zero OFDM subcarrier (in the frequency domain)
  // in block of ofdm_ca_num_ subcarriers.
  size_t ofdm_data_start_;

  // The index of the last non-zero OFDM subcarrier (in the frequency domain)
  // in block of ofdm_ca_num_ subcarriers.
  size_t ofdm_data_stop_;

  size_t ofdm_pilot_spacing_;

  std::string ul_modulation_;  // Modulation order as a string, e.g., "16QAM"
  size_t
      ul_mod_order_bits_;  // Number of binary bits used for a modulation order
  std::string dl_modulation_;
  size_t dl_mod_order_bits_;
  size_t dl_bcast_mod_order_bits_;

  // Modulation lookup table for mapping binary bits to constellation points
  Table<complex_float> ul_mod_table_;
  Table<complex_float> dl_mod_table_;

  LDPCconfig ul_ldpc_config_;        // Uplink LDPC parameters
  LDPCconfig dl_ldpc_config_;        // Downlink LDPC parameters
  LDPCconfig dl_bcast_ldpc_config_;  // Downlink Broadcast LDPC parameters
  nlohmann::json ul_mcs_params_;     // Uplink Modulation and Coding (MCS)
  nlohmann::json dl_mcs_params_;     // Downlink Modulation and Coding (MCS)
  size_t ul_mcs_index_;
  size_t dl_mcs_index_;
  size_t dl_code_rate_;
  size_t ul_code_rate_;
  bool scramble_enabled_;

  // A class that holds the frame configuration the id contains letters
  // representing the symbol types in the frame (e.g., 'P' for pilot symbols,
  // 'U' for uplink data symbols)
  FrameStats frame_;

  std::atomic<bool> running_;

  size_t dl_packet_length_;  // HAS_TIME & END_BURST, fixme

  std::vector<SubcarrierType> ul_symbol_map_;
  std::vector<SubcarrierType> dl_symbol_map_;
  std::vector<SubcarrierType> control_symbol_map_;
  std::vector<size_t> dl_symbol_data_id_;
  std::vector<size_t> dl_symbol_ctrl_id_;

  Table<int8_t> dl_bits_;
  Table<int8_t> ul_bits_;
  Table<int8_t> ul_mod_bits_;
  Table<int8_t> dl_mod_bits_;
  Table<complex_float> dl_iq_f_;
  Table<complex_float> ul_iq_f_;
  Table<std::complex<int16_t>> dl_iq_t_;
  Table<std::complex<int16_t>> ul_iq_t_;

  std::vector<std::complex<float>> gold_cf32_;
  std::vector<std::complex<int16_t>> beacon_ci16_;
  std::vector<uint32_t> coeffs_;

  /// I/Q samples of common pilot
  std::vector<std::complex<int16_t>> pilot_ci16_;

  std::vector<std::complex<float>> pilot_cf32_;

  /// I/Q samples of pilots per UE antenna per pilot symbol
  std::vector<std::vector<std::vector<std::complex<int16_t>>>> pilot_ue_ci16_;

  /// List of subcarriers used per UE to transmit pilot
  std::vector<arma::uvec> pilot_ue_sc_;

  std::vector<uint32_t> pilot_;
  std::vector<uint32_t> beacon_;
  complex_float* pilots_;
  complex_float* pilots_sgn_;
  complex_float* pilot_ifft_;
  Table<complex_float> ue_specific_pilot_;
  Table<complex_float> ue_pilot_ifft_;
  Table<std::complex<int16_t>> ue_specific_pilot_t_;
  std::vector<std::complex<float>> common_pilot_;

  std::vector<double> client_gain_tx_a_;
  std::vector<double> client_gain_tx_b_;
  std::vector<double> client_gain_rx_a_;
  std::vector<double> client_gain_rx_b_;

  std::vector<std::string> radio_id_;
  std::vector<std::string> hub_id_;
  std::vector<std::string> ue_radio_id_;
  std::vector<std::string> ue_radio_name_;
  std::vector<size_t> ref_radio_;
  std::vector<size_t> ref_ant_;
  std::vector<size_t> cell_id_;

  // Controls whether the synchronization and frame time keeping is done
  // in hardware or software
  // true: use hardware correlator; false: use software corrleator
  bool hw_framer_;
  bool ue_hw_framer_;
  size_t ue_resync_period_;

  double freq_;
  double rate_;
  double nco_;
  double radio_rf_freq_;
  double bw_filter_;
  bool single_gain_;
  double tx_gain_a_;
  double rx_gain_a_;
  double tx_gain_b_;
  double rx_gain_b_;
  double calib_tx_gain_a_;
  double calib_tx_gain_b_;
  std::vector<double> client_tx_gain_a_;
  std::vector<double> client_rx_gain_a_;
  std::vector<double> client_tx_gain_b_;
  std::vector<double> client_rx_gain_b_;

  size_t num_cells_;
  size_t num_radios_;
  size_t num_channels_;
  size_t num_ue_channels_;
  size_t beacon_ant_;
  size_t beacon_len_;
  size_t init_calib_repeat_;
  bool smooth_calib_;
  bool beamsweep_;
  bool sample_cal_en_;
  bool imbalance_cal_en_;
  size_t beamforming_algo_;
  size_t num_spatial_streams_;
  std::string beamforming_str_;
  std::vector<bool> external_ref_node_;
  std::string channel_;
  std::string ue_channel_;

  size_t core_offset_;
  size_t worker_thread_num_;
  size_t socket_thread_num_;
  size_t fft_thread_num_;
  size_t demul_thread_num_;
  size_t decode_thread_num_;
  size_t beam_thread_num_;

  size_t ue_core_offset_;
  size_t ue_worker_thread_num_;
  size_t ue_socket_thread_num_;

  // If true, accelerate small MIMO such as 1x1, 2x2, and 4x4. Vector operations
  // are done across subcarriers instead of looping through each subcarrier.
  // This is useful for performance optimization and is only implemented for
  // some antenna config (1x1, 2x2, 4x4) in DoBeamweight.cc and DeDemul.cc.
  bool small_mimo_acc_;

  // Number of OFDM data subcarriers handled in one demodulation event
  size_t demul_block_size_;
  // Derived from demul_block_size
  size_t demul_events_per_symbol_;

  /// Number of OFDM data subcarriers handled in 1 kBeam event
  size_t beam_block_size_;
  /// Beam Events generated per Frame.  Derived from beam_block_size
  size_t beam_events_per_symbol_;

  // Number of antennas handled in one FFT event
  size_t fft_block_size_;

  // Number of code blocks handled in one encode event
  size_t encode_block_size_;

  // Whether to enable frequency orthogonal pilot
  bool freq_orthogonal_pilot_;

  // Whether to group subcarriers in pilot symbol
  bool group_pilot_sc_;

  // Frequency orthogonal pilot subcarrier group size
  size_t pilot_sc_group_size_;

  // The number of zero IQ samples prepended to a time-domain symbol (i.e.,
  // before the cyclic prefix) before transmission. Its value depends on
  // over-the-air and RF delays, and is currently calculated by manual tuning.
  size_t ofdm_tx_zero_prefix_;

  // The number of zero IQ samples appended to a time-domain symbol before
  // transmission. Its value depends on over-the-air and RF delays, and is
  // currently calculated by manual tuning.
  size_t ofdm_tx_zero_postfix_;

  // The number of IQ samples to skip from the beginning of symbol received by
  // Agora on the uplink at the base station. Due to over-the-air and RF
  // delays, this can be different from (prefix + cp_len_), and is currently
  // calculated by manual tuning.
  size_t ofdm_rx_zero_prefix_bs_;

  size_t ofdm_rx_zero_prefix_cal_ul_;
  size_t ofdm_rx_zero_prefix_cal_dl_;

  // The number of IQ samples to skip from the beginning of symbol received by
  // Agora on the downlink at the client. Due to over-the-air and RF
  // delays, this can be different from (prefix + cp_len_), and is currently
  // calculated by manual tuning.
  size_t ofdm_rx_zero_prefix_client_;

  // The total number of IQ samples in one physical layer time-domain packet
  // received or sent by Agora
  size_t samps_per_symbol_;

  // The number of bytes in one physical layer time-domain packet received or
  // sent by Agora. This includes Agora's packet header, but not the
  // Ethernet/IP/UDP headers.
  size_t packet_length_;

  std::vector<int> cl_tx_advance_;
  std::vector<float> cl_corr_scale_;

  float scale_;  // Scaling factor for all transmit symbols

  bool bigstation_mode_;      // If true, use pipeline-parallel scheduling
  bool correct_phase_shift_;  // If true, do phase shift correction

  // The total number of uncoded uplink data bytes in each OFDM symbol
  size_t ul_data_bytes_num_persymbol_;

  // The total number of uplink MAC payload data bytes in each Frame
  size_t ul_mac_data_bytes_num_perframe_;

  // The total number of uplink MAC packet bytes in each Frame
  size_t ul_mac_bytes_num_perframe_;

  // The length (in bytes) of a uplink MAC packet including the header
  size_t ul_mac_packet_length_;

  // The length (in bytes) of a uplink MAC packet payload (data)
  size_t ul_mac_data_length_max_;

  // The total number of uncoded downlink data bytes in each OFDM symbol
  size_t dl_data_bytes_num_persymbol_;

  // The total number of downlink MAC payload data bytes in each Frame
  size_t dl_mac_data_bytes_num_perframe_;

  // The total number of downlink MAC packet bytes in each Frame
  size_t dl_mac_bytes_num_perframe_;

  // The length (in bytes) of a downlink MAC packet including the header
  size_t dl_mac_packet_length_;

  // The length (in bytes) of a downlink MAC packet payload (data)
  size_t dl_mac_data_length_max_;

  // The total number of downlink mac packets sent/received in each frame
  size_t dl_mac_packets_perframe_;

  // The total number of uplink mac packets sent/received in each frame
  size_t ul_mac_packets_perframe_;

  // IP address of the machine running the baseband processing for UE
  std::string ue_server_addr_;

  // IP address of the machine running the baseband processing for BS
  std::string bs_server_addr_;

  // IP address of the base station RRU, RRU emulator (sender),
  // or channel simulator
  std::string bs_rru_addr_;

  // IP address of the Ue RRU, RRU emulator (sender), or channel simulator
  // could be multiple addresses
  std::string ue_rru_addr_;

  // IP address of the data source/sink server communicating with MAC (BS/UE)
  std::string mac_remote_addr_;

  // IP address of the listening server for runtime stats log
  std::string log_listener_addr_;

  int bs_server_port_;  // Base UDP port used by BS to receive data

  // Base RRU/channel simulator UDP port used by BS to transmit downlink data
  int bs_rru_port_;

  int ue_server_port_;  // Base UDP port used by UEs to receive data

  // Base RRU/channel simulator UDP port used by UEs to transmit uplink data
  int ue_rru_port_;

  // Number of NIC ports used for DPDK
  uint16_t dpdk_num_ports_;

  // Offset of the first NIC port used by Agora's DPDK mode
  uint16_t dpdk_port_offset_;

  // MAC addresses of NIC ports separated by ';'
  std::string dpdk_mac_addrs_;

  // Port ID at BaseStation MAC layer side
  size_t bs_mac_rx_port_;
  size_t bs_mac_tx_port_;

  // Port ID at Client MAC layer side
  size_t ue_mac_rx_port_;
  size_t ue_mac_tx_port_;

  // Port ID at log listening server
  size_t log_listener_port_;

  // Number of logged subcarrier data samples
  size_t log_sc_num_;

  // Whether use unique timestamp as subdirectory of csv log files
  bool log_timestamp_;

  // Number of frames_ sent by sender during testing = number of frames_
  // processed by Agora before exiting.
  size_t frames_to_test_;

  // Size of tranport block given by upper layer
  size_t transport_block_size_;

  float noise_level_;

  // Number of bytes per code block
  size_t ul_num_bytes_per_cb_;
  size_t dl_num_bytes_per_cb_;

  // Number of padding bytes per code block
  size_t ul_num_padding_bytes_per_cb_;
  size_t dl_num_padding_bytes_per_cb_;

  bool fft_in_rru_;  // If true, the RRU does FFT instead of Agora
  const std::string config_filename_;
  std::string trace_file_;
  std::string timestamp_;
  std::vector<std::string> ul_tx_f_data_files_;
};
#endif /* CONFIG_HPP_ */
