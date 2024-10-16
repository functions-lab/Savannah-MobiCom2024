/**
 * @file dodecode_acc.cc
 * @brief Implmentation file for the DoDecode class with ACC100 acceleration. 
 */

#include "dodecode_acc.h"

#include "concurrent_queue_wrapper.h"
#include "rte_bbdev.h"
#include "rte_bbdev_op.h"
#include "rte_bus_vdev.h"

#define GET_SOCKET(socket_id) (((socket_id) == SOCKET_ID_ANY) ? 0 : (socket_id))
#define MAX_RX_BYTE_SIZE 1500
#define CACHE_SIZE 128
#define NUM_QUEUES 4
#define LCORE_ID 36
#define NUM_ELEMENTS_IN_POOL 2047
#define NUM_ELEMENTS_IN_MEMPOOL 16383
#define DATA_ROOM_SIZE 45488
static constexpr bool kPrintLLRData = false;
static constexpr bool kPrintDecodedData = false;
static constexpr bool kPrintACC100Byte = false;
static constexpr bool kPrintTxByte = false;
static constexpr bool kPrintMbufData = false;
static constexpr bool kMubfLLRCheck = false;

static void
ldpc_input_llr_scaling(struct rte_bbdev_op_data *input_ops,  const int8_t llr_size,
		const int8_t llr_decimals)
{
	if (input_ops == NULL)
		return;

	uint16_t i = 0, byte_idx;
	int16_t llr_max, llr_min, llr_tmp;
  // std::cout << "in ldpc_input_llr_scaling input_ops is not null !!!!!!!!!\n";
	llr_max = (1 << (llr_size - 1)) - 1;
	llr_min = -llr_max;
		struct rte_mbuf *m = input_ops[i].data;
		while (m != NULL) {
			int8_t *llr = rte_pktmbuf_mtod_offset(m, int8_t *,
					input_ops[i].offset);
			for (byte_idx = 0; byte_idx < rte_pktmbuf_data_len(m);
					++byte_idx) {

				llr_tmp = llr[byte_idx];
				if (llr_decimals == 4)
					llr_tmp *= 8;
				else if (llr_decimals == 2)
					llr_tmp *= 2;
				else if (llr_decimals == 0)
					llr_tmp /= 2;
				llr_tmp = RTE_MIN(llr_max,
						RTE_MAX(llr_min, llr_tmp));
				llr[byte_idx] = (int8_t) llr_tmp;
			}

			m = m->next;
		}
}

static constexpr size_t kVarNodesSize = 1024 * 1024 * sizeof(int16_t);

static unsigned int optimal_mempool_size(unsigned int val) {
  return rte_align32pow2(val + 1) - 1;
}

void print_uint32(const uint32_t* array, size_t totalByteLength) {
  size_t totalWordLength = (totalByteLength + 3) / 4;

  for (int i = 0; i < (int)totalWordLength; i++) {
    // Extract and print the byte
    printf("%08X ", array[i]);
  }
}

void print_casted_uint8_hex(const uint8_t* array, size_t totalByteLength) {
  for (size_t i = 0; i < totalByteLength; i++) {
    // Cast each int8_t to uint8_t and print as two-digit hexadecimal
    printf("%02X ", (uint8_t)array[i]);
  }
  printf("\n");  // Add a newline for readability
}

DoDecode_ACC::DoDecode_ACC(
    Config *in_config, int in_tid,
    PtrCube<kFrameWnd, kMaxSymbols, kMaxUEs, int8_t> &demod_buffers,
    // PtrCube<kFrameWnd, kMaxSymbols, kMaxUEs, uint32_t> &llr_buffers,
    PtrCube<kFrameWnd, kMaxSymbols, kMaxUEs, int8_t> &decoded_buffers,
    PhyStats *in_phy_stats, Stats *in_stats_manager)
    : Doer(in_config, in_tid),
      demod_buffers_(demod_buffers),
      // llr_buffers_(llr_buffers),
      decoded_buffers_(decoded_buffers),
      phy_stats_(in_phy_stats),
      scrambler_(std::make_unique<AgoraScrambler::Scrambler>()) {
  duration_stat_ = in_stats_manager->GetDurationStat(DoerType::kDecode, in_tid);
  resp_var_nodes_ = static_cast<int16_t *>(Agora_memory::PaddedAlignedAlloc(
      Agora_memory::Alignment_t::kAlign64, kVarNodesSize));
  std::string core_list = std::to_string(LCORE_ID);  // this is hard set to core 36
  const size_t num_ul_syms = cfg_->Frame().NumULSyms(); 
  const size_t num_ue = cfg_->UeAntNum();

  const char *rte_argv[] = {"txrx",        "-l",           core_list.c_str(),
                            "--log-level", "lib.eal:info", nullptr};
  int rte_argc = static_cast<int>(sizeof(rte_argv) / sizeof(rte_argv[0])) - 1;

  // Initialize DPDK environment
  int ret = rte_eal_init(rte_argc, const_cast<char **>(rte_argv));
  RtAssert(
      ret >= 0,
      "Failed to initialize DPDK.  Are you running with root permissions?");

  int nb_bbdevs = rte_bbdev_count();
  std::cout << "num bbdevs: " << nb_bbdevs << std::endl;

  if (nb_bbdevs == 0) rte_exit(EXIT_FAILURE, "No bbdevs detected!\n");
  dev_id = 0;
  struct rte_bbdev_info info;
  // rte_bbdev_info_get(dev_id, &info);
  rte_bbdev_intr_enable(dev_id);
  rte_bbdev_info_get(dev_id, &info);

  bbdev_op_pool =
      rte_bbdev_op_pool_create("bbdev_op_pool_dec", RTE_BBDEV_OP_LDPC_DEC,
                               NB_MBUF, CACHE_SIZE , rte_socket_id());
  ret = rte_bbdev_setup_queues(dev_id, NUM_QUEUES, info.socket_id);

  if (ret < 0) {
    printf("rte_bbdev_setup_queues(%u, %u, %d) ret %i\n", dev_id, NUM_QUEUES,
           rte_socket_id(), ret);
  }

  ret = rte_bbdev_intr_enable(dev_id);

  struct rte_bbdev_queue_conf qconf;
  qconf.socket = info.socket_id;
  qconf.queue_size = info.drv.queue_size_lim;
  qconf.op_type = RTE_BBDEV_OP_LDPC_DEC;
  qconf.priority = 0;

  std::cout << "device id is: " << static_cast<int>(dev_id) << std::endl;

  for (int q_id = 0; q_id < NUM_QUEUES; q_id++) {
    /* Configure all queues belonging to this bbdev device */
    ret = rte_bbdev_queue_configure(dev_id, q_id, &qconf);
    if (ret < 0)
      rte_exit(EXIT_FAILURE,
               "ERROR(%d): BBDEV %u queue %u not configured properly\n", ret,
               dev_id, q_id);
  }

  ret = rte_bbdev_start(dev_id);
  int socket_id = GET_SOCKET(info.socket_id);

  ops_mp = rte_bbdev_op_pool_create("RTE_BBDEV_OP_LDPC_DEC_poo",
                                    RTE_BBDEV_OP_LDPC_DEC, NUM_ELEMENTS_IN_POOL, OPS_CACHE_SIZE,
                                    socket_id);
  if (ops_mp == nullptr) {
    std::cerr << "Error: Failed to create memory pool for bbdev operations."
              << std::endl;
  } else {
    std::cout << "Memory pool for bbdev operations created successfully."
              << std::endl;
  }

  in_mbuf_pool = rte_pktmbuf_pool_create("in_pool_0", NUM_ELEMENTS_IN_MEMPOOL, 0, 0, DATA_ROOM_SIZE, 0);
  out_mbuf_pool =
      rte_pktmbuf_pool_create("hard_out_pool_0", NUM_ELEMENTS_IN_MEMPOOL, 0, 0, DATA_ROOM_SIZE, 0);

  if (in_mbuf_pool == nullptr or out_mbuf_pool == nullptr) {
    std::cerr << "Error: Unable to create mbuf pool: "
              << rte_strerror(rte_errno) << std::endl;
  }

  int rte_alloc_ref = rte_bbdev_dec_op_alloc_bulk(ops_mp, ref_dec_op, num_ul_syms * num_ue);
  if (rte_alloc_ref != TEST_SUCCESS ) {
    rte_exit(EXIT_FAILURE, "Failed to alloc bulk\n");
  }

  const struct rte_bbdev_op_cap *cap = info.drv.capabilities;
  const struct rte_bbdev_op_cap *capabilities = NULL;
  rte_bbdev_info_get(dev_id, &info);
  for (unsigned int i = 0; cap->type != RTE_BBDEV_OP_NONE; ++i, ++cap) {
    std::cout << "cap is: " << cap->type << std::endl;
    if (cap->type == RTE_BBDEV_OP_LDPC_DEC) {
      capabilities = cap;
      std::cout << "capability is being set to: " << capabilities->type
                << std::endl;
      break;
    }
  }

  inputs =
      (struct rte_bbdev_op_data **)malloc(sizeof(struct rte_bbdev_op_data *));
  hard_outputs =
      (struct rte_bbdev_op_data **)malloc(sizeof(struct rte_bbdev_op_data *));

  int ret_socket_in = allocate_buffers_on_socket(
      inputs, 1 * sizeof(struct rte_bbdev_op_data), 0);
  int ret_socket_hard_out = allocate_buffers_on_socket(
      hard_outputs, 1 * sizeof(struct rte_bbdev_op_data), 0);

  if (ret_socket_in != TEST_SUCCESS || ret_socket_hard_out != TEST_SUCCESS) {
    rte_exit(EXIT_FAILURE, "Failed to allocate socket\n");
  }

  ldpc_llr_decimals = capabilities->cap.ldpc_dec.llr_decimals;
  ldpc_llr_size = capabilities->cap.ldpc_dec.llr_size;
  ldpc_cap_flags = capabilities->cap.ldpc_dec.capability_flags;

  min_alignment = info.drv.min_alignment;

  const LDPCconfig &ldpc_config = cfg_->LdpcConfig(Direction::kUplink);

  int iter_num = num_ul_syms * num_ue;
  q_m = cfg_->ModOrderBits(Direction::kUplink);
  e = ldpc_config.NumCbCodewLen();

  for (int i = 0; i < iter_num; i++) {
  // ref_dec_op[i]->ldpc_dec.op_flags += RTE_BBDEV_LDPC_ITERATION_STOP_ENABLE;
    ref_dec_op[i]->ldpc_dec.basegraph = (uint8_t)ldpc_config.BaseGraph();
    ref_dec_op[i]->ldpc_dec.z_c = (uint16_t)ldpc_config.ExpansionFactor();
    ref_dec_op[i]->ldpc_dec.n_filler = (uint16_t)0;
    ref_dec_op[i]->ldpc_dec.rv_index = (uint8_t)0;
    ref_dec_op[i]->ldpc_dec.n_cb = (uint16_t)ldpc_config.NumCbCodewLen();
    // std::cout << "n_cb is: " << (uint16_t)ldpc_config.NumCbCodewLen() <<  std::endl;
    ref_dec_op[i]->ldpc_dec.q_m = (uint8_t)q_m;
    ref_dec_op[i]->ldpc_dec.code_block_mode = (uint8_t)1;
    ref_dec_op[i]->ldpc_dec.cb_params.e = (uint32_t)e;
    // std::cout << "e is: " << (uint32_t)e <<  std::endl;
    if (!check_bit(ref_dec_op[i]->ldpc_dec.op_flags,
                  RTE_BBDEV_LDPC_ITERATION_STOP_ENABLE)) {
      ref_dec_op[i]->ldpc_dec.op_flags += RTE_BBDEV_LDPC_ITERATION_STOP_ENABLE;
    }
    ref_dec_op[i]->ldpc_dec.iter_max = (uint8_t)ldpc_config.MaxDecoderIter();
    ref_dec_op[i]->opaque_data = (void *)(uintptr_t)i;
  }
  std::cout << "" << std::endl;
  AGORA_LOG_INFO("rte_pktmbuf_alloc successful\n");
}

DoDecode_ACC::~DoDecode_ACC() {
  std::free(resp_var_nodes_);
}

int DoDecode_ACC::allocate_buffers_on_socket(struct rte_bbdev_op_data **buffers,
                                             const int len, const int socket) {
  int i;
  *buffers = static_cast<struct rte_bbdev_op_data *>(
      rte_zmalloc_socket(NULL, len, 0, socket));
  if (*buffers == NULL) {
    printf("WARNING: Failed to allocate op_data on socket %d\n", socket);
    /* try to allocate memory on other detected sockets */
    for (i = 0; i < socket; i++) {
      *buffers = static_cast<struct rte_bbdev_op_data *>(
          rte_zmalloc_socket(NULL, len, 0, i));
      if (*buffers != NULL) break;
    }
  }
  return (*buffers == NULL) ? TEST_FAILED : TEST_SUCCESS;
}

EventData DoDecode_ACC::Launch(size_t tag) {
  const LDPCconfig &ldpc_config = cfg_->LdpcConfig(Direction::kUplink);
  const size_t frame_id = gen_tag_t(tag).frame_id_;
  const size_t symbol_id = gen_tag_t(tag).symbol_id_;
  const size_t symbol_idx_ul = cfg_->Frame().GetULSymbolIdx(symbol_id);
  const size_t num_ul_syms = cfg_->Frame().NumULSyms();  
  const size_t cb_id = gen_tag_t(tag).cb_id_;
  const size_t symbol_offset =
      cfg_->GetTotalDataSymbolIdxUl(frame_id, symbol_idx_ul);
  const size_t cur_cb_id = (cb_id % ldpc_config.NumBlocksInSymbol());
  const size_t ue_id = (cb_id / ldpc_config.NumBlocksInSymbol());
  const size_t num_ue = cfg_->UeAntNum();
  const size_t frame_slot = (frame_id % kFrameWnd);
  const size_t num_bytes_per_cb = cfg_->NumBytesPerCb(Direction::kUplink);
  if (kDebugPrintInTask == true) {
    std::printf(
        "In doDecode thread %d: frame: %zu, symbol: %zu, code block: "
        "%zu, ue: %zu offset %zu\n",
        tid_, frame_id, symbol_id, cur_cb_id, ue_id, symbol_offset);
  }
  
#if defined(ENQUEUE_BULK)
  if ((symbol_idx_ul == num_ul_syms - 1) && (ue_id == num_ue - 1)) {
    // std::cout<<"[In If]callling doDecode launch, Frame id is " << frame_id << " symbol id is: " << symbol_id <<std::endl;
    size_t start_tsc = GetTime::WorkerRdtsc();

    int8_t *llr_buffer_ptr;
    struct rte_mbuf *m_head;
    struct rte_mbuf *m_head_out;
    size_t index = 0;

    for (size_t temp_ue_id = 0; temp_ue_id < num_ue; temp_ue_id++){
      for (size_t temp_idx = 0; temp_idx < num_ul_syms; temp_idx++){
        llr_buffer_ptr = demod_buffers_[frame_slot][temp_idx][temp_ue_id] +
                                (cfg_->ModOrderBits(Direction::kUplink) *
                                  (ldpc_config.NumCbCodewLen() * cur_cb_id));
        
        int8_t* ref_byte_cp = cfg_->GetInfoBits(cfg_->UlBits(), Direction::kUplink, temp_idx, temp_ue_id, cur_cb_id);
        uint8_t* ref_byte_new = reinterpret_cast<uint8_t*>(ref_byte_cp);

        char *data;
        struct rte_bbdev_op_data *bufs = *inputs;
        m_head = rte_pktmbuf_alloc(in_mbuf_pool);

        bufs[0].data = m_head;
        bufs[0].offset = 0;
        bufs[0].length = 0;

        data = rte_pktmbuf_append(m_head, ldpc_config.NumCbCodewLen());

        // Copy data from demod_data to the mbuf
        rte_memcpy(data, llr_buffer_ptr + (0 * ldpc_config.NumCbCodewLen()),
                  ldpc_config.NumCbCodewLen());
        bufs[0].length += ldpc_config.NumCbCodewLen();

        rte_bbdev_op_data *bufs_out = *hard_outputs;
        m_head_out = rte_pktmbuf_alloc(out_mbuf_pool);

        bufs_out[0].data = m_head_out;
        bufs_out[0].offset = 0;
        bufs_out[0].length = 0;

        // Prepare the mbuf to receive the output data
        char *data_out = rte_pktmbuf_append(m_head_out, ldpc_config.NumCbCodewLen());
        assert(data_out == RTE_PTR_ALIGN(data_out, min_alignment));

        // BUG: This line causes a irregular stop of the program when fft_size = 4096,
        //      ofdm_data_num = 3168 = demul_block_size = beam_block_size, SISO, any
        //      sampling rate.
        // Assuming you will copy data from decoded_buffer_ptr to data
        rte_memcpy(data_out, ref_byte_new, ldpc_config.NumCbCodewLen());

        bufs_out[0].length += ldpc_config.NumCbCodewLen();
  
        ref_dec_op[index]->ldpc_dec.input = *inputs[0];
        ref_dec_op[index]->ldpc_dec.hard_output = *hard_outputs[0];

        index++;
        rte_pktmbuf_free(m_head);
        rte_pktmbuf_free(m_head_out);
      }
  }

    size_t start_tsc1 = GetTime::WorkerRdtsc();
    duration_stat_->task_duration_[1] += start_tsc1 - start_tsc;

    uint16_t enq = 0, deq = 0;
    uint64_t start_time = 0, last_time = 0;
    
    for (enq = 0, deq = 0; enq < (num_ul_syms * num_ue);) {
      enq += rte_bbdev_enqueue_ldpc_dec_ops(0, 0, &ref_dec_op[enq], 1);
      deq += rte_bbdev_dequeue_ldpc_dec_ops(0, 0, &ops_deq[deq], enq - deq);
    }

    int retry_count = 0;

    while (deq < enq && retry_count < MAX_DEQUEUE_TRIAL) {
      // rte_delay_ms(10);  // Wait for 10 milliseconds
      deq += rte_bbdev_dequeue_ldpc_dec_ops(0, 0, &ops_deq[deq], enq - deq);
      retry_count++;
    }

    AGORA_LOG_INFO("ACC100: enq = %d, deq = %d\n", enq, deq);
    // if (cfg_->ScrambleEnabled()) {
    //   scrambler_->Descramble(decoded_buffer_ptr, num_bytes_per_cb);
    // }

    size_t start_tsc2 = GetTime::WorkerRdtsc();
    duration_stat_->task_duration_[2] += start_tsc2 - start_tsc1;

    size_t BLER(0);
    uint8_t rx_byte[MAX_RX_BYTE_SIZE];
    int8_t* ref_byte;
    uint8_t tx_byte;
    uint32_t tx_word;
    size_t block_error(0);

    if ((kEnableMac == false) && (kPrintPhyStats == true) &&
        (symbol_idx_ul >= cfg_->Frame().ClientUlPilotSymbols())) {
      // get mbuf from the ops_deq

      struct rte_bbdev_op_ldpc_dec *ops_td;
      unsigned int i = 2;
      struct rte_bbdev_op_data *hard_output;
      struct rte_mbuf *temp_m;
      size_t offset = 0;

      for (size_t temp_ue_id = 0; temp_ue_id < num_ue; temp_ue_id++){
        size_t BLER(0);
        for (size_t temp_idx = 2; temp_idx < num_ul_syms; temp_idx++){
          ref_byte = 
            cfg_->GetInfoBits(cfg_->UlBits(), Direction::kUplink, temp_idx,
                              temp_ue_id, cur_cb_id);
          tx_byte = static_cast<uint8_t>(
            cfg_->GetInfoBits(cfg_->UlBits(), Direction::kUplink, temp_idx,
                              temp_ue_id, cur_cb_id)[i]);
          phy_stats_->UpdateBitErrors(temp_ue_id, symbol_offset, frame_slot,
                                      tx_byte, tx_byte);
          // tx_word = static_cast<uint8_t>(tx_byte);

          ops_td = &ops_deq[i]->ldpc_dec;
          hard_output = &ops_td->hard_output;
          temp_m = hard_output->data;
          uint8_t* temp_data = rte_pktmbuf_mtod(temp_m, uint8_t*);

          // If temp_m contains multiple bytes, use memcmp to compare
          if (kPrintACC100Byte){
            if (frame_id > 100 && frame_id % 100 == 0) {
              std::cout << "CB size = " << num_bytes_per_cb << " bytes\n";
              std::cout << "Content of the CB (in uint32):\n";
              print_uint32(temp_data, num_bytes_per_cb);
              std::cout << std::endl << std::endl;
            }
          }

          size_t memcmp_ret = memcmp(reinterpret_cast<uint8_t*>(temp_data), reinterpret_cast<uint8_t*>(ref_byte), num_bytes_per_cb);
        // std::cout << "memcmp_ret is: " << memcmp_ret << std::endl;
          if (memcmp_ret != 0) {
              // Data matches, do something
              block_error++;
          }
          i++;
        }
        phy_stats_->UpdateDecodedBits(temp_ue_id, symbol_offset, frame_slot,
                                      num_bytes_per_cb * 8);
        phy_stats_->IncrementDecodedBlocks(temp_ue_id, symbol_offset, frame_slot);
        // memcmp(rte_pktmbuf_mtod_offset(m, uint32_t *, 0),
        phy_stats_->UpdateBlockErrors(temp_ue_id, symbol_offset, frame_slot, block_error);
      }
    }

    size_t end = GetTime::WorkerRdtsc();
    size_t duration_3 = end - start_tsc2;
    size_t duration = end - start_tsc;

    rte_pktmbuf_free(m_head);
    rte_pktmbuf_free(m_head_out);

    duration_stat_->task_duration_[3] += duration_3;
    duration_stat_->task_duration_[0] += duration;
    // duration_stat_->task_duration_[0] += 0;

    duration_stat_->task_count_++;
    if (GetTime::CyclesToUs(duration, cfg_->FreqGhz()) > 500) {
      std::printf("Thread %d Decode takes %.2f\n", tid_,
                  GetTime::CyclesToUs(duration, cfg_->FreqGhz()));
    }

  }
#else
  if ((symbol_idx_ul == num_ul_syms - 1) && (ue_id == num_ue - 1)) {
    size_t start_tsc = GetTime::WorkerRdtsc();

    int8_t *llr_buffer_ptr;
    struct rte_mbuf *m_head;
    struct rte_mbuf *m_head_out;

    llr_buffer_ptr = demod_buffers_[frame_slot][symbol_idx_ul][ue_id] +
                            (cfg_->ModOrderBits(Direction::kUplink) *
                              (ldpc_config.NumCbCodewLen() * cur_cb_id));


    if (symbol_idx_ul == 11 && frame_id > 100 && frame_id % 100 == 0 && kPrintLLRData) {
        std::printf("LLR data, symbol_offset: %zu\n", symbol_offset);
        for (size_t i = 0; i < ldpc_config.NumCbCodewLen(); i++) {
          std::printf("%04X ", *(llr_buffer_ptr + i));
        }
        std::printf("\n");
    }

    char *data;
    struct rte_bbdev_op_data *bufs = *inputs;

    m_head = rte_pktmbuf_alloc(in_mbuf_pool);

    bufs[0].data = m_head;
    bufs[0].offset = 0;
    bufs[0].length = 0;

    data = rte_pktmbuf_append(m_head, ldpc_config.NumCbCodewLen());

    // Copy data from demod_data to the mbuf
    rte_memcpy(data, llr_buffer_ptr + (0 * ldpc_config.NumCbCodewLen()),
              ldpc_config.NumCbCodewLen());
    bufs[0].length += ldpc_config.NumCbCodewLen();

    rte_bbdev_op_data *bufs_out = *hard_outputs;
    m_head_out = rte_pktmbuf_alloc(out_mbuf_pool);

    bufs_out[0].data = m_head_out;
    bufs_out[0].offset = 0;
    bufs_out[0].length = 0;
    // BUG: This line causes a irregular stop of the program when fft_size = 4096,
    //      ofdm_data_num = 3168 = demul_block_size = beam_block_size, SISO, any
    //      sampling rate.
    bufs_out[0].length += ldpc_config.NumCbCodewLen();

    ref_dec_op[enq_index]->ldpc_dec.input = *inputs[0];
    ref_dec_op[enq_index]->ldpc_dec.hard_output = *hard_outputs[0];

    rte_pktmbuf_free(m_head);
    rte_pktmbuf_free(m_head_out);

    size_t start_tsc1 = GetTime::WorkerRdtsc();
    duration_stat_->task_duration_[1] += start_tsc1 - start_tsc;
    
    enq += rte_bbdev_enqueue_ldpc_dec_ops(0, 0, &ref_dec_op[enq], 1);

    int retry_count = 0;
    while (deq < enq && retry_count < MAX_DEQUEUE_TRIAL) {
      deq += rte_bbdev_dequeue_ldpc_dec_ops(0, 0, &ops_deq[deq], enq - deq);
      retry_count++;
    }
    AGORA_LOG_INFO("ACC100: enq = %d, deq = %d\n", enq, deq);

    enq = 0;
    deq = 0;
    enq_index = 0;

    size_t start_tsc2 = GetTime::WorkerRdtsc();
    duration_stat_->task_duration_[2] += start_tsc2 - start_tsc1;

    uint8_t rx_byte[MAX_RX_BYTE_SIZE];
    int8_t* ref_byte;
    uint8_t tx_byte;
    uint32_t tx_word;
    size_t block_error(0);


    if ((kEnableMac == false) && (kPrintPhyStats == true) &&
        (symbol_idx_ul >= cfg_->Frame().ClientUlPilotSymbols())) {
      // get mbuf from the ops_deq

      struct rte_bbdev_op_ldpc_dec *ops_td;
      unsigned int i = 2;
      struct rte_bbdev_op_data *hard_output;
      struct rte_mbuf *temp_m;
      // size_t offset = 0;

      for (size_t temp_ue_id = 0; temp_ue_id < num_ue; temp_ue_id++){
        for (size_t temp_idx = 2; temp_idx < num_ul_syms; temp_idx++){
          ref_byte = 
            cfg_->GetInfoBits(cfg_->UlBits(), Direction::kUplink, temp_idx,
                              temp_ue_id, cur_cb_id);
          tx_byte = static_cast<uint8_t>(
            cfg_->GetInfoBits(cfg_->UlBits(), Direction::kUplink, temp_idx,
                              temp_ue_id, cur_cb_id)[i]);
          // tx_word = static_cast<uint8_t>(tx_byte);

          ops_td = &ops_deq[i]->ldpc_dec;
          hard_output = &ops_td->hard_output;
          temp_m = hard_output->data;
          uint8_t* temp_data = rte_pktmbuf_mtod(temp_m, uint8_t*);
          
          // If temp_m contains multiple bytes, use memcmp to compare
          if (cfg_->ScrambleEnabled()) {
            scrambler_->Descramble(temp_data, num_bytes_per_cb);
          }
          if (kPrintACC100Byte){
            if (frame_id > 100 && frame_id % 100 == 0) {
              std::cout << "CB size = " << num_bytes_per_cb << " bytes\n";
              std::cout << "Content of the CB (in uint32):\n";
              print_casted_uint8_hex(temp_data, num_bytes_per_cb);
              // print_uint32(temp_data, num_bytes_per_cb);
              std::cout << std::endl << std::endl;
            }
          }

          if (kPrintACC100Byte){
            if (frame_id > 100 && frame_id % 100 == 0) {
              std::cout << "CB size = " << num_bytes_per_cb << " bytes\n";
              std::cout << "Content of the ref_bytes are:\n";
              print_casted_uint8_hex(reinterpret_cast<uint8_t*>(ref_byte), num_bytes_per_cb);
              // print_uint32(temp_data, num_bytes_per_cb);
              std::cout << std::endl << std::endl;
            }
          }

          // std::cout << "Content of tx_word: " << tx_word << std::endl; // Prints in hexadecimal format
        size_t memcmp_ret = memcmp(reinterpret_cast<uint8_t*>(temp_data), reinterpret_cast<uint8_t*>(ref_byte), num_bytes_per_cb);
        // std::cout << "memcmp_ret is: " << memcmp_ret << std::endl;
        if (memcmp_ret != 0) {
            // Data matches, do something
            block_error++;
        }
          i++;
        }
        phy_stats_->UpdateDecodedBits(temp_ue_id, symbol_offset, frame_slot,
                                      num_bytes_per_cb * 8);
        phy_stats_->IncrementDecodedBlocks(temp_ue_id, symbol_offset, frame_slot);
        // memcmp(rte_pktmbuf_mtod_offset(m, uint32_t *, 0),
        phy_stats_->UpdateBlockErrors(temp_ue_id, symbol_offset, frame_slot, block_error);
      }
    }

    size_t end = GetTime::WorkerRdtsc();
    size_t duration_3 = end - start_tsc2;
    size_t duration = end - start_tsc;

    rte_pktmbuf_free(m_head);
    rte_pktmbuf_free(m_head_out);

    duration_stat_->task_duration_[3] += duration_3;
    duration_stat_->task_duration_[0] += duration;
    // duration_stat_->task_duration_[0] += 0;

    duration_stat_->task_count_++;
    if (GetTime::CyclesToUs(duration, cfg_->FreqGhz()) > 500) {
      std::printf("Thread %d Decode takes %.2f\n", tid_,
                  GetTime::CyclesToUs(duration, cfg_->FreqGhz()));
    }
  }
  else {
    // std::cout<<"symbol_idx_ul is: " << symbol_idx_ul << ", ue_id is: " << ue_id << std::endl;
    size_t start_tsc_else = GetTime::WorkerRdtsc();

    int8_t *llr_buffer_ptr;
    struct rte_mbuf *m_head;
    struct rte_mbuf *m_head_out;

    llr_buffer_ptr = demod_buffers_[frame_slot][symbol_idx_ul][ue_id] +
                            (cfg_->ModOrderBits(Direction::kUplink) *
                              (ldpc_config.NumCbCodewLen() * cur_cb_id));     

    if (symbol_idx_ul == 14 && frame_id > 100 && frame_id % 100 == 0 && kPrintLLRData) {
        std::printf("Hex: LLR data, symbol_offset, symbol_idx is: %zu\n", symbol_idx_ul);
        for (size_t i = 0; i < ldpc_config.NumCbCodewLen(); i++) {
            // Cast to uint8_t to ensure correct printing of the byte in hexadecimal
            std::printf("%02X ", static_cast<uint8_t>(*(llr_buffer_ptr + i)));
        }
        std::printf("\n");
    }
 
    char *data;
    struct rte_bbdev_op_data *bufs = *inputs;

    m_head = rte_pktmbuf_alloc(in_mbuf_pool);

    bufs[0].data = m_head;
    bufs[0].offset = 0;
    bufs[0].length = 0;

    data = rte_pktmbuf_append(m_head, ldpc_config.NumCbCodewLen());

        // Copy data from demod_data to the mbuf
    rte_memcpy(data, llr_buffer_ptr + (0 * ldpc_config.NumCbCodewLen()),
              ldpc_config.NumCbCodewLen());
    bufs[0].length += ldpc_config.NumCbCodewLen();

    rte_bbdev_op_data *bufs_out = *hard_outputs;
    m_head_out = rte_pktmbuf_alloc(out_mbuf_pool);

    bufs_out[0].data = m_head_out;
    bufs_out[0].offset = 0;
    bufs_out[0].length = 0;

    // Prepare the mbuf to receive the output data
    // char *data_out = rte_pktmbuf_append(m_head_out, ldpc_config.NumCbCodewLen());
    // assert(data_out == RTE_PTR_ALIGN(data_out, min_alignment));

    // BUG: This line causes a irregular stop of the program when fft_size = 4096,
    //      ofdm_data_num = 3168 = demul_block_size = beam_block_size, SISO, any
    //      sampling rate.
    // rte_memcpy(data_out, ref_byte_new, ldpc_config.NumCbCodewLen());
    bufs_out[0].length += ldpc_config.NumCbCodewLen();

    ref_dec_op[enq_index]->ldpc_dec.input = *inputs[0];
    ref_dec_op[enq_index]->ldpc_dec.hard_output = *hard_outputs[0];

    rte_pktmbuf_free(m_head);
    rte_pktmbuf_free(m_head_out);

    if (kMubfLLRCheck){
      struct rte_mbuf *mbuf_1 = bufs[0].data;
      uint8_t *mbuf_data = rte_pktmbuf_mtod(mbuf_1, uint8_t *);
      size_t data_length = mbuf_1->data_len;
      
      if (kPrintMbufData) {
        size_t num_bytes = data_length;
        size_t num_32bit_values = num_bytes / 4;
        size_t remaining_bytes = num_bytes % 4;

        for (size_t k = 0; k < num_32bit_values; k++) {
        uint32_t value = 0;
        for (size_t j = 0; j < 4; j++) {
            value |= static_cast<uint32_t>(mbuf_data[k * 4 + j]) << (j * 8);
        }
        if (k > 0) {
            std::printf(", ");
        }
        std::printf("0x%08x", value);
        }

        // Handle remaining bytes if any
        if (remaining_bytes > 0) {
            if (num_32bit_values > 0) {
                std::printf(", ");
            }
            uint32_t value = 0;
            for (size_t j = 0; j < remaining_bytes; j++) {
                value |= static_cast<uint32_t>(mbuf_data[num_32bit_values * 4 + j]) << (j * 8);
            }
            std::printf("0x%08x", value);
        }

        std::printf("\n");
      }

      if (memcmp(mbuf_data, llr_buffer_ptr, data_length) == 0) {
      std::printf("Data in inputs[%u].data matches llr_buffer_ptr\n", 0 );
  } else {
      std::printf("Data mismatch in inputs[%zu].data\n", 0);

      // Print mismatched bytes
      for (size_t j = 0; j < data_length; ++j) {
          if (mbuf_data[j] != (uint8_t)llr_buffer_ptr[j]) {
              std::printf("Mismatch at byte %zu: mbuf_data=%02x, llr_buffer_ptr=%02x\n",
                          j, mbuf_data[j], (uint8_t)llr_buffer_ptr[j]);
          }
      }
    }
  }
    enq += rte_bbdev_enqueue_ldpc_dec_ops(0, 0, &ref_dec_op[enq], 1);

    size_t end_else = GetTime::WorkerRdtsc();
    size_t duration_else = end_else - start_tsc_else;

    duration_stat_->task_duration_[0] += duration_else;
    enq_index++;
  }
#endif
  return EventData(EventType::kDecode, tag);
}







