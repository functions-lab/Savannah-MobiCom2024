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
static constexpr bool kPrintLLRData = false;
static constexpr bool kPrintDecodedData = false;
static constexpr bool kPrintACC100Byte = true;

static constexpr size_t kVarNodesSize = 1024 * 1024 * sizeof(int16_t);

static unsigned int optimal_mempool_size(unsigned int val) {
  return rte_align32pow2(val + 1) - 1;
}

static int init_op_data_objs_from_table(
    struct rte_bbdev_op_data *bufs, int8_t *demod_data,
    struct rte_mempool *mbuf_pool, const uint16_t n, uint16_t min_alignment,
    size_t seg_length,  // Added seg_length as a parameter
    // rte_mbuf *m_head
    struct rte_mbuf *mbuf) {
  int ret;
  unsigned int i, j;

  for (i = 0; i < n; ++i) {
    char *data;
    // std::cout << rte_mempool_avail_count(mbuf_pool) << std::endl;
    // if (rte_mempool_avail_count(mbuf_pool) == 0) {
    // printf("No more mbufs available in the pool! - input\n");
    //     return -1;
    // }
    struct rte_mbuf *m_head = rte_pktmbuf_alloc(mbuf_pool);
    // if (m_head == nullptr) {
    //     std::cerr << "Error: Unable to create mbuf pool: " << rte_strerror(rte_errno) << std::endl;
    //     return -1;
    // }

    bufs[i].data = m_head;
    bufs[i].offset = 0;
    bufs[i].length = 0;

    data = rte_pktmbuf_append(m_head, seg_length);

    // Copy data from demod_data to the mbuf
    rte_memcpy(data, demod_data + (i * seg_length), seg_length);
    bufs[i].length += seg_length;

    // Continue the same as before
    mbuf = m_head;
    // rte_pktmbuf_free(m_head);
  }

  return 0;
}

static int init_op_output_objs_from_buffer(
    struct rte_bbdev_op_data *bufs, uint8_t *decoded_buffer_ptr,
    struct rte_mempool *mbuf_pool, const uint16_t n, uint16_t min_alignment,
    size_t seg_length, struct rte_mbuf *mbuf) {
  unsigned int i;

  for (i = 0; i < n; ++i) {
    // std::cout << "out: " << rte_mempool_avail_count(mbuf_pool) << std::endl;
    // if (rte_mempool_avail_count(mbuf_pool) == 0) {
    // 	printf("No more mbufs available in the pool! - output\n");
    // 	return -1;
    // }
    struct rte_mbuf *m_head = rte_pktmbuf_alloc(mbuf_pool);
    // if (m_head == nullptr) {
    // 	std::cerr << "Error: Unable to create mbuf pool: " << rte_strerror(rte_errno) << std::endl;
    // 	return -1;  // Exit the program with an error code
    // }

    bufs[i].data = m_head;
    bufs[i].offset = 0;
    bufs[i].length = 0;

    // Prepare the mbuf to receive the output data
    char *data = rte_pktmbuf_append(m_head, seg_length);
    assert(data == RTE_PTR_ALIGN(data, min_alignment));

    // Assuming you will copy data from decoded_buffer_ptr to data
    rte_memcpy(data, decoded_buffer_ptr + i * seg_length, seg_length);

    bufs[i].length += seg_length;
    mbuf = m_head;
    // rte_pktmbuf_free(m_head);
  }
  return 0;
}

void print_uint32(const uint32_t* array, size_t totalByteLength) {
  // Round up to the nearest multiple of 4
  size_t totalWordLength = (totalByteLength + 3) / 4;

  for (int i = 0; i < totalWordLength; i++) {
    // Extract and print the byte
    printf("%08X ", array[i]);
  }
}

DoDecode_ACC::DoDecode_ACC(
    Config *in_config, int in_tid,
    PtrCube<kFrameWnd, kMaxSymbols, kMaxUEs, int8_t> &demod_buffers,
    PtrCube<kFrameWnd, kMaxSymbols, kMaxUEs, int8_t> &decoded_buffers,
    PhyStats *in_phy_stats, Stats *in_stats_manager)
    : Doer(in_config, in_tid),
      demod_buffers_(demod_buffers),
      decoded_buffers_(decoded_buffers),
      phy_stats_(in_phy_stats),
      // num_ul_syms(cfg_->Frame().NumULSyms()), 
      // ref_dec_op(num_ul_syms), 
      // ops_deq(num_ul_syms),
      scrambler_(std::make_unique<AgoraScrambler::Scrambler>()) {
  duration_stat_ = in_stats_manager->GetDurationStat(DoerType::kDecode, in_tid);
  // duration_stat_enq_ = in_stats_manager->GetDurationStat(DoerType::kEnqueue, in_tid);
  resp_var_nodes_ = static_cast<int16_t *>(Agora_memory::PaddedAlignedAlloc(
      Agora_memory::Alignment_t::kAlign64, kVarNodesSize));
  // std::cout<<"decode constructor"<<std::endl;
  std::string core_list = std::to_string(36);  // this is hard set to core 36
  const size_t num_ul_syms = cfg_->Frame().NumULSyms(); 
  const size_t num_ue = cfg_->UeAntNum();
  //  + "," + std::to_string(35) + "," + std::to_string(36) + "," + std::to_string(37);
  const char *rte_argv[] = {"txrx",        "-l",           core_list.c_str(),
                            "--log-level", "lib.eal:info", nullptr};
  int rte_argc = static_cast<int>(sizeof(rte_argv) / sizeof(rte_argv[0])) - 1;

  // Initialize DPDK environment
  // std::cout<<"getting ready to init dpdk" << std::endl;
  int ret = rte_eal_init(rte_argc, const_cast<char **>(rte_argv));
  RtAssert(
      ret >= 0,
      "Failed to initialize DPDK.  Are you running with root permissions?");

  int nb_bbdevs = rte_bbdev_count();
  std::cout << "num bbdevs: " << nb_bbdevs << std::endl;

  if (nb_bbdevs == 0) rte_exit(EXIT_FAILURE, "No bbdevs detected!\n");
  dev_id = 0;
  int ret_acc;
  struct rte_bbdev_info info;
  rte_bbdev_info_get(dev_id, &info);
  const struct rte_bbdev_info *dev_info = &info;
  const struct rte_bbdev_op_cap *op_cap = dev_info->drv.capabilities;
  // for (unsigned int i = 0; op_cap->type != RTE_BBDEV_OP_NONE; ++i, ++op_cap) {
  //   std::cout<<"capabilities is: " << op_cap->type << std::endl;
  // }

  rte_bbdev_intr_enable(dev_id);
  rte_bbdev_info_get(dev_id, &info);

  bbdev_op_pool =
      rte_bbdev_op_pool_create("bbdev_op_pool_dec", RTE_BBDEV_OP_LDPC_DEC,
                               NB_MBUF, 128, rte_socket_id());
  ret = rte_bbdev_setup_queues(dev_id, 4, info.socket_id);

  if (ret < 0) {
    printf("rte_bbdev_setup_queues(%u, %u, %d) ret %i\n", dev_id, 4,
           rte_socket_id(), ret);
  }

  ret = rte_bbdev_intr_enable(dev_id);

  struct rte_bbdev_queue_conf qconf;
  qconf.socket = info.socket_id;
  qconf.queue_size = info.drv.queue_size_lim;
  qconf.op_type = RTE_BBDEV_OP_LDPC_DEC;
  qconf.priority = 0;

  std::cout << "device id is: " << static_cast<int>(dev_id) << std::endl;

  for (int q_id = 0; q_id < 4; q_id++) {
    /* Configure all queues belonging to this bbdev device */
    ret = rte_bbdev_queue_configure(dev_id, q_id, &qconf);
    if (ret < 0)
      rte_exit(EXIT_FAILURE,
               "ERROR(%d): BBDEV %u queue %u not configured properly\n", ret,
               dev_id, q_id);
  }

  ret = rte_bbdev_start(dev_id);
  int socket_id = GET_SOCKET(info.socket_id);

  // unsigned int ops_pool_size = optimal_mempool_size(RTE_MAX(
  // 	/* Ops used plus 1 reference op */
  // 	RTE_MAX((unsigned int)(4 * num_ops + 1),
  // 	/* Minimal cache size plus 1 reference op */
  // 	(unsigned int)(1.5 * rte_lcore_count() *
  // 			OPS_CACHE_SIZE + 1)),
  // 	OPS_POOL_SIZE_MIN));

  // std::cout<<"ops_pool size is: " << ops_pool_size << std::endl;

  // ops_mp = rte_bbdev_op_pool_create("RTE_BBDEV_OP_LDPC_DEC_poo", RTE_BBDEV_OP_LDPC_DEC,
  //   num_ops, ops_pool_size, socket_id);

  ops_mp = rte_bbdev_op_pool_create("RTE_BBDEV_OP_LDPC_DEC_poo",
                                    RTE_BBDEV_OP_LDPC_DEC, 2047, OPS_CACHE_SIZE,
                                    socket_id);
  if (ops_mp == nullptr) {
    std::cerr << "Error: Failed to create memory pool for bbdev operations."
              << std::endl;
  } else {
    std::cout << "Memory pool for bbdev operations created successfully."
              << std::endl;
  }

  mbuf_pool =
      rte_pktmbuf_pool_create("bbdev_mbuf_pool", NB_MBUF, 256, 0,
                              RTE_MBUF_DEFAULT_BUF_SIZE, rte_socket_id());
  if (mbuf_pool == NULL) rte_exit(EXIT_FAILURE, "Unable to create\n");

  // char pool_name[RTE_MEMPOOL_NAMESIZE];

  in_mbuf_pool = rte_pktmbuf_pool_create("in_pool_0", 16383, 0, 0, 22744, 0);
  out_mbuf_pool =
      rte_pktmbuf_pool_create("hard_out_pool_0", 16383, 0, 0, 22744, 0);

  if (in_mbuf_pool == nullptr or out_mbuf_pool == nullptr) {
    std::cerr << "Error: Unable to create mbuf pool: "
              << rte_strerror(rte_errno) << std::endl;
  }

  // int rte_alloc_ref = rte_bbdev_dec_op_alloc_bulk(ops_mp, ref_dec_op, burst_sz);
  int rte_alloc_ref = rte_bbdev_dec_op_alloc_bulk(ops_mp, ref_dec_op, num_ul_syms * num_ue);

  // std::cout<<"rte_alloc_ref is: " << rte_alloc_ref << std::endl;
  // std::cout<<"rte_alloc is: " << rte_alloc << std::endl;
  // std::cout<< "op alloc bulk is: " << (rte_bbdev_dec_op_alloc_bulk(bbdev_op_pool, ref_dec_op, burst_sz)) << std::endl;

  ret = rte_pktmbuf_alloc_bulk(mbuf_pool, input_pkts_burst, MAX_PKT_BURST);
  ret = rte_pktmbuf_alloc_bulk(mbuf_pool, output_pkts_burst, MAX_PKT_BURST);
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

  ldpc_llr_decimals = capabilities->cap.ldpc_dec.llr_decimals;
  ldpc_llr_size = capabilities->cap.ldpc_dec.llr_size;
  ldpc_cap_flags = capabilities->cap.ldpc_dec.capability_flags;

  min_alignment = info.drv.min_alignment;

  const LDPCconfig &ldpc_config = cfg_->LdpcConfig(Direction::kUplink);

  int iter_num = num_ul_syms * num_ue;
  // std::cout<<"iter_num is: " << iter_num << std::endl;
  q_m = cfg_->ModOrderBits(Direction::kUplink);
  e = ldpc_config.NumCbCodewLen();
  // std::cout<<"NumCbCodeLen is: " << ldpc_config.NumCbCodewLen() <<std::endl;

  for (int i = 0; i < iter_num; i++) {
  // ref_dec_op[i]->ldpc_dec.op_flags += RTE_BBDEV_LDPC_ITERATION_STOP_ENABLE;
    ref_dec_op[i]->ldpc_dec.basegraph = (uint8_t)ldpc_config.BaseGraph();
    ref_dec_op[i]->ldpc_dec.z_c = (uint16_t)ldpc_config.ExpansionFactor();
    ref_dec_op[i]->ldpc_dec.n_filler = (uint16_t)0;
    ref_dec_op[i]->ldpc_dec.rv_index = (uint8_t)0;
    ref_dec_op[i]->ldpc_dec.n_cb = (uint16_t)ldpc_config.NumCbCodewLen();
    ref_dec_op[i]->ldpc_dec.q_m = (uint8_t)q_m;
    ref_dec_op[i]->ldpc_dec.code_block_mode = (uint8_t)1;
    ref_dec_op[i]->ldpc_dec.cb_params.e = (uint32_t)e;
    if (!check_bit(ref_dec_op[i]->ldpc_dec.op_flags,
                  RTE_BBDEV_LDPC_ITERATION_STOP_ENABLE)) {
      ref_dec_op[i]->ldpc_dec.op_flags += RTE_BBDEV_LDPC_ITERATION_STOP_ENABLE;
    }
    // if (check_bit(ref_dec_op[i]->ldpc_dec.op_flags, RTE_BBDEV_LDPC_ITERATION_STOP_ENABLE)){
    //   ref_dec_op[i]->ldpc_dec.op_flags -= RTE_BBDEV_LDPC_ITERATION_STOP_ENABLE;
    // }
    ref_dec_op[i]->ldpc_dec.iter_max = (uint8_t)ldpc_config.MaxDecoderIter();
    // ref_dec_op[i]->ldpc_dec.iter_count = (uint8_t) ref_dec_op[i]->ldpc_dec.iter_max;
    // std::cout<<"iter count is: " << unsigned(ref_dec_op[i]->ldpc_dec.iter_count) << std::endl;
    // ref_dec_op[i]->ldpc_dec.op_flags = RTE_BBDEV_LDPC_ITERATION_STOP_ENABLE;
    ref_dec_op[i]->opaque_data = (void *)(uintptr_t)i;
  }
  // in_m_head = rte_pktmbuf_alloc(in_mbuf_pool);
  // out_m_head = rte_pktmbuf_alloc(out_mbuf_pool);
  std::cout << "rte_pktmbuf_alloc successful" << std::endl;
}

DoDecode_ACC::~DoDecode_ACC() {
  std::free(resp_var_nodes_);
  // cleanup_bbdev_device("8086:0d5c"); // TODO: hardcoded
}

int DoDecode_ACC::allocate_buffers_on_socket(struct rte_bbdev_op_data **buffers,
                                             const int len, const int socket) {
  int i;
  // std::cout<<"start to allocate to socket"<<std::endl;
  *buffers = static_cast<struct rte_bbdev_op_data *>(
      rte_zmalloc_socket(NULL, len, 0, socket));
  // std::cout<<"no error"<<std::endl;
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
  // std::cout<<"num_ue is: " << num_ue << std::endl;
  // std::cout<<"ue_id is: " << ue_id << std::endl;
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
    uint8_t *decoded_buffer_ptr;
    struct rte_mbuf *m_head;
    struct rte_mbuf *m_head_out;
    // std::cout<<"inside the loop " << std::endl;
    size_t index = 0;

    for (size_t temp_ue_id = 0; temp_ue_id < num_ue; temp_ue_id++){
      for (size_t temp_idx = 0; temp_idx < num_ul_syms; temp_idx++){
        llr_buffer_ptr = demod_buffers_[frame_slot][temp_idx][temp_ue_id] +
                                (cfg_->ModOrderBits(Direction::kUplink) *
                                  (ldpc_config.NumCbCodewLen() * cur_cb_id));

        decoded_buffer_ptr =
            (uint8_t *)decoded_buffers_[frame_slot][temp_idx][temp_ue_id] +
            (cur_cb_id * Roundup<64>(num_bytes_per_cb));

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
        char *data_out = rte_pktmbuf_append(m_head, ldpc_config.NumCbCodewLen());
        assert(data_out == RTE_PTR_ALIGN(data_out, min_alignment));

        // BUG: This line causes a irregular stop of the program when fft_size = 4096,
        //      ofdm_data_num = 3168 = demul_block_size = beam_block_size, SISO, any
        //      sampling rate.
        // Assuming you will copy data from decoded_buffer_ptr to data
        rte_memcpy(data_out, decoded_buffer_ptr + 0 * ldpc_config.NumCbCodewLen(),
                  ldpc_config.NumCbCodewLen());

        bufs_out[0].length += ldpc_config.NumCbCodewLen();

        if (kPrintDecodedData) {
          std::printf("Decoded data after init hard_outputs\n");
          for (size_t i = 0; i < (ldpc_config.NumCbLen() >> 3); i++) {
            std::printf("%u ", *(decoded_buffer_ptr + i));
          }
          std::printf("\n");
        }
        ref_dec_op[index]->ldpc_dec.input = *inputs[0];
        ref_dec_op[index]->ldpc_dec.hard_output = *hard_outputs[0];


        if (kPrintDecodedData) {
          std::printf("Decoded data after putting to LDPC\n");
          for (size_t i = 0; i < (ldpc_config.NumCbLen() >> 3); i++) {
            std::printf("%u ", *(decoded_buffer_ptr + i));
          }
          std::printf("\n");
        }
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
    // commented for now
    // struct rte_mbuf *decoded_mbuf = (struct rte_mbuf *)(hard_outputs[0]->data);
    // Assuming data is contiguous in the mbuf, and not scattered across multiple segments
    // char *ldpc_decoded_data = rte_pktmbuf_mtod(decoded_mbuf, char *);
    // // rte_memcpy(decoded_buffer_ptr, ldpc_decoded_data, hard_outputs[0]->length);

    if (cfg_->ScrambleEnabled()) {
      scrambler_->Descramble(decoded_buffer_ptr, num_bytes_per_cb);
    }

    size_t start_tsc2 = GetTime::WorkerRdtsc();
    duration_stat_->task_duration_[2] += start_tsc2 - start_tsc1;

    size_t BLER(0);

    uint8_t rx_byte[MAX_RX_BYTE_SIZE];
    uint8_t tx_byte;
    uint32_t tx_word;
    size_t block_error(0);

    if ((kEnableMac == false) && (kPrintPhyStats == true) &&
        (symbol_idx_ul >= cfg_->Frame().ClientUlPilotSymbols())) {
      // get mbuf from the ops_deq

      struct rte_bbdev_op_ldpc_dec *ops_td;
      unsigned int i;
      struct rte_bbdev_op_data *hard_output;
      struct rte_mbuf *temp_m;
      size_t offset = 0;

      for (size_t temp_ue_id = 0; temp_ue_id < num_ue; temp_ue_id++){
        size_t BLER(0);
        for (size_t temp_idx = 0; temp_idx < num_ul_syms; temp_idx++){
          tx_byte = static_cast<uint8_t>(
            cfg_->GetInfoBits(cfg_->UlBits(), Direction::kUplink, temp_idx,
                              temp_ue_id, cur_cb_id)[i]);
          phy_stats_->UpdateBitErrors(temp_ue_id, symbol_offset, frame_slot,
                                      tx_byte, tx_byte);
          tx_word = static_cast<uint8_t>(tx_byte);

          ops_td = &ops_deq[i]->ldpc_dec;
          hard_output = &ops_td->hard_output;
          temp_m = hard_output->data;
          uint32_t* temp_data = rte_pktmbuf_mtod(temp_m, uint32_t*);
          RtAssert(num_bytes_per_cb == rte_pktmbuf_data_len(temp_m),
                   "num_bytes_per_cb does not match the length of the mbuf.");

          // If temp_m contains multiple bytes, use memcmp to compare
          if (kPrintACC100Byte){
            if (frame_id > 1000 && frame_id % 1000 == 0) {
              std::cout << "CB size = " << num_bytes_per_cb << " bytes\n";
              std::cout << "Content of the CB (in uint32):\n";
              print_uint32(temp_data, num_bytes_per_cb);
              std::cout << std::endl << std::endl;
            }
          }

          // std::cout << "Content of tx_word: " << tx_word << std::endl; // Prints in hexadecimal format
          if (memcmp(temp_data, &tx_word, num_bytes_per_cb) != 0) {
            // Data matches, do something
            block_error++;
          }
        // if (memcmp(rte_pktmbuf_mtod_offset(temp_m, uint32_t *, 0),
        //     hard_data_orig->segments[0].addr,
        //     hard_data_orig->segments[0].length))
        //   BLER++;

        
        // struct rte_mbuf *segment = temp_m;
        // while (segment) {
        //   // Get the data pointer and data length of the segment
        //   uint8_t *segment_data = rte_pktmbuf_mtod(segment, uint8_t *);
        //   size_t segment_length = rte_pktmbuf_data_len(segment);
        //   // Copy the segment data into rx_byte at the current offset
        //   memcpy(&rx_byte[offset], segment_data, segment_length);
        //   offset += segment_length;
        //   segment = segment->next;
        // }
          i++;
        }
        phy_stats_->UpdateDecodedBits(temp_ue_id, symbol_offset, frame_slot,
                                      num_bytes_per_cb * 8);
        phy_stats_->IncrementDecodedBlocks(temp_ue_id, symbol_offset, frame_slot);
        // memcmp(rte_pktmbuf_mtod_offset(m, uint32_t *, 0),
        phy_stats_->UpdateBlockErrors(temp_ue_id, symbol_offset, frame_slot, BLER);
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
    // std::cout<<"[In If]callling doDecode launch, Frame id is " << frame_id << " symbol id is: " << symbol_id <<std::endl;
    size_t start_tsc = GetTime::WorkerRdtsc();

    int8_t *llr_buffer_ptr;
    uint8_t *decoded_buffer_ptr;
    struct rte_mbuf *m_head;
    struct rte_mbuf *m_head_out;
    // size_t index = ue_id * (num_ul_syms * num_ue) + symbol_idx_ul * num_ue + ue_id;
    // size_t index = ue_id * num_ul_syms + symbol_idx_ul;
    // std::cout<<"enq_index is: " << enq_index << std::endl;

    llr_buffer_ptr = demod_buffers_[frame_slot][symbol_idx_ul][ue_id] +
                            (cfg_->ModOrderBits(Direction::kUplink) *
                              (ldpc_config.NumCbCodewLen() * cur_cb_id));

    decoded_buffer_ptr =
        (uint8_t *)decoded_buffers_[frame_slot][symbol_idx_ul][ue_id] +
        (cur_cb_id * Roundup<64>(num_bytes_per_cb));

    // struct rte_mbuf *inmbuf;
    // struct rte_mbuf *outmbuf;

    // int ret_init_op = init_op_data_objs_from_table(*inputs, llr_buffer_ptr, in_mbuf_pool, 1, min_alignment, ldpc_config.NumCbCodewLen(), in_mbuf);
    // std::cout<<"ret_init_op is " << ret_init_op << std::endl;
    char *data;
    struct rte_bbdev_op_data *bufs = *inputs;
    // std::cout << rte_mempool_avail_count(mbuf_pool) << std::endl;
    // if (rte_mempool_avail_count(mbuf_pool) == 0) {
    //     printf("No more mbufs available in the pool! - input\n");
    //     return -1;
    // }
    m_head = rte_pktmbuf_alloc(in_mbuf_pool);
    // if (m_head == nullptr) {
    //     std::cerr << "Error: Unable to create mbuf pool: " << rte_strerror(rte_errno) << std::endl;
    //     return -1;
    // }

    bufs[0].data = m_head;
    bufs[0].offset = 0;
    bufs[0].length = 0;

    data = rte_pktmbuf_append(m_head, ldpc_config.NumCbCodewLen());

    // Copy data from demod_data to the mbuf
    rte_memcpy(data, llr_buffer_ptr + (0 * ldpc_config.NumCbCodewLen()),
              ldpc_config.NumCbCodewLen());
    bufs[0].length += ldpc_config.NumCbCodewLen();

    // int ret_init_op = init_op_output_objs_from_buffer(*hard_outputs, decoded_buffer_ptr, out_mbuf_pool, 1, min_alignment, ldpc_config.NumCbCodewLen(), out_mbuf);
    // std::cout<<"ret_init_op is " << ret_init_op << std::endl;

    // std::cout << "out: " << rte_mempool_avail_count(mbuf_pool) << std::endl;
    // if (rte_mempool_avail_count(mbuf_pool) == 0) {
    // 	printf("No more mbufs available in the pool! - output\n");
    // 	return -1;
    // }

    rte_bbdev_op_data *bufs_out = *hard_outputs;
    m_head_out = rte_pktmbuf_alloc(out_mbuf_pool);
    // if (m_head == nullptr) {
    // 	std::cerr << "Error: Unable to create mbuf pool: " << rte_strerror(rte_errno) << std::endl;
    // 	return -1;  // Exit the program with an error code
    // }

    bufs_out[0].data = m_head_out;
    bufs_out[0].offset = 0;
    bufs_out[0].length = 0;

    // Prepare the mbuf to receive the output data
    char *data_out = rte_pktmbuf_append(m_head, ldpc_config.NumCbCodewLen());
    assert(data_out == RTE_PTR_ALIGN(data_out, min_alignment));

    // BUG: This line causes a irregular stop of the program when fft_size = 4096,
    //      ofdm_data_num = 3168 = demul_block_size = beam_block_size, SISO, any
    //      sampling rate.
    // Assuming you will copy data from decoded_buffer_ptr to data
    rte_memcpy(data_out, decoded_buffer_ptr + 0 * ldpc_config.NumCbCodewLen(),
              ldpc_config.NumCbCodewLen());

    bufs_out[0].length += ldpc_config.NumCbCodewLen();

    ref_dec_op[enq_index]->ldpc_dec.input = *inputs[0];
    ref_dec_op[enq_index]->ldpc_dec.hard_output = *hard_outputs[0];

    // rte_pktmbuf_free(out_mbuf);
    rte_pktmbuf_free(m_head);
    rte_pktmbuf_free(m_head_out);

    size_t start_tsc1 = GetTime::WorkerRdtsc();
    duration_stat_->task_duration_[1] += start_tsc1 - start_tsc;
    
    // bool first_time = true;
    uint64_t start_time = 0, last_time = 0;
    
    enq += rte_bbdev_enqueue_ldpc_dec_ops(0, 0, &ref_dec_op[enq], 1);

    int retry_count = 0;

    while (deq < enq && retry_count < MAX_DEQUEUE_TRIAL) {
      // rte_delay_ms(10);  // Wait for 10 milliseconds
      deq += rte_bbdev_dequeue_ldpc_dec_ops(0, 0, &ops_deq[deq], enq - deq);
      retry_count++;
    }
    AGORA_LOG_INFO("ACC100: enq = %d, deq = %d\n", enq, deq);

    enq = 0;
    deq = 0;
    enq_index = 0;

    // commented for now
    // struct rte_mbuf *decoded_mbuf = (struct rte_mbuf *)(hard_outputs[0]->data);
    // Assuming data is contiguous in the mbuf, and not scattered across multiple segments
    // char *ldpc_decoded_data = rte_pktmbuf_mtod(decoded_mbuf, char *);
    // // rte_memcpy(decoded_buffer_ptr, ldpc_decoded_data, hard_outputs[0]->length);

    if (cfg_->ScrambleEnabled()) {
      scrambler_->Descramble(decoded_buffer_ptr, num_bytes_per_cb);
      // std::cout<<"scramble enabled";
    }

    size_t start_tsc2 = GetTime::WorkerRdtsc();
    duration_stat_->task_duration_[2] += start_tsc2 - start_tsc1;

    size_t BLER(0);

    uint8_t rx_byte[MAX_RX_BYTE_SIZE];
    uint8_t tx_byte;
    uint32_t tx_word;
    size_t block_error(0);

    if ((kEnableMac == false) && (kPrintPhyStats == true) &&
        (symbol_idx_ul >= cfg_->Frame().ClientUlPilotSymbols())) {
      // get mbuf from the ops_deq

      struct rte_bbdev_op_ldpc_dec *ops_td;
      unsigned int i = 0;
      struct rte_bbdev_op_data *hard_output;
      struct rte_mbuf *temp_m;
      size_t offset = 0;

      for (size_t temp_ue_id = 0; temp_ue_id < num_ue; temp_ue_id++){
        size_t BLER(0);
        for (size_t temp_idx = 0; temp_idx < num_ul_syms; temp_idx++){
          tx_byte = static_cast<uint8_t>(
            cfg_->GetInfoBits(cfg_->UlBits(), Direction::kUplink, temp_idx,
                              temp_ue_id, cur_cb_id)[i]);
          phy_stats_->UpdateBitErrors(temp_ue_id, symbol_offset, frame_slot,
                                      tx_byte, tx_byte);
          tx_word = static_cast<uint8_t>(tx_byte);

          ops_td = &ops_deq[i]->ldpc_dec;
          hard_output = &ops_td->hard_output;
          temp_m = hard_output->data;
          uint32_t* temp_data = rte_pktmbuf_mtod(temp_m, uint32_t*);
          RtAssert(num_bytes_per_cb == rte_pktmbuf_data_len(temp_m),
                   "num_bytes_per_cb does not match the length of the mbuf.");

          // If temp_m contains multiple bytes, use memcmp to compare
          if (kPrintACC100Byte){
            if (frame_id > 1000 && frame_id % 1000 == 0) {
              std::cout << "CB size = " << num_bytes_per_cb << " bytes\n";
              std::cout << "Content of the CB (in uint32):\n";
              print_uint32(temp_data, num_bytes_per_cb);
              std::cout << std::endl << std::endl;
            }
          }

          // std::cout << "Content of tx_word: " << tx_word << std::endl; // Prints in hexadecimal format
          if (memcmp(temp_data, &tx_word, num_bytes_per_cb) != 0) {
            // Data matches, do something
            block_error++;
          }
        // if (memcmp(rte_pktmbuf_mtod_offset(temp_m, uint32_t *, 0),
        //     hard_data_orig->segments[0].addr,
        //     hard_data_orig->segments[0].length))
        //   BLER++;

        
        // struct rte_mbuf *segment = temp_m;
        // while (segment) {
        //   // Get the data pointer and data length of the segment
        //   uint8_t *segment_data = rte_pktmbuf_mtod(segment, uint8_t *);
        //   size_t segment_length = rte_pktmbuf_data_len(segment);
        //   // Copy the segment data into rx_byte at the current offset
        //   memcpy(&rx_byte[offset], segment_data, segment_length);
        //   offset += segment_length;
        //   segment = segment->next;
        // }
          i++;
        }
        phy_stats_->UpdateDecodedBits(temp_ue_id, symbol_offset, frame_slot,
                                      num_bytes_per_cb * 8);
        phy_stats_->IncrementDecodedBlocks(temp_ue_id, symbol_offset, frame_slot);
        // memcmp(rte_pktmbuf_mtod_offset(m, uint32_t *, 0),
        phy_stats_->UpdateBlockErrors(temp_ue_id, symbol_offset, frame_slot, BLER);
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
    uint8_t *decoded_buffer_ptr;
    struct rte_mbuf *m_head;
    struct rte_mbuf *m_head_out;

    llr_buffer_ptr = demod_buffers_[frame_slot][symbol_idx_ul][ue_id] +
                            (cfg_->ModOrderBits(Direction::kUplink) *
                              (ldpc_config.NumCbCodewLen() * cur_cb_id));

    decoded_buffer_ptr =
        (uint8_t *)decoded_buffers_[frame_slot][symbol_idx_ul][ue_id] +
        (cur_cb_id * Roundup<64>(num_bytes_per_cb));

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
    char *data_out = rte_pktmbuf_append(m_head, ldpc_config.NumCbCodewLen());
    assert(data_out == RTE_PTR_ALIGN(data_out, min_alignment));

    // BUG: This line causes a irregular stop of the program when fft_size = 4096,
    //      ofdm_data_num = 3168 = demul_block_size = beam_block_size, SISO, any
    //      sampling rate.
    // Assuming you will copy data from decoded_buffer_ptr to data
    rte_memcpy(data_out, decoded_buffer_ptr + 0 * ldpc_config.NumCbCodewLen(),
              ldpc_config.NumCbCodewLen());
    bufs_out[0].length += ldpc_config.NumCbCodewLen();

        // std::cout<<"index is: " << index << std::endl;
    ref_dec_op[enq_index]->ldpc_dec.input = *inputs[0];
    ref_dec_op[enq_index]->ldpc_dec.hard_output = *hard_outputs[0];

    rte_pktmbuf_free(m_head);
    rte_pktmbuf_free(m_head_out);


    enq += rte_bbdev_enqueue_ldpc_dec_ops(0, 0, &ref_dec_op[enq], 1);


    size_t end_else = GetTime::WorkerRdtsc();
    size_t duration_else = end_else - start_tsc_else;

    duration_stat_->task_duration_[0] += duration_else;
    enq_index++;
  }
#endif
  return EventData(EventType::kDecode, tag);
}







