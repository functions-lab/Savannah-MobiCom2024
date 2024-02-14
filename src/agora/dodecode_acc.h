/**
 * @file dodecode.h
 * @brief Declaration file for the DoDecode class.
 */

#ifdef USE_ACC100

#ifndef DODECODE_ACC_H_
#define DODECODE_ACC_H_

#include <gflags/gflags.h>
#include <immintrin.h>
#include <netinet/ether.h>
#include <rte_byteorder.h>
#include <rte_cycles.h>
#include <rte_debug.h>
#include <rte_distributor.h>
#include <rte_eal.h>
#include <rte_ethdev.h>
#include <rte_ether.h>
#include <rte_flow.h>
#include <rte_ip.h>
#include <rte_malloc.h>
#include <rte_pause.h>
#include <rte_prefetch.h>
#include <rte_udp.h>
#include <unistd.h>

#include <bitset>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <memory>
#include <random>

#include "config.h"
#include "doer.h"
#include "memory_manage.h"
#include "message.h"
#include "phy_stats.h"
#include "rte_eal.h"
#include "rte_lcore.h"
#include "rte_malloc.h"
#include "rte_mbuf.h"
#include "rte_mempool.h"
#include "scrambler.h"
#include "stats.h"
#include "logger.h"


#define NB_MBUF 8192
#define MBUF_CACHE_SIZE 256
#define RTE_MBUF_DEFAULT_DATAROOM 2048  // adjust based on needs
#define TEST_SUCCESS 0
#define TEST_FAILED -1
#define TEST_SKIPPED 1
#define MAX_QUEUES RTE_MAX_LCORE
#define OPS_CACHE_SIZE 256U
#define OPS_POOL_SIZE_MIN 511U /* 0.5K per queue */

#define MAX_PKT_BURST 32
#define MAX_BURST 512U
#define SYNC_START 1
#define MAX_DEQUEUE_TRIAL 1000000


// struct rte_bbdev_dec_op {
// 	/** Status of operation that was performed */
// 	int status;
// 	/** Mempool which op instance is in */
// 	struct rte_mempool *mempool;
// 	/** Opaque pointer for user data */
// 	void *opaque_data;
// 	union {
// 		/** Contains LDPC decoder specific parameters */
// 		struct rte_bbdev_op_ldpc_dec ldpc_dec;
// 	};
// };
class DoDecode_ACC : public Doer {
 public:
  DoDecode_ACC(
      Config* in_config, int in_tid,
      PtrCube<kFrameWnd, kMaxSymbols, kMaxUEs, int8_t>& demod_buffers,
      PtrCube<kFrameWnd, kMaxSymbols, kMaxUEs, int8_t>& decoded_buffers,
      PhyStats* in_phy_stats, Stats* in_stats_manager);
  ~DoDecode_ACC() override;

  EventData Launch(size_t tag) override;
  static int allocate_buffers_on_socket(struct rte_bbdev_op_data** buffers,
                                        const int len, const int socket);
  static inline bool check_bit(uint32_t bitmap, uint32_t bitmask) {
    return bitmap & bitmask;
  }

 private:
  int16_t* resp_var_nodes_;
  PtrCube<kFrameWnd, kMaxSymbols, kMaxUEs, int8_t>& demod_buffers_;
  PtrCube<kFrameWnd, kMaxSymbols, kMaxUEs, int8_t>& decoded_buffers_;
  PhyStats* phy_stats_;
  DurationStat* duration_stat_;
  // DurationStat* duration_stat_enq_;
  std::unique_ptr<AgoraScrambler::Scrambler> scrambler_;

  // struct rte_bbdev_dec_op;

  uint8_t dev_id;
  int ldpc_llr_decimals;
  int ldpc_llr_size;
  uint32_t ldpc_cap_flags;
  uint16_t min_alignment;
  uint16_t num_ops = 2047;
  uint16_t burst_sz = 1;
  size_t q_m;
  size_t e;
  // const size_t num_ul_syms = cfg_->Frame().NumULSyms();

  struct rte_mempool* ops_mp;
  struct rte_mempool* in_mbuf_pool;
  struct rte_mempool* out_mbuf_pool;
  struct rte_mempool* bbdev_op_pool;

  struct rte_mbuf* in_mbuf;
  struct rte_mbuf* out_mbuf;

  // size_t num_ul_syms;
  // std::vector<rte_bbdev_dec_op* > ref_dec_op;
  // std::vector<rte_bbdev_dec_op* > ops_deq;
  
  struct rte_bbdev_dec_op* ref_dec_op[64];
  struct rte_bbdev_dec_op* ops_deq[64];

  struct rte_bbdev_op_data** inputs;
  struct rte_bbdev_op_data** hard_outputs;

  rte_mbuf* input_pkts_burst[54];
  rte_mbuf* output_pkts_burst[54];
  rte_mempool* mbuf_pool;

//     static inline void
// rte_bbdev_dec_op_free_bulk(struct rte_bbdev_dec_op **ops, unsigned int num_ops)
// {
// 	if (num_ops > 0)
// 		rte_mempool_put_bulk(ops[0]->mempool, (void **)ops, num_ops);
// }
  // rte_mbuf *in_m_head;
  // rte_mbuf *out_m_head;
};

#endif  // DODECODE_ACC_H_

#endif  // USE_ACC100
