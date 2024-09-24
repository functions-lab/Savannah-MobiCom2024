/**
 * @file data_generator.h
 * @brief Implementation file for the Data generator class to generate binary
 * files as inputs to Agora, sender and correctness tests
 */
#ifndef DATA_GENERATOR_H_
#define DATA_GENERATOR_H_

#include <cstdint>
#include <string>
#include <vector>

#include "config.h"
#include "ldpc_config.h"
#include "memory_manage.h"
#include "message.h"

#if defined(USE_ACC100_ENCODE)
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
#endif
/**
 * @brief Building blocks for generating end-to-end or unit test workloads for
 * Agora
 */
class DataGenerator {
 public:
  // The profile of the input information bits
  enum class Profile {
    kRandom,  // The input information bytes are chosen at random

    // The input information bytes are {1, 2, 3, 1, 2, 3, ...} for UE 0,
    // {4, 5, 6, 4, 5, 6, ...} for UE 1, and so on
    kProfile123
  };

  explicit DataGenerator(Config* cfg, uint64_t seed = 0,
                         Profile profile = Profile::kRandom);

  void DoDataGeneration(const std::string& directory);

  /**
   * @brief                        Generate random Mac payload bit
   * sequence
   *
   * @param  information           The generated input bit sequence
   * @param  ue_id                 ID of the UE that this codeblock belongs to
   */
  void GenMacData(MacPacketPacked* mac, size_t ue_id);

  /**
   * @brief                        Generate one raw information bit sequence
   *
   * @param  information           The generated input bit sequence
   * @param  ue_id                 ID of the UE that this codeblock belongs to
   */
  void GenRawData(const LDPCconfig& lc, std::vector<int8_t>& information,
                  size_t ue_id);

  /**
   * @brief                        Generate the encoded bit sequence for one
   * code block for the active LDPC configuration from the input bit sequence
   *
   * @param  input_ptr             The input bit sequence to be encoded
   * @param  encoded_codeword      The generated encoded codeword bit sequence
   */
  static std::vector<int8_t> GenCodeblock(const LDPCconfig& lc,
                                          const int8_t* input_ptr,
                                          size_t input_size,
                                          bool scramble_enabled = false);

#if defined(USE_ACC100_ENCODE)
    /**
   * @brief                        Generate the encoded bit sequence for one
   * code block for the active LDPC configuration from the input bit sequence using ACC100
   *
   * @param  input_ptr             The input bit sequence to be encoded
   * @param  encoded_codeword      The generated encoded codeword bit sequence
   */
  std::vector<int8_t> GenCodeblock_ACC100(const LDPCconfig& lc,
                                          const int8_t* input_ptr,
                                          size_t input_size,
                                          bool scramble_enabled = false, size_t enq_index = 0);
#endif

  /**
   * @brief Return the output of modulating the encoded codeword
   * @param encoded_codeword The encoded LDPC codeword bit sequence
   * @return An array of complex floats with OfdmDataNum() elements
   */
  static std::vector<complex_float> GetModulation(
      const int8_t* encoded_codeword, Table<complex_float> mod_table,
      const size_t num_bits, const size_t num_subcarriers,
      const size_t mod_order_bits);

  static std::vector<complex_float> MapOFDMSymbol(
      Config* cfg, const std::vector<complex_float>& modulated_codeword,
      complex_float* pilot_seq, SymbolType symbol_type);

  /**
   * @param modulated_codeword The modulated codeword with OfdmDataNum()
   * elements
   * @brief An array with OfdmDataNum() elements with the OfdmDataNum()
   * modulated elements binned at the center
   */
  static std::vector<complex_float> BinForIfft(
      Config* cfg, const std::vector<complex_float>& modulated_codeword,
      bool is_fftshifted = false);

  /// Return the frequency-domain pilot symbol with OfdmCaNum complex floats
  std::vector<complex_float> GetCommonPilotFreqDomain() const;

  /// Return the user-spepcific frequency-domain pilot symbol with OfdmCaNum complex floats
  Table<complex_float> GetUeSpecificPilotFreqDomain() const;

  void GetNoisySymbol(const std::vector<complex_float>& modulated_symbol,
                      std::vector<complex_float>& noisy_symbol,
                      float noise_level);
  void GetNoisySymbol(const complex_float* modulated_symbol,
                      complex_float* noisy_symbol, size_t length,
                      float noise_level);

  static void GetNoisySymbol(complex_float* modulated_symbol, size_t length,
                             float noise_level, unsigned seed = 0);

  static void GetDecodedData(int8_t* demoded_data, uint8_t* decoded_codewords,
                             const LDPCconfig& ldpc_config,
                             size_t num_decoded_bytes,
                             bool scramble_enabled = false);

  static void GetDecodedDataBatch(Table<int8_t>& demoded_data,
                                  Table<uint8_t>& decoded_codewords,
                                  const LDPCconfig& ldpc_config,
                                  size_t num_codeblocks,
                                  size_t num_decoded_bytes,
                                  bool scramble_enabled = false);

 private:
  FastRand fast_rand_;  // A fast random number generator
  Config* cfg_;         // The global Agora config
  uint64_t seed_;
  const Profile profile_;  // The pattern of the input byte sequence
#if defined(USE_ACC100_ENCODE)
  uint8_t dev_id;
  int ldpc_llr_decimals;
  int ldpc_llr_size;
  uint32_t ldpc_cap_flags;
  uint16_t min_alignment;
  uint16_t num_ops = 2047;
  uint16_t burst_sz = 1;
  size_t q_m;
  size_t e;
  uint16_t enq = 0;
  uint16_t deq = 0;
  size_t enq_index = 0;

  struct rte_mempool* ops_mp;
  struct rte_mempool* in_mbuf_pool;
  struct rte_mempool* out_mbuf_pool;
  struct rte_mempool* bbdev_op_pool;

  struct rte_mbuf* in_mbuf;
  struct rte_mbuf* out_mbuf;

  struct rte_bbdev_enc_op* ref_enc_op[64];
  struct rte_bbdev_enc_op* ops_deq[64];

  struct rte_bbdev_op_data** inputs;
  struct rte_bbdev_op_data** hard_outputs;

  rte_mbuf* input_pkts_burst[54];
  rte_mbuf* output_pkts_burst[54];
  rte_mempool* mbuf_pool;

#endif
};

#endif  // DATA_GENERATOR_H_