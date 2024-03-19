/**
 * @file test_ldpc_baseband.cc
 * @brief Test LDPC performance after encoding, modulation, demodulation,
 * and decoding when different levels of
 * Gaussian noise is added to CSI
 */
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
#include <fstream>
#include <iostream>
#include <random>

#include "armadillo"
#include "comms-lib.h"
#include "config.h"
#include "data_generator.h"
#include "datatype_conversion.h"
#include "gettime.h"
#include "memory_manage.h"
#include "modulation.h"
#include "phy_ldpc_decoder_5gnr.h"
#include "rte_bbdev.h"
#include "rte_bbdev_op.h"
#include "rte_bus_vdev.h"
#include "rte_eal.h"
#include "rte_lcore.h"
#include "rte_malloc.h"
#include "rte_mbuf.h"
#include "rte_mempool.h"
#include "utils_ldpc.h"

#define TEST_SUCCESS 0
#define TEST_FAILED -1
#define TEST_SKIPPED 1
#define MAX_QUEUES RTE_MAX_LCORE
#define OPS_CACHE_SIZE 256U
#define MAX_PKT_BURST 32
#define MAX_BURST 512U
#define SYNC_START 1

struct thread_params {
  uint8_t dev_id;
  uint16_t queue_id;
  uint32_t lcore_id;
  uint64_t start_time;
  double ops_per_sec;
  double mbps;
  uint8_t iter_count;
  double iter_average;
  double bler;
  uint16_t nb_dequeued;
  int16_t processing_status;
  uint16_t burst_sz;
  struct test_op_params *op_params;
  struct rte_bbdev_dec_op *dec_ops[MAX_BURST];
  struct rte_bbdev_enc_op *enc_ops[MAX_BURST];
};

static inline bool check_bit(uint32_t bitmap, uint32_t bitmask) {
  return bitmap & bitmask;
}

#define TEST_ASSERT_SUCCESS(val, msg, ...)                                 \
  do {                                                                     \
    typeof(val) _val = (val);                                              \
    if (!(_val == 0)) {                                                    \
      printf("TestCase %s() line %d failed (err %d): " msg "\n", __func__, \
             __LINE__, _val, ##__VA_ARGS__);                               \
      return TEST_FAILED;                                                  \
    }                                                                      \
  } while (0)

static inline void mbuf_reset(struct rte_mbuf *m) {
  m->pkt_len = 0;

  do {
    m->data_len = 0;
    m = m->next;
  } while (m != NULL);
}

static int allocate_buffers_on_socket(struct rte_bbdev_op_data **buffers,
                                      const int len, const int socket) {
  int i;
  std::cout << "start to allocate to socket" << std::endl;
  *buffers = static_cast<struct rte_bbdev_op_data *>(
      rte_zmalloc_socket(NULL, len, 0, socket));
  std::cout << "no error" << std::endl;
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

static struct active_device {
  const char *driver_name;
  uint8_t dev_id;
  uint16_t supported_ops;
  uint16_t queue_ids[MAX_QUEUES];
  uint16_t nb_queues;
  struct rte_mempool *ops_mempool;
  struct rte_mempool *in_mbuf_pool;
  struct rte_mempool *hard_out_mbuf_pool;
  struct rte_mempool *soft_out_mbuf_pool;
  struct rte_mempool *harq_in_mbuf_pool;
  struct rte_mempool *harq_out_mbuf_pool;
} active_devs[RTE_BBDEV_MAX_DEVS];

enum BaseGraph { BG1 = 1, BG2 = 2 };

static uint8_t nb_active_devs;
// static bool intr_enabled;

// Define constants for the mbuf pool
#define NB_MBUF 8192
#define MBUF_CACHE_SIZE 256
#define RTE_MBUF_DEFAULT_DATAROOM \
  2048  // This should be adjusted based on your needs

// Create the mbuf pool
// if (mbuf_pool == NULL) {
//     rte_exit(EXIT_FAILURE, "Cannot create mbuf pool: %s\n", rte_strerror(rte_errno));
// }

static constexpr bool kVerbose = false;
static constexpr bool kPrintUplinkInformationBytes = false;
static constexpr float kNoiseLevels[15] = {
    1.7783, 1.3335, 1.0000, 0.7499, 0.5623, 0.4217, 0.3162, 0.2371,
    0.1778, 0.1334, 0.1000, 0.0750, 0.0562, 0.0422, 0.0316};
static constexpr float kSnrLevels[15] = {-5, -2.5, 0,  2.5,  5,  7.5,  10, 12.5,
                                         15, 17.5, 20, 22.5, 25, 27.5, 30};
DEFINE_string(profile, "random",
              "The profile of the input user bytes (e.g., 'random', '123')");
DEFINE_string(
    conf_file,
    TOSTRING(PROJECT_DIRECTORY) "/files/config/ci/tddconfig-sim-test.json",
    "Agora config filename");

float RandFloat(float min, float max) {
  return ((float(rand()) / float(RAND_MAX)) * (max - min)) + min;
}

float RandFloatFromShort(float min, float max) {
  float rand_val =
      ((static_cast<float>(rand()) / static_cast<float>(RAND_MAX)) *
       (max - min)) +
      min;
  auto rand_val_ushort = static_cast<short>(rand_val * kShrtFltConvFactor);
  rand_val = (float)rand_val_ushort / kShrtFltConvFactor;
  return rand_val;
}

static inline void set_avail_op(struct active_device *ad,
                                enum rte_bbdev_op_type op_type) {
  ad->supported_ops |= (1 << op_type);
}

size_t get_varNodes_length(int16_t z, int16_t nRows, int16_t numFillerBits,
                           uint16_t basegraph) {
  size_t length = 0;
  if (basegraph == BG1) {
    length = z * 22 + z * nRows - z * 2 - numFillerBits;
  } else if (basegraph == BG2) {
    length = z * 10 + z * nRows - z * 2 - numFillerBits;
  }
  return length;
}

size_t get_compactedMessageBytes_length(int16_t z, uint16_t basegraph,
                                        int16_t numFillerBits) {
  int BG_value = (basegraph == BG1) ? 22 : 10;
  size_t length = z * BG_value - numFillerBits;
  return (length + 7) / 8;  // This performs the ceiling division by 8
}

static int init_op_data_objs_from_table(struct rte_bbdev_op_data *bufs,
                                        Table<int8_t> &demod_data_all_symbols,
                                        struct rte_mempool *mbuf_pool,
                                        const uint16_t n,
                                        uint16_t min_alignment) {
  int ret;
  unsigned int i, j;

  // Assuming dim2_ represents the segment length
  // std::cout<<"start to get seg length"<<std::endl;
  size_t seg_length = demod_data_all_symbols.Dim2();
  // std::cout<<"finished getting length"<<std::endl;

  for (i = 0; i < n; ++i) {
    char *data;
    std::cout << rte_mempool_avail_count(mbuf_pool) << std::endl;
    if (rte_mempool_avail_count(mbuf_pool) == 0) {
      printf("No more mbufs available in the pool!\n");
      return -1;
    }
    struct rte_mbuf *m_head = rte_pktmbuf_alloc(mbuf_pool);
    if (m_head == nullptr) {
      std::cerr << "Error: Unable to create mbuf pool: "
                << rte_strerror(rte_errno) << std::endl;
      return -1;  // Exit the program with an error code
    } else {
    }

    // TEST_ASSERT_NOT_NULL(m_head,
    // 		"Not enough mbufs in %d data type mbuf pool (needed %u, available %u)",
    // 		op_type, n, mbuf_pool->size);

    bufs[i].data = m_head;
    bufs[i].offset = 0;
    bufs[i].length = 0;

    // if ((op_type == DATA_INPUT) || (op_type == DATA_HARQ_INPUT)) {
    data = rte_pktmbuf_append(m_head, seg_length);

    // TEST_ASSERT_NOT_NULL(data,
    //   "Couldn't append %u bytes to mbuf from %d data type mbuf pool",
    //   seg_length, op_type);

    // std::cout << "test data RTE PTR Aligh: " << (data == RTE_PTR_ALIGN(data, min_alignment)) << std::endl;

    // Copy data from demod_data_all_symbols to the mbuf
    rte_memcpy(data, demod_data_all_symbols[i], seg_length);
    bufs[i].length += seg_length;

    // If you have more segments to copy, chain additional mbufs
    // for (j = 1; j < /*Number of additional segments*/; ++j) {
    // 	struct rte_mbuf *m_tail = rte_pktmbuf_alloc(mbuf_pool);
    // 	// TEST_ASSERT_NOT_NULL(m_tail,
    // 	// 		"Not enough mbufs in %d data type mbuf pool (needed %u, available %u)",
    // 	// 		op_type, n, mbuf_pool->size);

    // 	data = rte_pktmbuf_append(m_tail, seg_length);
    // 	// TEST_ASSERT_NOT_NULL(data,
    // 	// 		"Couldn't append %u bytes to mbuf from %d data type mbuf pool",
    // 	// 		seg_length, op_type);

    // 	rte_memcpy(data, /* Pointer to the next segment in demod_data_all_symbols */, seg_length);
    // 	bufs[i].length += seg_length;

    // 	ret = rte_pktmbuf_chain(m_head, m_tail);
    // 	// TEST_ASSERT_SUCCESS(ret, "Couldn't chain mbufs from %d data type mbuf pool", op_type);
    // }

    // }
  }

  return 0;
}

static int init_op_output_objs_from_table(struct rte_bbdev_op_data *bufs,
                                          Table<uint8_t> &decoded_codewords,
                                          struct rte_mempool *mbuf_pool,
                                          const uint16_t n,
                                          uint16_t min_alignment) {
  unsigned int i, j;

  // Assuming dim2_ represents the segment length
  size_t seg_length = decoded_codewords.Dim2();

  for (i = 0; i < n; ++i) {
    if (rte_mempool_avail_count(mbuf_pool) == 0) {
      printf("No more mbufs available in the pool!\n");
      return -1;
    }
    struct rte_mbuf *m_head = rte_pktmbuf_alloc(mbuf_pool);
    if (m_head == nullptr) {
      std::cerr << "Error: Unable to create mbuf pool: "
                << rte_strerror(rte_errno) << std::endl;
      return -1;  // Exit the program with an error code
    }

    bufs[i].data = m_head;
    bufs[i].offset = 0;
    bufs[i].length = 0;

    // Prepare the mbuf to receive the output data
    char *data = rte_pktmbuf_append(m_head, seg_length);
    assert(data == RTE_PTR_ALIGN(data, min_alignment));

    bufs[i].length += seg_length;

    // If you have more segments to handle, chain additional mbufs
    // for (j = 1; j < /*Number of additional segments*/; ++j) {
    // 	struct rte_mbuf *m_tail = rte_pktmbuf_alloc(mbuf_pool);
    // 	data = rte_pktmbuf_append(m_tail, seg_length);
    // 	bufs[i].length += seg_length;
    // 	rte_pktmbuf_chain(m_head, m_tail);
    // }
  }

  return 0;
}

static void ldpc_input_llr_scaling(struct rte_bbdev_op_data *input_ops,
                                   const uint16_t n, const int8_t llr_size,
                                   const int8_t llr_decimals) {
  // printf("ldpc_input_llr_scaling being called\n");

  if (input_ops == NULL) return;

  uint16_t i, byte_idx;
  // printf("in ldpc_input_llr_scaling input_ops is not null !!!!!!!!!\n");
  int16_t llr_max, llr_min, llr_tmp;
  llr_max = (1 << (llr_size - 1)) - 1;
  llr_min = -llr_max;
  for (i = 0; i < n; ++i) {
    struct rte_mbuf *m = input_ops[i].data;
    while (m != NULL) {
      int8_t *llr = rte_pktmbuf_mtod_offset(m, int8_t *, input_ops[i].offset);
      for (byte_idx = 0; byte_idx < rte_pktmbuf_data_len(m); ++byte_idx) {
        llr_tmp = llr[byte_idx];
        if (llr_decimals == 4)
          llr_tmp *= 8;
        else if (llr_decimals == 2)
          llr_tmp *= 2;
        else if (llr_decimals == 0)
          llr_tmp /= 2;
        llr_tmp = RTE_MIN(llr_max, RTE_MAX(llr_min, llr_tmp));
        llr[byte_idx] = (int8_t)llr_tmp;
      }

      m = m->next;
    }
  }
}

void configure_bbdev_device(uint16_t dev_id) {
  struct rte_bbdev_info dev_info;
  std::cout << "in configure bbdev" << std::endl;
  uint16_t temp_dev_id = rte_bbdev_find_next(-1);
  uint16_t num_dev = rte_bbdev_count();
  std::cout << "num_dev: " << num_dev << std::endl;
  std::cout << "is valid?" << rte_bbdev_is_valid(temp_dev_id) << std::endl;
  // Get the bbdev device information
  int ret = rte_bbdev_info_get(dev_id, &dev_info);
  if (ret != 0) {
    // Handle error
    return;
  }

  // Get the maximum number of queues supported by the device
  uint16_t max_nb_queues = dev_info.num_queues;
  std::cout << "num queues is: " << unsigned(max_nb_queues) << std::endl;

  // Create the bbdev device
  // rte_vdev_init("baseband_ldpc_sw", NULL);

  // Setup queues for the bbdev device
  std::cout << "no issue here" << std::endl;
  rte_bbdev_setup_queues(dev_id, max_nb_queues, rte_socket_id());

  // Enable interrupts for the bbdev device if supported
  rte_bbdev_intr_enable(dev_id);
}

static int add_bbdev_dev(uint8_t dev_id, struct rte_bbdev_info *info) {
  std::cout << "current setting device: " << (unsigned)dev_id << std::endl;
  int ret;
  unsigned int queue_id;
  struct rte_bbdev_queue_conf qconf;
  struct active_device *ad = &active_devs[nb_active_devs];
  unsigned int nb_queues;
  enum rte_bbdev_op_type op_type = RTE_BBDEV_OP_LDPC_DEC;

  /* Configure fpga lte fec with PF & VF values
 * if '-i' flag is set and using fpga device
 */

  printf("Configure FPGA 5GNR FEC Driver %s with default values\n",
         info->drv.driver_name);
  // realized that the pf-bbdev should already set it to default, might be needed for future if implemented to Agora.

  /* Let's refresh this now this is configured */
  rte_bbdev_info_get(dev_id, info);
  std::cout << "max_num_queues is: " << info->drv.max_num_queues << std::endl;
  std::cout << "rte lcores is : " << rte_lcore_count() << std::endl;
  nb_queues = RTE_MIN(rte_lcore_count(), info->drv.max_num_queues);
  nb_queues = RTE_MIN(nb_queues, (unsigned int)MAX_QUEUES);

  /* setup device */
  std::cout << "num queues is: " << unsigned(nb_queues) << std::endl;

  ret = rte_bbdev_setup_queues(dev_id, nb_queues, info->socket_id);
  std::cout << "!!!!!!!!!!!! ret of setup queue is: " << ret << std::endl;
  if (ret < 0) {
    printf("rte_bbdev_setup_queues(%u, %u, %d) ret %i\n", dev_id, nb_queues,
           info->socket_id, ret);
    return TEST_FAILED;
  }

  ret = rte_bbdev_intr_enable(dev_id);
  std::cout << "ret for intr enable is: " << ret << std::endl;
  /* configure interrupts if needed */
  // if (intr_enabled) {
  // 	ret = rte_bbdev_intr_enable(dev_id);
  // 	if (ret < 0) {
  // 		printf("rte_bbdev_intr_enable(%u) ret %i\n", dev_id,
  // 				ret);
  // 		return TEST_FAILED;
  // 	}
  // }

  /* setup device queues */
  qconf.socket = info->socket_id;
  std::cout << "qcof.socket is: " << qconf.socket << std::endl;
  qconf.queue_size = info->drv.default_queue_conf.queue_size;
  std::cout << "queue size is" << qconf.queue_size << std::endl;
  qconf.priority = 0;
  qconf.deferred_start = 0;
  qconf.op_type = op_type;
  std::cout << "op type is: " << qconf.op_type << std::endl;

  for (queue_id = 0; queue_id < nb_queues; ++queue_id) {
    ret = rte_bbdev_queue_configure(dev_id, queue_id, &qconf);
    if (ret != 0) {
      printf("Allocated all queues (id=%u) at prio%u on dev%u\n", queue_id,
             qconf.priority, dev_id);
      qconf.priority++;
      ret = rte_bbdev_queue_configure(ad->dev_id, queue_id, &qconf);
    }
    if (ret != 0) {
      printf("All queues on dev %u allocated: %u\n", dev_id, queue_id);
      break;
    }
    ad->queue_ids[queue_id] = queue_id;
  }
  // TEST_ASSERT(queue_id != 0,
  // 		"ERROR Failed to configure any queues on dev %u",
  // 		dev_id);
  ad->nb_queues = queue_id;
  std::cout << "ad->nb_queues is: " << unsigned(ad->nb_queues) << std::endl;

  set_avail_op(ad, op_type);

  return TEST_SUCCESS;
}

static int add_active_device(uint8_t dev_id, struct rte_bbdev_info *info) {
  int ret;

  active_devs[0].driver_name = info->drv.driver_name;
  active_devs[0].dev_id = dev_id;

  ret = add_bbdev_dev(dev_id, info);
  if (ret == TEST_SUCCESS) ++nb_active_devs;
  return ret;
}

int main(int argc, char *argv[]) {
  // EAL initialization && ACC100 Card device detection and initialization
  std::string core_list = std::to_string(34);
  // + "," + std::to_string(35) + "," + std::to_string(36) + "," + std::to_string(37) + "," + std::to_string(38);
  const char *rte_argv[] = {"txrx",        "-l",           core_list.c_str(),
                            "--log-level", "lib.eal:info", nullptr};
  int rte_argc = static_cast<int>(sizeof(rte_argv) / sizeof(rte_argv[0])) - 1;

  printf("Remaining command line arguments:\n");
  for (int i = 0; i < argc; ++i) {
    printf("argv[%d]: %s\n", i, rte_argv[i]);
  }
  // Initialize DPDK environment
  std::cout << "getting ready to init dpdk" << std::endl;
  std::cout << "rte_argc is:" << rte_argc << std::endl;
  int ret = rte_eal_init(rte_argc, const_cast<char **>(rte_argv));
  // int ret = rte_eal_init(18, const_cast<char**>(rte_argv));

  RtAssert(
      ret >= 0,
      "Failed to initialize DPDK.  Are you running with root permissions?");

  int nb_bbdevs = rte_bbdev_count();
  if (nb_bbdevs == 0) rte_exit(EXIT_FAILURE, "No bbdevs detected!\n");
  std::cout << "nb_bbdevs is: " << nb_bbdevs << std::endl;
  std::cout << "trying to setup HW acc100" << std::endl;
  int ret_acc;
  uint8_t dev_id;
  // uint8_t nb_devs_added = 0;
  struct rte_bbdev_info info;
  // std::cout << "dev_id: " << unsigned(dev_id) << std::endl;
  // RTE_BBDEV_FOREACH(dev_id) {
  //   std::cout << "dev_id: " << unsigned(dev_id) << std::endl;
  rte_bbdev_info_get(dev_id, &info);

  const struct rte_bbdev_info *dev_info = &info;
  const struct rte_bbdev_op_cap *op_cap = dev_info->drv.capabilities;
  for (unsigned int i = 0; op_cap->type != RTE_BBDEV_OP_NONE; ++i, ++op_cap) {
    std::cout << "capabilities is: " << op_cap->type << std::endl;
  }

  std::cout << "[1] added device name is: " << info.dev_name << std::endl;
  std::cout << "[1] added device socket id is: " << info.socket_id << std::endl;
  std::cout << "[1] added device driver name is: " << info.drv.driver_name
            << std::endl;
  std::cout << "[1] Number of queues currently configured is: "
            << info.num_queues << std::endl;

  rte_bbdev_intr_enable(dev_id);
  struct active_device *ad;
  ad = &active_devs[dev_id];
  rte_bbdev_info_get(ad->dev_id, &info);

  std::cout << "double check: !!!!" << std::endl;
  std::cout << "[2] added device name is: " << info.dev_name << std::endl;
  std::cout << "[2] added device socket id is: " << info.socket_id << std::endl;
  std::cout << "[2] added device driver name is: " << info.drv.driver_name
            << std::endl;
  std::cout << "[2] Number of queues currently configured is: "
            << info.num_queues << std::endl;

  // shoudl create mempool for bbdev operations
  struct thread_params *t_params;
  uint16_t num_lcores = 4;
  unsigned int lcore_id, used_cores = 0;

  t_params = static_cast<struct thread_params *>(rte_zmalloc(
      NULL, num_lcores * sizeof(struct thread_params), RTE_CACHE_LINE_SIZE));

  uint16_t num_ops = 2047;
  uint16_t burst_sz = 1;
  // int rte_alloc;
  struct rte_mempool *ops_mp;
  struct rte_mempool *in_mbuf_pool;
  struct rte_mempool *out_mbuf_pool;

  ret = rte_bbdev_setup_queues(dev_id, 4, info.socket_id);
  if (ret < 0)
    rte_exit(EXIT_FAILURE, "ERROR(%d): BBDEV %u not configured properly\n", ret,
             dev_id);

  struct rte_bbdev_queue_conf qconf;
  qconf.socket = info.socket_id;
  qconf.queue_size = info.drv.queue_size_lim;
  qconf.priority = 0;
  qconf.deferred_start = 0;
  qconf.op_type = RTE_BBDEV_OP_LDPC_DEC;

  for (int q_id = 0; q_id < 4; q_id++) {
    /* Configure all queues belonging to this bbdev device */
    ret = rte_bbdev_queue_configure(dev_id, q_id, &qconf);
    if (ret != 0) {
      printf("Allocated all queues (id=%u) at prio%u on dev%u\n", q_id,
             qconf.priority, dev_id);
      qconf.priority++;
      ret = rte_bbdev_queue_configure(dev_id, q_id, &qconf);
    }
    if (ret != 0) {
      printf("All queues on dev %u allocated: %u\n", dev_id, q_id);
      break;
    }
    // if (ret < 0)
    //   rte_exit(EXIT_FAILURE,
    //   "ERROR(%d): BBDEV %u queue %u not configured properly\n",
    //   ret, dev_id, q_id);
  }

  ret = rte_bbdev_start(dev_id);
  std::cout << "start bbdev value is: " << ret << std::endl;

  ops_mp = rte_bbdev_op_pool_create("RTE_BBDEV_OP_LDPC_DEC_poo",
                                    RTE_BBDEV_OP_LDPC_DEC, 2047, OPS_CACHE_SIZE,
                                    rte_socket_id());

  std::cout << "num ops is: " << unsigned(num_ops) << std::endl;
  std::cout << "socket id is: " << unsigned(rte_socket_id()) << std::endl;

  if (ops_mp == nullptr) {
    std::cerr << "Error: Failed to create memory pool for bbdev operations."
              << std::endl;
  } else {
    std::cout << "Memory pool for bbdev operations created successfully."
              << std::endl;
  }

  rte_mempool *mbuf_pool =
      rte_pktmbuf_pool_create("bbdev_mbuf_pool", NB_MBUF, 256, 0,
                              RTE_MBUF_DEFAULT_BUF_SIZE, rte_socket_id());
  if (mbuf_pool == NULL) rte_exit(EXIT_FAILURE, "Unable to create\n");

  char pool_name[RTE_MEMPOOL_NAMESIZE];

  size_t start_tsc = GetTime::WorkerRdtsc();

  in_mbuf_pool = rte_pktmbuf_pool_create("in_pool_0", 16383, 0, 0, 22744, 0);
  out_mbuf_pool =
      rte_pktmbuf_pool_create("hard_out_pool_0", 16383, 0, 0, 22744, 0);

  size_t middle_time = GetTime::WorkerRdtsc();

  //   rte_mempool_free(in_mbuf_pool);
  //   rte_mempool_free(out_mbuf_pool);

  size_t end = GetTime::WorkerRdtsc();

  size_t duration_1 = middle_time - start_tsc;
  size_t duration_2 = end - middle_time;
  size_t total = end - start_tsc;

  double freq_ghz = GetTime::MeasureRdtscFreq();

  std::cout << std::endl;
  std::printf("Creating pktmbuf takes %.2f us \n",
              GetTime::CyclesToUs(duration_1, freq_ghz));
  std::printf("Freeing pktmbuf takes %.2f us \n",
              GetTime::CyclesToUs(duration_2, freq_ghz));
  std::printf("Total time takes %.2f us \n",
              GetTime::CyclesToUs(total, freq_ghz));

  struct rte_bbdev_op_data **inputs;
  struct rte_bbdev_op_data **hard_outputs;

  inputs =
      (struct rte_bbdev_op_data **)malloc(sizeof(struct rte_bbdev_op_data *));
  hard_outputs =
      (struct rte_bbdev_op_data **)malloc(sizeof(struct rte_bbdev_op_data *));

  int ret_socket_in = allocate_buffers_on_socket(
      inputs, 1 * sizeof(struct rte_bbdev_op_data), 0);
  int ret_socket_hard_out = allocate_buffers_on_socket(
      hard_outputs, 1 * sizeof(struct rte_bbdev_op_data), 0);

  char *data;
  struct rte_bbdev_op_data *bufs = *inputs;
  // std::cout << rte_mempool_avail_count(mbuf_pool) << std::endl;
  // if (rte_mempool_avail_count(mbuf_pool) == 0) {
  //     printf("No more mbufs available in the pool! - input\n");
  //     return -1;
  // }

  size_t start_pktmbuf_alloc = GetTime::WorkerRdtsc();

  struct rte_mbuf *m_head = rte_pktmbuf_alloc(in_mbuf_pool);

  size_t end_pktmbuf_alloc = GetTime::WorkerRdtsc();

  // if (m_head == nullptr) {
  //     std::cerr << "Error: Unable to create mbuf pool: " << rte_strerror(rte_errno) << std::endl;
  //     return -1;
  // }

  bufs[0].data = m_head;
  bufs[0].offset = 0;
  bufs[0].length = 0;

  //   data = rte_pktmbuf_append(m_head, ldpc_config.NumCbCodewLen());

  // Copy data from demod_data to the mbuf
  //   rte_memcpy(data, llr_buffer_ptr + (0 * ldpc_config.NumCbCodewLen()), ldpc_config.NumCbCodewLen());
  //   bufs[0].length += ldpc_config.NumCbCodewLen();
  //   auto cfg = std::make_unique<Config>(FLAGS_conf_file.c_str());

  //   Direction dir =
  //       cfg->Frame().NumULSyms() > 0 ? Direction::kUplink : Direction::kDownlink;
  //   const LDPCconfig& ldpc_config = cfg->LdpcConfig(dir);

  size_t start_pktmbuf_free = GetTime::WorkerRdtsc();

  rte_pktmbuf_free(m_head);

  size_t end_pktmbuf_free = GetTime::WorkerRdtsc();

  std::printf(
      "pktmbuf alloc takes %.2f us \n",
      GetTime::CyclesToUs(end_pktmbuf_alloc - start_pktmbuf_alloc, freq_ghz));
  std::printf(
      "Freeing pktmbuf takes %.2f us \n",
      GetTime::CyclesToUs(end_pktmbuf_free - start_pktmbuf_free, freq_ghz));
  std::printf("Total time takes %.2f us \n\n",
              GetTime::CyclesToUs((end_pktmbuf_alloc - start_pktmbuf_alloc) +
                                      (end_pktmbuf_free - start_pktmbuf_free),
                                  freq_ghz));

  //   std::printf("ref_op_time %.2f us \n", GetTime::CyclesToUs(start_ref_op_create - end_ref_op_create, freq_ghz));
  struct rte_bbdev_dec_op *ref_dec_op[1];

  int rte_alloc_ref = rte_bbdev_dec_op_alloc_bulk(ops_mp, ref_dec_op, burst_sz);

  uint8_t test = 1;
  uint16_t test2 = 1;
  uint32_t test3 = 1;

  // ref_dec_op[i]->ldpc_dec.op_flags += RTE_BBDEV_LDPC_ITERATION_STOP_ENABLE;
  size_t start_ref_op_create = GetTime::WorkerRdtsc();
  ref_dec_op[0]->ldpc_dec.basegraph = (uint8_t)test;
  ref_dec_op[0]->ldpc_dec.z_c = (uint16_t)test2;
  ref_dec_op[0]->ldpc_dec.n_filler = (uint16_t)test2;
  ref_dec_op[0]->ldpc_dec.rv_index = (uint8_t)test;
  ref_dec_op[0]->ldpc_dec.n_cb = (uint16_t)test2;
  ref_dec_op[0]->ldpc_dec.q_m = (uint8_t)test;
  ref_dec_op[0]->ldpc_dec.code_block_mode = (uint8_t)test3;
  ref_dec_op[0]->ldpc_dec.cb_params.e = (uint32_t)44;
  if (!check_bit(ref_dec_op[0]->ldpc_dec.op_flags,
                 RTE_BBDEV_LDPC_ITERATION_STOP_ENABLE)) {
    ref_dec_op[0]->ldpc_dec.op_flags += RTE_BBDEV_LDPC_ITERATION_STOP_ENABLE;
  }
  // if (check_bit(ref_dec_op[i]->ldpc_dec.op_flags, RTE_BBDEV_LDPC_ITERATION_STOP_ENABLE)){
  //   ref_dec_op[i]->ldpc_dec.op_flags -= RTE_BBDEV_LDPC_ITERATION_STOP_ENABLE;
  // }
  ref_dec_op[0]->ldpc_dec.iter_max = (uint8_t)test;
  // ref_dec_op[i]->ldpc_dec.iter_count = (uint8_t) ref_dec_op[i]->ldpc_dec.iter_max;
  // std::cout<<"iter count is: " << unsigned(ref_dec_op[i]->ldpc_dec.iter_count) << std::endl;

  // ref_dec_op[i]->ldpc_dec.op_flags = RTE_BBDEV_LDPC_ITERATION_STOP_ENABLE;
  ref_dec_op[0]->opaque_data = (void *)(uintptr_t)0;
  size_t end_ref_op_create = GetTime::WorkerRdtsc();

  std::printf(
      "ref_op time is %.2f us \n\n",
      GetTime::CyclesToUs(end_ref_op_create - start_ref_op_create, freq_ghz));

  rte_mempool_free(in_mbuf_pool);
  rte_mempool_free(out_mbuf_pool);

  return 0;
}
