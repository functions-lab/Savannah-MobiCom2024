/**
 * @file agora_worker.h
 * @brief Declaration file for the main Agora worker class
 */

#ifndef AGORA_WORKER_H_
#define AGORA_WORKER_H_

#include <memory>
#include <thread>
#include <vector>

#include "agora_buffer.h"
#include "config.h"
#include "csv_logger.h"
#include "doer.h"
#include "mac_scheduler.h"
#include "mat_logger.h"
#include "phy_stats.h"
#include "stats.h"

class AgoraWorker {
 public:
  explicit AgoraWorker(Config* cfg, MacScheduler* mac_sched, Stats* stats,
                       PhyStats* phy_stats, MessageInfo* message,
                       AgoraBuffer* buffer, FrameInfo* frame);
  ~AgoraWorker();

#ifdef SINGLE_THREAD
  void RunWorker();
#endif

 private:
#ifdef SINGLE_THREAD
  void InitializeWorker();

  std::vector<std::shared_ptr<Doer> > computers_vec;
  std::vector<EventType> events_vec;
  int tid;  // TODO: remove thread id for single-core
  size_t cur_qid;
  size_t empty_queue_itrs;
  bool empty_queue;
#else
  void WorkerThread(int tid);
  void CreateThreads();
  void JoinThreads();

  std::vector<std::thread> workers_;
#endif

  const size_t base_worker_core_offset_;

  Config* const config_;

  MacScheduler* mac_sched_;
  Stats* stats_;
  PhyStats* phy_stats_;
  MessageInfo* message_;
  AgoraBuffer* buffer_;
  FrameInfo* frame_;
};

#endif  // AGORA_WORKER_H_