/**
 * @file agora_buffer.h
 * @brief Declaration file for the AgoraBuffer class
 */

#ifndef AGORA_BUFFER_H_
#define AGORA_BUFFER_H_

#include <array>
#include <cstddef>
#include <queue>

#include "common_typedef_sdk.h"
#include "concurrent_queue_wrapper.h"
#include "concurrentqueue.h"
#include "config.h"
#include "mac_scheduler.h"
#include "memory_manage.h"
#include "message.h"
#include "symbols.h"
#include "utils.h"

class AgoraBuffer {
 public:
  explicit AgoraBuffer(Config* const cfg);
  // Delete copy constructor and copy assignment
  AgoraBuffer(AgoraBuffer const&) = delete;
  AgoraBuffer& operator=(AgoraBuffer const&) = delete;
  ~AgoraBuffer();

  inline PtrGrid<kFrameWnd, kMaxUEs, complex_float>& GetCsi() {
    return csi_buffer_;
  }
  inline PtrGrid<kFrameWnd, kMaxDataSCs, complex_float>& GetUlBeamMatrix() {
    return ul_beam_matrix_;
  }
  inline PtrGrid<kFrameWnd, kMaxDataSCs, complex_float>& GetDlBeamMatrix() {
    return dl_beam_matrix_;
  }
  inline PtrCube<kFrameWnd, kMaxSymbols, kMaxUEs, int8_t>& GetDemod() {
    return demod_buffer_;
  }

  inline PtrCube<kFrameWnd, kMaxSymbols, kMaxUEs, uint32_t>& GetLLR() {
    return llr_buffer_;
  }

  inline PtrCube<kFrameWnd, kMaxSymbols, kMaxUEs, int8_t>& GetDecod() {
    return decoded_buffer_;
  }
  inline Table<complex_float>& GetFft() { return fft_buffer_; }
  inline Table<complex_float>& GetEqual() { return equal_buffer_; }
  inline Table<complex_float>& GetUeSpecPilot() {
    return ue_spec_pilot_buffer_;
  }
  inline Table<complex_float>& GetIfft() { return dl_ifft_buffer_; }
  inline Table<complex_float>& GetCalibUlMsum() {
    return calib_ul_msum_buffer_;
  }
  inline Table<complex_float>& GetCalibDlMsum() {
    return calib_dl_msum_buffer_;
  }
  inline Table<std::complex<int16_t>> GetDlBcastSignal() {
    return dl_bcast_socket_buffer_;
  }
  inline Table<int8_t>& GetDlModBits() { return dl_mod_bits_buffer_; }
  inline Table<int8_t>& GetDlBits() { return dl_bits_buffer_; }
  inline Table<int8_t>& GetDlBitsStatus() { return dl_bits_buffer_status_; }

  inline size_t GetUlSocketSize() const { return ul_socket_buf_size_; }
  inline Table<char>& GetUlSocket() { return ul_socket_buffer_; }
  inline char* GetDlSocket() { return dl_socket_buffer_; }
  inline Table<complex_float>& GetCalibUl() { return calib_ul_buffer_; }
  inline Table<complex_float>& GetCalibDl() { return calib_dl_buffer_; }
  inline Table<complex_float>& GetCalib() { return calib_buffer_; }

  inline std::array<arma::fmat, kFrameWnd>& GetUlPhaseBase() {
    return ul_phase_base_;
  }
  inline std::array<arma::fmat, kFrameWnd>& GetUlPhaseShiftPerSymbol() {
    return ul_phase_shift_per_symbol_;
  }

 private:
  void AllocateTables();
  void AllocatePhaseShifts();
  void FreeTables();

  Config* const config_;
  const size_t ul_socket_buf_size_;

  PtrGrid<kFrameWnd, kMaxUEs, complex_float> csi_buffer_;
  PtrGrid<kFrameWnd, kMaxDataSCs, complex_float> ul_beam_matrix_;
  PtrGrid<kFrameWnd, kMaxDataSCs, complex_float> dl_beam_matrix_;
  PtrCube<kFrameWnd, kMaxSymbols, kMaxUEs, int8_t> demod_buffer_;
  PtrCube<kFrameWnd, kMaxSymbols, kMaxUEs, uint32_t> llr_buffer_;
  PtrCube<kFrameWnd, kMaxSymbols, kMaxUEs, int8_t> decoded_buffer_;
  Table<complex_float> fft_buffer_;
  Table<complex_float> equal_buffer_;
  Table<complex_float> ue_spec_pilot_buffer_;
  Table<complex_float> dl_ifft_buffer_;
  Table<complex_float> calib_ul_msum_buffer_;
  Table<complex_float> calib_dl_msum_buffer_;
  Table<complex_float> calib_buffer_;
  Table<int8_t> dl_mod_bits_buffer_;
  Table<int8_t> dl_bits_buffer_;
  Table<int8_t> dl_bits_buffer_status_;
  Table<std::complex<int16_t>> dl_bcast_socket_buffer_;

  std::array<arma::fmat, kFrameWnd> ul_phase_base_;
  std::array<arma::fmat, kFrameWnd> ul_phase_shift_per_symbol_;

  Table<char> ul_socket_buffer_;
  char* dl_socket_buffer_;
  Table<complex_float> calib_ul_buffer_;
  Table<complex_float> calib_dl_buffer_;
};

#if !defined(SINGLE_THREAD)
struct SchedInfo {
  moodycamel::ConcurrentQueue<EventData> concurrent_q_;
  moodycamel::ProducerToken* ptok_;
};
#endif

// Used to communicate between the manager and the streamer/worker class
// Needs to manage its own memory
class MessageInfo {
 public:
  explicit MessageInfo(size_t queue_size, size_t rx_queue_size,
                       size_t num_socket_thread)
      : num_socket_thread(num_socket_thread) {
    tx_concurrent_queue = moodycamel::ConcurrentQueue<EventData>(queue_size);
    rx_concurrent_queue = moodycamel::ConcurrentQueue<EventData>(rx_queue_size);

    for (size_t i = 0; i < num_socket_thread; i++) {
      rx_ptoks_ptr_[i] = new moodycamel::ProducerToken(rx_concurrent_queue);
      tx_ptoks_ptr_[i] = new moodycamel::ProducerToken(tx_concurrent_queue);
    }

#if !defined(SINGLE_THREAD)
    // Allocate memory for the task concurrent queues
    Alloc(queue_size);
#endif
  }
  ~MessageInfo() {
    for (size_t i = 0; i < num_socket_thread; i++) {
      delete rx_ptoks_ptr_[i];
      delete tx_ptoks_ptr_[i];
      rx_ptoks_ptr_[i] = nullptr;
      tx_ptoks_ptr_[i] = nullptr;
    }

#if !defined(SINGLE_THREAD)
    // Free memory for the task concurrent queues
    Free();
#endif
  }

  inline moodycamel::ConcurrentQueue<EventData>* GetTxConQ() {
    return &tx_concurrent_queue;
  }
  inline moodycamel::ConcurrentQueue<EventData>* GetRxConQ() {
    return &rx_concurrent_queue;
  }
  inline moodycamel::ProducerToken** GetTxPTokPtr() { return tx_ptoks_ptr_; }
  inline moodycamel::ProducerToken** GetRxPTokPtr() { return rx_ptoks_ptr_; }
  inline moodycamel::ProducerToken* GetTxPTokPtr(size_t idx) {
    return tx_ptoks_ptr_[idx];
  }
  inline moodycamel::ProducerToken* GetRxPTokPtr(size_t idx) {
    return rx_ptoks_ptr_[idx];
  }

#ifdef SINGLE_THREAD
  inline std::queue<EventData>* GetTaskQueue(EventType event_type, size_t qid) {
    return &task_queues.at(qid).at(static_cast<size_t>(event_type));
  }
  inline std::queue<EventData>& GetCompQueue(size_t qid) {
    return complete_task_queues_.at(qid);
  }
  inline void EnqueueEventTaskQueue(EventType event_type, size_t qid,
                                    EventData event) {
    this->GetTaskQueue(event_type, qid)->push(event);
  }
  inline size_t DequeueEventCompQueueBulk(size_t qid,
                                          std::vector<EventData>& events_list) {
    size_t total_events = 0;
    size_t max_events = events_list.size();
    std::queue<EventData>* comp_queue = &this->GetCompQueue(qid);
    while (!comp_queue->empty() && total_events < max_events) {
      events_list.at(total_events) = comp_queue->front();
      comp_queue->pop();
      ++total_events;
    }
    // if (total_events == max_events) {
    //   printf("Note: use up max space of complete queue\n");
    // }
    return total_events;
  }
#else
  inline moodycamel::ProducerToken* GetPtok(EventType event_type, size_t qid) {
    return task_queue_.at(qid).at(static_cast<size_t>(event_type)).ptok_;
  }
  inline moodycamel::ConcurrentQueue<EventData>* GetTaskQueue(
      EventType event_type, size_t qid) {
    return &task_queue_.at(qid)
                .at(static_cast<size_t>(event_type))
                .concurrent_q_;
  }
  inline moodycamel::ConcurrentQueue<EventData>& GetCompQueue(size_t qid) {
    return complete_task_queue_.at(qid);
  }
  inline moodycamel::ProducerToken* GetWorkerPtok(size_t qid,
                                                  size_t worker_id) {
    return worker_ptoks_ptr_.at(qid).at(worker_id);
  }
  inline void EnqueueEventTaskQueue(EventType event_type, size_t qid,
                                    EventData event) {
    TryEnqueueFallback(this->GetTaskQueue(event_type, qid),
                       this->GetPtok(event_type, qid), event);
  }
  inline size_t DequeueEventCompQueueBulk(size_t qid,
                                          std::vector<EventData>& events_list) {
    return this->GetCompQueue(qid).try_dequeue_bulk(&events_list.at(0),
                                                    events_list.size());
  }
#endif

 private:
  size_t num_socket_thread;
  // keep the concurrent queue to communicate to streamer thread
  moodycamel::ConcurrentQueue<EventData> tx_concurrent_queue;
  moodycamel::ConcurrentQueue<EventData> rx_concurrent_queue;
  moodycamel::ProducerToken* rx_ptoks_ptr_[kMaxThreads];
  moodycamel::ProducerToken* tx_ptoks_ptr_[kMaxThreads];

#ifdef SINGLE_THREAD
  std::array<std::array<std::queue<EventData>, kNumEventTypes>, kScheduleQueues>
      task_queues;
  std::array<std::queue<EventData>, kScheduleQueues> complete_task_queues_;
#else
  std::array<std::array<SchedInfo, kNumEventTypes>, kScheduleQueues>
      task_queue_;
  std::array<moodycamel::ConcurrentQueue<EventData>, kScheduleQueues>
      complete_task_queue_;
  std::array<std::array<moodycamel::ProducerToken*, kMaxThreads>,
             kScheduleQueues>
      worker_ptoks_ptr_;

  inline void Alloc(size_t queue_size) {
    // Allocate memory for the task concurrent queues
    for (auto& queue : complete_task_queue_) {
      queue = moodycamel::ConcurrentQueue<EventData>(queue_size);
    }
    for (auto& queue : task_queue_) {
      for (auto& event : queue) {
        event.concurrent_q_ =
            moodycamel::ConcurrentQueue<EventData>(queue_size);
        event.ptok_ = new moodycamel::ProducerToken(event.concurrent_q_);
      }
    }

    size_t queue_count = 0;
    for (auto& queue : worker_ptoks_ptr_) {
      for (auto& worker : queue) {
        worker =
            new moodycamel::ProducerToken(complete_task_queue_.at(queue_count));
      }
      queue_count++;
    }
  }

  inline void Free() {
    for (auto& queue : task_queue_) {
      for (auto& event : queue) {
        delete event.ptok_;
        event.ptok_ = nullptr;
      }
    }
    for (auto& queue : worker_ptoks_ptr_) {
      for (auto& worker : queue) {
        delete worker;
        worker = nullptr;
      }
    }
  }
#endif
};

struct FrameInfo {
  size_t cur_sche_frame_id_;
  size_t cur_proc_frame_id_;
};

#endif  // AGORA_BUFFER_H_
