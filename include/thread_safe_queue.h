#ifndef THREAD_SAFE_QUEUE_H
#define THREAD_SAFE_QUEUE_H

#include <queue>
#include <mutex>
#include <condition_variable>
#include <atomic>

template <typename T>
class ThreadSafeQueue {
public:
    ThreadSafeQueue() : shutdown_(false) {}

    void push(T value) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (shutdown_) return; // Don't accept new items if shutting down
        queue_.push(std::move(value));
        cond_.notify_one();
    }

    // Now returns 'false' if the queue was shut down
    bool wait_and_pop(T& value) {
        std::unique_lock<std::mutex> lock(mutex_);
        cond_.wait(lock, [this]{ return !queue_.empty() || shutdown_; });
        
        if (shutdown_ && queue_.empty()) {
            return false;
        }

        value = std::move(queue_.front());
        queue_.pop();
        return true;
    }
    
    // Wakes up any waiting threads and stops the queue
    void shutdown() {
        std::lock_guard<std::mutex> lock(mutex_);
        shutdown_ = true;
        cond_.notify_all();
    }

private:
    mutable std::mutex mutex_;
    std::queue<T> queue_;
    std::condition_variable cond_;
    std::atomic<bool> shutdown_;
};

#endif // THREAD_SAFE_QUEUE_H