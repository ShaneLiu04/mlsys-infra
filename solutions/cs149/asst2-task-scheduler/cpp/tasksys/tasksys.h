// CS149 asst2 task systems in C++17: the official IRunnable / ITaskSystem API,
// Part A (serial, spawn, spinning pool, sleeping pool) and Part B (async with deps).
#pragma once

#include <atomic>
#include <condition_variable>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "../common/common.h"

typedef int TaskID;

class IRunnable {
public:
    virtual ~IRunnable() {}
    virtual void runTask(int task_id, int num_total_tasks) = 0;
};

class ITaskSystem {
public:
    explicit ITaskSystem(int num_threads) : num_threads_(num_threads) {}
    virtual ~ITaskSystem() {}
    virtual const char* name() = 0;
    virtual void run(IRunnable* runnable, int num_total_tasks) = 0;
    virtual TaskID runAsyncWithDeps(IRunnable* runnable, int num_total_tasks, const std::vector<TaskID>& deps) = 0;
    virtual void sync() = 0;

protected:
    int num_threads_;
};

// Part A systems implement async the way the official Part A starter does: run synchronously.
class SyncOnlyTaskSystem : public ITaskSystem {
public:
    using ITaskSystem::ITaskSystem;
    TaskID runAsyncWithDeps(IRunnable* r, int n, const std::vector<TaskID>&) override {
        run(r, n);
        return next_id_++;
    }
    void sync() override {}

private:
    TaskID next_id_ = 0;
};

class TaskSystemSerial : public SyncOnlyTaskSystem {
public:
    TaskSystemSerial(int n, const std::vector<int>& = {}) : SyncOnlyTaskSystem(n) {}
    const char* name() override { return "serial"; }
    void run(IRunnable* r, int n) override {
        for (int i = 0; i < n; ++i) r->runTask(i, n);
    }
};

// Spawns num_threads threads on every run(); tasks are handed out by an atomic counter.
class TaskSystemParallelSpawn : public SyncOnlyTaskSystem {
public:
    TaskSystemParallelSpawn(int n, const std::vector<int>& cpus = {}) : SyncOnlyTaskSystem(n), cpus_(cpus) {}
    const char* name() override { return "spawn"; }
    void run(IRunnable* r, int n) override {
        std::atomic<int> next{0};
        int nt = std::min(num_threads_, std::max(n, 1));
        std::vector<std::thread> th;
        th.reserve(nt);
        for (int t = 0; t < nt; ++t)
            th.emplace_back([&, t] {
                pin_thread(cpu_for(cpus_, t));
                for (int i; (i = next.fetch_add(1, std::memory_order_relaxed)) < n;) r->runTask(i, n);
            });
        for (auto& x : th) x.join();
    }

private:
    std::vector<int> cpus_;
};

// Shared bulk-launch state for the Part A pools. `state_` packs (generation << 32 | next index)
// so a worker can only claim an index of the launch whose runnable/total it read: a CAS that
// races with the next launch's publish fails instead of stealing a task from the new launch.
class BulkLaunchState {
protected:
    std::atomic<unsigned long long> state_{0};
    std::atomic<IRunnable*> runnable_{nullptr};
    std::atomic<int> total_{0};
    std::atomic<int> done_{0};

    static unsigned gen_of(unsigned long long v) { return (unsigned)(v >> 32); }
    static int idx_of(unsigned long long v) { return (int)(v & 0xffffffffu); }

    // Claims and runs tasks of generation `g` until none are left. Returns true if this worker
    // completed the last task of the launch.
    bool drain(unsigned g) {
        bool last = false;
        unsigned long long v = state_.load(std::memory_order_acquire);
        while (gen_of(v) == g) {
            int total = total_.load(std::memory_order_relaxed);
            int i = idx_of(v);
            if (i >= total) break;
            if (!state_.compare_exchange_weak(v, v + 1, std::memory_order_acq_rel)) continue;
            runnable_.load(std::memory_order_relaxed)->runTask(i, total);
            if (done_.fetch_add(1, std::memory_order_acq_rel) + 1 == total) last = true;
            v = state_.load(std::memory_order_acquire);
        }
        return last;
    }

    unsigned publish(IRunnable* r, int n) {
        unsigned g = gen_of(state_.load(std::memory_order_relaxed)) + 1;
        runnable_.store(r, std::memory_order_relaxed);
        total_.store(n, std::memory_order_relaxed);
        done_.store(0, std::memory_order_relaxed);
        state_.store((unsigned long long)g << 32, std::memory_order_release);
        return g;
    }
};

// Persistent workers busy-wait for work; the caller busy-waits for completion.
class TaskSystemParallelThreadPoolSpinning : public SyncOnlyTaskSystem, BulkLaunchState {
public:
    TaskSystemParallelThreadPoolSpinning(int n, const std::vector<int>& cpus = {}) : SyncOnlyTaskSystem(n) {
        for (int t = 0; t < n; ++t)
            th_.emplace_back([this, t, cpus] {
                pin_thread(cpu_for(cpus, t));
                while (!stop_.load(std::memory_order_relaxed)) {
                    unsigned long long v = state_.load(std::memory_order_acquire);
                    if (idx_of(v) < total_.load(std::memory_order_relaxed)) drain(gen_of(v));
                    else cpu_relax();
                }
            });
    }
    ~TaskSystemParallelThreadPoolSpinning() override {
        stop_ = true;
        for (auto& x : th_) x.join();
    }
    const char* name() override { return "spinning"; }
    void run(IRunnable* r, int n) override {
        if (n <= 0) return;
        publish(r, n);
        while (done_.load(std::memory_order_acquire) < n) cpu_relax();
    }

private:
    std::atomic<bool> stop_{false};
    std::vector<std::thread> th_;
};

// Persistent workers sleep on a condition variable between launches.
class TaskSystemParallelThreadPoolSleeping : public SyncOnlyTaskSystem, BulkLaunchState {
public:
    TaskSystemParallelThreadPoolSleeping(int n, const std::vector<int>& cpus = {}) : SyncOnlyTaskSystem(n) {
        for (int t = 0; t < n; ++t)
            th_.emplace_back([this, t, cpus] {
                pin_thread(cpu_for(cpus, t));
                unsigned seen = 0;
                for (;;) {
                    unsigned g;
                    {
                        std::unique_lock<std::mutex> lk(m_);
                        work_cv_.wait(lk, [&] { return stop_ || gen_of(state_.load()) != seen; });
                        if (stop_) return;
                        g = seen = gen_of(state_.load());
                    }
                    if (drain(g)) {
                        std::lock_guard<std::mutex> lk(m_);
                        finished_gen_ = g;
                        done_cv_.notify_one();
                    }
                }
            });
    }
    ~TaskSystemParallelThreadPoolSleeping() override {
        {
            std::lock_guard<std::mutex> lk(m_);
            stop_ = true;
        }
        work_cv_.notify_all();
        for (auto& x : th_) x.join();
    }
    const char* name() override { return "sleeping"; }
    void run(IRunnable* r, int n) override {
        if (n <= 0) return;
        unsigned g;
        {
            std::lock_guard<std::mutex> lk(m_);
            g = publish(r, n);
        }
        work_cv_.notify_all();
        std::unique_lock<std::mutex> lk(m_);
        done_cv_.wait(lk, [&] { return finished_gen_ == g; });
    }

private:
    std::mutex m_;
    std::condition_variable work_cv_, done_cv_;
    bool stop_ = false;
    unsigned finished_gen_ = 0;
    std::vector<std::thread> th_;
};

// Part B: bulk launches with dependencies. One mutex guards the launch table and the ready
// queue; a launch enters the ready queue when its pending-dependency count reaches zero, and
// workers take one task at a time from the front ready launch, so independent launches
// interleave and fill each other's tail. Spin=true polls instead of sleeping.
template <bool Spin>
class TaskSystemAsync : public ITaskSystem {
public:
    TaskSystemAsync(int n, const std::vector<int>& cpus = {}) : ITaskSystem(n) {
        for (int t = 0; t < n; ++t)
            th_.emplace_back([this, t, cpus] {
                pin_thread(cpu_for(cpus, t));
                worker();
            });
    }
    ~TaskSystemAsync() override {
        {
            std::lock_guard<std::mutex> lk(m_);
            stop_ = true;
        }
        stop_flag_ = true;
        work_cv_.notify_all();
        for (auto& x : th_) x.join();
    }
    const char* name() override { return Spin ? "async_spinning" : "async_sleeping"; }

    void run(IRunnable* r, int n) override {
        runAsyncWithDeps(r, n, {});
        sync();
    }

    TaskID runAsyncWithDeps(IRunnable* r, int n, const std::vector<TaskID>& deps) override {
        std::lock_guard<std::mutex> lk(m_);
        TaskID id = next_id_++;
        if (n <= 0 && deps.empty()) return id;
        auto L = std::make_unique<Launch>();
        L->r = r;
        L->total = n;
        for (TaskID d : deps) {
            auto it = live_.find(d);
            if (it == live_.end()) continue;  // already finished
            it->second->dependents.push_back(id);
            L->pending++;
        }
        Launch* p = L.get();
        live_[id] = std::move(L);
        unfinished_.fetch_add(1);
        if (p->pending == 0) make_ready(id, p);
        return id;
    }

    void sync() override {
        if (Spin) {
            while (unfinished_.load(std::memory_order_acquire) > 0) cpu_relax();
        } else {
            std::unique_lock<std::mutex> lk(m_);
            done_cv_.wait(lk, [&] { return unfinished_.load() == 0; });
        }
    }

private:
    struct Launch {
        IRunnable* r = nullptr;
        int total = 0, next = 0, done = 0, pending = 0;
        std::vector<TaskID> dependents;
    };

    // Caller holds m_.
    void make_ready(TaskID id, Launch* L) {
        if (L->total == 0) {
            finish(id, L);
            return;
        }
        ready_.push_back({id, L});
        ready_count_.fetch_add(1, std::memory_order_release);
        if (!Spin) work_cv_.notify_all();
    }

    // Caller holds m_. Releases dependents, then erases the launch.
    void finish(TaskID id, Launch* L) {
        std::vector<TaskID> deps;
        deps.swap(L->dependents);
        live_.erase(id);
        for (TaskID d : deps) {
            Launch* D = live_[d].get();
            if (--D->pending == 0) make_ready(d, D);
        }
        if (unfinished_.fetch_sub(1) == 1 && !Spin) done_cv_.notify_all();
    }

    void worker() {
        std::unique_lock<std::mutex> lk(m_, std::defer_lock);
        for (;;) {
            if (Spin) {
                while (ready_count_.load(std::memory_order_acquire) == 0) {
                    if (stop_flag_.load(std::memory_order_relaxed)) return;
                    cpu_relax();
                }
                lk.lock();
                if (ready_.empty()) {
                    lk.unlock();
                    continue;
                }
            } else {
                lk.lock();
                work_cv_.wait(lk, [&] { return stop_ || !ready_.empty(); });
                if (stop_) return;
            }
            auto [id, L] = ready_.front();
            int i = L->next++;
            if (L->next == L->total) {
                ready_.pop_front();
                ready_count_.fetch_sub(1, std::memory_order_relaxed);
            }
            lk.unlock();
            L->r->runTask(i, L->total);
            lk.lock();
            if (++L->done == L->total) finish(id, L);
            lk.unlock();
        }
    }

    std::mutex m_;
    std::condition_variable work_cv_, done_cv_;
    bool stop_ = false;
    std::atomic<bool> stop_flag_{false};
    std::atomic<int> unfinished_{0}, ready_count_{0};
    TaskID next_id_ = 0;
    std::unordered_map<TaskID, std::unique_ptr<Launch>> live_;
    std::deque<std::pair<TaskID, Launch*>> ready_;
    std::vector<std::thread> th_;
};

using TaskSystemAsyncSleeping = TaskSystemAsync<false>;
using TaskSystemAsyncSpinning = TaskSystemAsync<true>;

inline const std::vector<std::string>& all_systems() {
    static const std::vector<std::string> v = {"serial", "spawn", "spinning", "sleeping", "async_spinning",
                                               "async_sleeping"};
    return v;
}

inline std::unique_ptr<ITaskSystem> make_system(const std::string& s, int n, const std::vector<int>& cpus) {
    if (s == "serial") return std::make_unique<TaskSystemSerial>(n, cpus);
    if (s == "spawn") return std::make_unique<TaskSystemParallelSpawn>(n, cpus);
    if (s == "spinning") return std::make_unique<TaskSystemParallelThreadPoolSpinning>(n, cpus);
    if (s == "sleeping") return std::make_unique<TaskSystemParallelThreadPoolSleeping>(n, cpus);
    if (s == "async_spinning") return std::make_unique<TaskSystemAsyncSpinning>(n, cpus);
    if (s == "async_sleeping") return std::make_unique<TaskSystemAsyncSleeping>(n, cpus);
    std::fprintf(stderr, "unknown system %s\n", s.c_str());
    std::exit(2);
}
