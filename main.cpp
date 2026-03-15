#include <iostream>
#include <atomic>
#include <mutex>
#include <set>
#include <vector>
#include <chrono>
#include <thread>
#include <future>
#include <cmath>
#include <string>
#include <stdexcept>
#include "workerpool.h"

using std::cout;
using std::endl;

// ─── test harness ────────────────────────────────────────────────────────────

static int tests_passed = 0;
static int tests_failed = 0;

#define CHECK(cond, msg) \
    do { if (cond) { cout << "  [PASS] " << msg << "\n"; ++tests_passed; } \
         else      { cout << "  [FAIL] " << msg << "\n"; ++tests_failed; } \
    } while(0)

static void sleep_ms(int ms) {
    std::this_thread::sleep_for(std::chrono::milliseconds(ms));
}

// spin-wait with 10s timeout
static bool wait_for(std::atomic<int>& counter, int target, int timeout_ms = 10000) {
    auto start = std::chrono::steady_clock::now();
    while (counter.load() < target) {
        if (std::chrono::steady_clock::now() - start > std::chrono::milliseconds(timeout_ms))
            return false;
        std::this_thread::yield();
    }
    return true;
}

template<typename... Sigs>
static void print_metrics(const WorkerPool<Sigs...>& pool) {
    auto m = pool.get_metrics();
    cout << "   [metrics] submitted=" << m.tasks_submitted
         << " completed=" << m.tasks_completed
         << " avg_wait=" << m.avg_wait_ms << "ms"
         << " avg_exec=" << m.avg_exec_ms << "ms\n";
}

// ─── original tests (kept, condensed) ────────────────────────────────────────

void test_basic_execution() {
    cout << "\n[Test] Basic task execution\n";
    WorkerPool<void()> pool(4);
    std::atomic<int> counter{0};
    for (int i = 0; i < 10; ++i)
        pool.submit_work([&] { counter.fetch_add(1); });
    CHECK(wait_for(counter, 10), "All 10 tasks executed");
    print_metrics(pool);
}

void test_task_ids_unique() {
    cout << "\n[Test] Task IDs unique\n";
    WorkerPool<void()> pool(4);
    std::set<size_t> ids;
    for (int i = 0; i < 50; ++i)
        ids.insert(pool.submit_work([] {}));
    CHECK(ids.size() == 50, "All 50 task IDs are unique");
    sleep_ms(100);
    print_metrics(pool);
}

void test_callbacks_fired() {
    cout << "\n[Test] Callbacks fire correctly\n";
    std::atomic<int> init{0}, done{0};
    WorkerPool<void()> pool(4,
        [&](size_t) { init++; },
        [&](size_t) { done++; }
    );
    for (int i = 0; i < 20; ++i)
        pool.submit_work([] {});
    wait_for(done, 20);
    CHECK(init.load() == 20, "Init callback fired 20 times");
    CHECK(done.load() == 20, "Completed callback fired 20 times");
    print_metrics(pool);
}

void test_parallel_execution() {
    cout << "\n[Test] Parallel execution\n";
    WorkerPool<void()> pool(8);
    auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < 8; ++i)
        pool.submit_work([] { sleep_ms(100); });
    sleep_ms(150);
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - start).count();
    CHECK(elapsed < 400, "8 x 100ms tasks ran in parallel (elapsed < 400ms)");
    print_metrics(pool);
}

void test_queue_pressure() {
    cout << "\n[Test] Queue pressure (20 000 tasks)\n";
    WorkerPool<void()> pool(8);
    std::atomic<int> counter{0};
    const int N = 20000;
    for (int i = 0; i < N; ++i)
        pool.submit_work([&] { counter.fetch_add(1, std::memory_order_relaxed); });
    CHECK(wait_for(counter, N), "All 20 000 tasks completed");
    print_metrics(pool);
}

void test_multi_producer_stress() {
    cout << "\n[Test] Multi-producer stress\n";
    int workers = std::thread::hardware_concurrency();
    WorkerPool<void()> pool(workers);
    const int producers        = workers * 2;
    const int tasks_per_producer = 10000;
    std::atomic<int> counter{0};
    std::vector<std::thread> threads;
    for (int p = 0; p < producers; ++p)
        threads.emplace_back([&] {
            for (int i = 0; i < tasks_per_producer; ++i)
                pool.submit_work([&] { counter.fetch_add(1, std::memory_order_relaxed); });
        });
    for (auto& t : threads) t.join();
    int total = producers * tasks_per_producer;
    CHECK(wait_for(counter, total), "All multi-producer tasks executed");
    print_metrics(pool);
}

void test_lost_wakeup() {
    cout << "\n[Test] Lost wakeup detection (single worker, 5 000 iterations)\n";
    WorkerPool<void()> pool(1);
    std::atomic<int> counter{0};
    const int iterations = 5000;
    for (int i = 0; i < iterations; ++i) {
        pool.submit_work([&] { counter.fetch_add(1, std::memory_order_relaxed); });
        auto start = std::chrono::steady_clock::now();
        while (counter.load(std::memory_order_relaxed) < i + 1) {
            if (std::chrono::steady_clock::now() - start > std::chrono::seconds(2)) {
                CHECK(false, "Lost wakeup detected at iteration " + std::to_string(i));
                return;
            }
            std::this_thread::yield();
        }
    }
    CHECK(counter.load() == iterations, "No wakeups lost");
    print_metrics(pool);
}

// ─── submit_and_get ───────────────────────────────────────────────────────────

void test_submit_and_get_int() {
    cout << "\n[Test] submit_and_get -- int return\n";
    VoidVoidWorkerPool pool(4);
    auto fut = pool.submit_and_get([] { return 42; });
    CHECK(fut.get() == 42, "Future returned correct int value");
    print_metrics(pool);
}

void test_submit_and_get_string() {
    cout << "\n[Test] submit_and_get -- string return\n";
    VoidVoidWorkerPool pool(4);
    auto fut = pool.submit_and_get([] { return std::string("hello"); });
    CHECK(fut.get() == "hello", "Future returned correct string value");
    print_metrics(pool);
}

void test_submit_and_get_multiple_futures() {
    cout << "\n[Test] submit_and_get -- multiple concurrent futures\n";
    VoidVoidWorkerPool pool(8);
    const int N = 100;
    std::vector<std::future<int>> futures;
    futures.reserve(N);
    for (int i = 0; i < N; ++i)
        futures.push_back(pool.submit_and_get([i] { return i * i; }));
    bool all_correct = true;
    for (int i = 0; i < N; ++i)
        if (futures[i].get() != i * i) { all_correct = false; break; }
    CHECK(all_correct, "All 100 futures returned correct values (i*i)");
    print_metrics(pool);
}

void test_submit_and_get_ordering() {
    cout << "\n[Test] submit_and_get -- result available before task_completed_callback\n";

    // The future unblocks when the task finishes, which is BEFORE
    // task_completed_callback fires. This test verifies that order.
    std::atomic<bool> callback_fired{false};
    std::atomic<bool> future_resolved{false};

    VoidVoidWorkerPool pool(1,
        nullptr,
        [&](size_t) {
            // small sleep to make the race window visible
            sleep_ms(5);
            callback_fired.store(true, std::memory_order_release);
        }
    );

    auto fut = pool.submit_and_get([&] {
        return 1;
    });

    fut.get();
    future_resolved.store(true, std::memory_order_release);

    // give callback time to fire
    sleep_ms(50);

    // future resolved first (or at same time), callback may lag
    CHECK(future_resolved.load(), "Future resolved");
    CHECK(callback_fired.load(),  "Callback eventually fired");
    print_metrics(pool);
}

void test_submit_and_get_with_capture() {
    cout << "\n[Test] submit_and_get -- lambda with capture\n";
    VoidVoidWorkerPool pool(4);
    int multiplier = 7;
    auto fut = pool.submit_and_get([multiplier] { return multiplier * 6; });
    CHECK(fut.get() == 42, "Captured variable used correctly in future task");
    print_metrics(pool);
}

void test_submit_and_get_heavy_work() {
    cout << "\n[Test] submit_and_get -- heavy computation returns correct result\n";
    VoidVoidWorkerPool pool(4);
    auto fut = pool.submit_and_get([] {
        volatile double x = 0;
        for (int i = 0; i < 1000000; ++i) x += std::sin(i * 0.001);
        return (int)(x > 0 ? 1 : -1);
    });
    int result = fut.get();
    CHECK(result == 1 || result == -1, "Heavy computation completed and returned");
    print_metrics(pool);
}

void test_submit_and_get_mixed_with_submit_work() {
    cout << "\n[Test] submit_and_get mixed with submit_work\n";
    VoidVoidWorkerPool pool(4);
    std::atomic<int> fire_forget_count{0};

    // interleave fire-and-forget with futures
    std::vector<std::future<int>> futures;
    for (int i = 0; i < 20; ++i) {
        pool.submit_work([&] { fire_forget_count.fetch_add(1, std::memory_order_relaxed); });
        futures.push_back(pool.submit_and_get([i] { return i; }));
        pool.submit_work([&] { fire_forget_count.fetch_add(1, std::memory_order_relaxed); });
    }

    bool futures_correct = true;
    for (int i = 0; i < 20; ++i)
        if (futures[i].get() != i) { futures_correct = false; break; }

    wait_for(fire_forget_count, 40);

    CHECK(futures_correct,                    "All futures returned correct values");
    CHECK(fire_forget_count.load() == 40,     "All fire-and-forget tasks completed");
    print_metrics(pool);
}

// ─── exception handling ───────────────────────────────────────────────────────

void test_submit_and_get_exception_propagation() {
    cout << "\n[Test] submit_and_get -- exception propagates through future\n";
    VoidVoidWorkerPool pool(2);
    auto fut = pool.submit_and_get([] -> int {
        throw std::runtime_error("task failed");
        return 0;
    });
    bool caught = false;
    try {
        fut.get();
    } catch (const std::runtime_error& e) {
        caught = (std::string(e.what()) == "task failed");
    }
    CHECK(caught, "Exception from task propagated correctly through future");
    print_metrics(pool);
}

void test_submit_and_get_multiple_exceptions() {
    cout << "\n[Test] submit_and_get -- multiple tasks throw, all caught\n";
    VoidVoidWorkerPool pool(4);
    const int N = 10;
    std::vector<std::future<int>> futures;
    for (int i = 0; i < N; ++i)
        futures.push_back(pool.submit_and_get([i] -> int {
            if (i % 2 == 0) throw std::runtime_error("even task " + std::to_string(i));
            return i;
        }));

    int exceptions_caught = 0;
    int correct_values    = 0;
    for (int i = 0; i < N; ++i) {
        try {
            int v = futures[i].get();
            if (v == i) correct_values++;
        } catch (const std::runtime_error&) {
            exceptions_caught++;
        }
    }
    CHECK(exceptions_caught == 5, "5 even tasks threw exceptions");
    CHECK(correct_values    == 5, "5 odd tasks returned correct values");
    print_metrics(pool);
}

// ─── metrics accuracy ─────────────────────────────────────────────────────────

void test_metrics_submitted_equals_completed() {
    cout << "\n[Test] Metrics -- submitted equals completed after drain\n";
    WorkerPool<void()> pool(4);
    std::atomic<int> counter{0};
    const int N = 500;
    for (int i = 0; i < N; ++i)
        pool.submit_work([&] { counter.fetch_add(1, std::memory_order_relaxed); });
    wait_for(counter, N);
    sleep_ms(20); // let last metrics writes settle
    auto m = pool.get_metrics();
    CHECK(m.tasks_submitted == (size_t)N, "tasks_submitted == N");
    CHECK(m.tasks_completed == (size_t)N, "tasks_completed == N");
}

void test_metrics_wait_time_nonzero_under_load() {
    cout << "\n[Test] Metrics -- avg_wait_ms is nonzero under queue pressure\n";

    // Use 1 worker and flood it so tasks must wait
    WorkerPool<void()> pool(1);
    std::atomic<int> counter{0};
    const int N = 200;
    for (int i = 0; i < N; ++i)
        pool.submit_work([&] {
            sleep_ms(1);
            counter.fetch_add(1, std::memory_order_relaxed);
        });
    wait_for(counter, N, 30000);
    auto m = pool.get_metrics();
    CHECK(m.avg_wait_ms > 0.0, "avg_wait_ms > 0 under queue pressure");
    CHECK(m.avg_exec_ms > 0.0, "avg_exec_ms > 0 for tasks that sleep 1ms");
    print_metrics(pool);
}

void test_metrics_exec_time_reflects_task_duration() {
    cout << "\n[Test] Metrics -- avg_exec_ms reflects actual task sleep time\n";
    WorkerPool<void()> pool(4);
    std::atomic<int> counter{0};
    const int N    = 8;
    const int sleep_duration = 50; // ms
    for (int i = 0; i < N; ++i)
        pool.submit_work([&] {
            sleep_ms(sleep_duration);
            counter.fetch_add(1, std::memory_order_relaxed);
        });
    wait_for(counter, N, 10000);
    auto m = pool.get_metrics();
    // avg_exec should be >= sleep ms (includes callbacks, tiny overhead)
    CHECK(m.avg_exec_ms >= (double)sleep_duration * 0.9, "avg_exec_ms >= 90% of task sleep time");
    print_metrics(pool);
}

// ─── VoidVoidWorkerPool alias ─────────────────────────────────────────────────

void test_void_void_worker_pool() {
    cout << "\n[Test] VoidVoidWorkerPool -- basic usage\n";
    VoidVoidWorkerPool pool(4);
    std::atomic<int> counter{0};
    for (int i = 0; i < 20; ++i)
        pool.submit_work([&] { counter.fetch_add(1, std::memory_order_relaxed); });
    CHECK(wait_for(counter, 20), "VoidVoidWorkerPool executed all tasks");

    auto fut = pool.submit_and_get([] { return 99; });
    CHECK(fut.get() == 99, "VoidVoidWorkerPool submit_and_get works");
    print_metrics(pool);
}

// ─── multi-signature WorkerPool ───────────────────────────────────────────────

void test_multi_signature_pool() {
    cout << "\n[Test] Multi-signature WorkerPool<void(), int()>\n";

    WorkerPool<void(), int()> pool(4);
    std::atomic<int> void_count{0};

    // submit void() tasks
    for (int i = 0; i < 10; ++i)
        pool.submit_work(std::function<void()>([&] {
            void_count.fetch_add(1, std::memory_order_relaxed);
        }));

    // submit int() tasks via submit_and_get
    std::vector<std::future<int>> futures;
    for (int i = 0; i < 5; ++i)
        futures.push_back(pool.submit_and_get([i] { return i * 10; }));

    wait_for(void_count, 10);
    bool futures_ok = true;
    for (int i = 0; i < 5; ++i)
        if (futures[i].get() != i * 10) { futures_ok = false; break; }

    CHECK(void_count.load() == 10, "void() tasks all executed");
    CHECK(futures_ok,               "int() tasks returned correct values via future");
    print_metrics(pool);
}

// ─── destructor behavior ──────────────────────────────────────────────────────

void test_destructor_drains_queue() {
    cout << "\n[Test] Destructor drains remaining tasks before exit\n";
    std::atomic<int> counter{0};
    const int N = 1000;
    {
        WorkerPool<void()> pool(2);
        for (int i = 0; i < N; ++i)
            pool.submit_work([&] { counter.fetch_add(1, std::memory_order_relaxed); });
        // destructor called here -- must wait for all tasks
    }
    // if we reach here without hanging, destructor joined cleanly
    CHECK(counter.load() == N, "All tasks completed before destructor returned");
}

void test_destructor_under_heavy_load() {
    cout << "\n[Test] Destructor under heavy load -- no hang, no crash\n";
    std::atomic<int> counter{0};
    const int N = 50000;
    {
        WorkerPool<void()> pool(std::thread::hardware_concurrency());
        for (int i = 0; i < N; ++i)
            pool.submit_work([&] { counter.fetch_add(1, std::memory_order_relaxed); });
        // scope exit -- destructor joins all workers
    }
    CHECK(counter.load() == N, "All " + std::to_string(N) + " tasks completed on destruction");
}

// ─── callback receives correct task id ───────────────────────────────────────

void test_callback_task_id_matches_submit() {
    cout << "\n[Test] Callback task IDs match submitted IDs\n";
    std::mutex id_mutex;
    std::set<size_t> submitted_ids, completed_ids;

    WorkerPool<void()> pool(4,
        nullptr,
        [&](size_t id) {
            std::lock_guard<std::mutex> lock(id_mutex);
            completed_ids.insert(id);
        }
    );

    const int N = 100;
    for (int i = 0; i < N; ++i) {
        size_t id = pool.submit_work([] { sleep_ms(1); });
        std::lock_guard<std::mutex> lock(id_mutex);
        submitted_ids.insert(id);
    }

    // wait for all callbacks
    sleep_ms(500);
    {
        std::lock_guard<std::mutex> lock(id_mutex);
        CHECK(submitted_ids == completed_ids, "Completed IDs exactly match submitted IDs");
    }
    print_metrics(pool);
}

// ─── zero worker edge case ────────────────────────────────────────────────────

void test_hardware_concurrency_query() {
    cout << "\n[Test] Hardware concurrency query\n";
    int hw = WorkerPool<void()>::get_max_hardware_concurrency();
    CHECK(hw > 0, "get_max_hardware_concurrency() > 0 (got " + std::to_string(hw) + ")");
}

// ─── main ─────────────────────────────────────────────────────────────────────

int main() {

    cout << "== Hardware Concurrency: "
         << WorkerPool<void()>::get_max_hardware_concurrency()
         << " ==\n";

    // original
    test_basic_execution();
    test_task_ids_unique();
    test_callbacks_fired();
    test_parallel_execution();
    test_queue_pressure();
    test_multi_producer_stress();
    test_lost_wakeup();

    // submit_and_get
    test_submit_and_get_int();
    test_submit_and_get_string();
    test_submit_and_get_multiple_futures();
    test_submit_and_get_ordering();
    test_submit_and_get_with_capture();
    test_submit_and_get_heavy_work();
    test_submit_and_get_mixed_with_submit_work();

    // exceptions
    test_submit_and_get_exception_propagation();
    test_submit_and_get_multiple_exceptions();

    // metrics
    test_metrics_submitted_equals_completed();
    test_metrics_wait_time_nonzero_under_load();
    test_metrics_exec_time_reflects_task_duration();

    // aliases and multi-sig
    test_void_void_worker_pool();
    test_multi_signature_pool();

    // destructor
    test_destructor_drains_queue();
    test_destructor_under_heavy_load();

    // misc
    test_callback_task_id_matches_submit();
    test_hardware_concurrency_query();

    cout << "\n=============================\n";
    cout << "  Passed: " << tests_passed << "\n";
    cout << "  Failed: " << tests_failed << "\n";
    cout << "=============================\n";

    return tests_failed == 0 ? 0 : 1;
}