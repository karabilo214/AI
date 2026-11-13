
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <functional>
#include <mutex>
#include <random>
#include <thread>
#include <vector>
struct Job;
static std::atomic<int> mode{0};          // 0 — без задержек, 1 — сценарий аудита, 2 — засыпание, 3 — случайные
static std::atomic<bool> delay_first{true};
static thread_local std::mt19937 rng(std::random_device{}());
static void sleep_us(int us) { std::this_thread::sleep_for(std::chrono::microseconds(us)); }
static void hook_before_wait() {
    if (mode == 2 && delay_first.exchange(false)) sleep_us(200000);  // рабочий «спит» 200 мс
    if (mode == 3 && rng() % 4 == 0) sleep_us(rng() % 300);
}
static void hook_after_take(Job*) {
    if (mode == 1 && delay_first.exchange(false)) sleep_us(200000);  // проснулся, взял задачу и замер
    if (mode == 3 && rng() % 4 == 0) sleep_us(rng() % 300);
}
struct Job {
    const std::function<void(int, int)>* f;
    int total, chunk;
    std::atomic<int> next{0};
};

struct Pool {
    std::vector<std::thread> th;
    std::mutex m, run_m;
    std::condition_variable cv, cv_done;
    Job* cur = nullptr;
    unsigned gen = 0;
    int nworkers = 0, exited = 0;
    bool stop = false;
    std::atomic<int> failed{0};

    void work(Job& j) {
        for (;;) {
            int a = j.next.fetch_add(j.chunk);
            if (a >= j.total) break;
            int b = a + j.chunk < j.total ? a + j.chunk : j.total;
            try {
                (*j.f)(a, b);
            } catch (...) {  // исключение не должно убить процесс; ошибка — флагом
                failed.store(1);
            }
        }
    }
    void start(int n) {
        nworkers = n > 1 ? n - 1 : 0;
        for (int i = 0; i < nworkers; i++)
            th.emplace_back([this] {
                unsigned seen = 0;
                for (;;) {
                    hook_before_wait();
                    Job* j;
                    {
                        std::unique_lock<std::mutex> lk(m);
                        cv.wait(lk, [&] { return stop || gen != seen; });
                        if (stop) return;
                        seen = gen;
                        j = cur;
                    }
                    hook_after_take(j);
                    work(*j);
                    {
                        std::lock_guard<std::mutex> lk(m);
                        if (++exited == nworkers) cv_done.notify_one();
                    }
                }
            });
    }
    void run(int n, int ch, const std::function<void(int, int)>& f) {
        if (n <= 0) return;
        std::lock_guard<std::mutex> rl(run_m);
        Job j;
        j.f = &f; j.total = n; j.chunk = ch > 0 ? ch : 1;
        if (nworkers == 0) { work(j); return; }
        {
            std::lock_guard<std::mutex> lk(m);
            cur = &j; exited = 0; ++gen;
        }
        cv.notify_all();
        work(j);
        std::unique_lock<std::mutex> lk(m);
        cv_done.wait(lk, [&] { return exited == nworkers; });  // барьер: все вышли из поколения
        cur = nullptr;
    }
    void shutdown() {
        {
            std::lock_guard<std::mutex> lk(m);
            stop = true;
        }
        cv.notify_all();
        for (auto& t : th) t.join();
        th.clear();
    }
};

static int fails = 0;
static void check(bool ok, const char* what) { if (!ok) { printf("ПРОВАЛ: %s\n", what); fails++; } }

int main() {
    // 1) сценарий аудита: две задачи по одному куску; рабочий задерживается внутри первой
    for (int rep = 0; rep < 3; rep++) {
        Pool p; p.start(2);
        mode = 1; delay_first = true;
        std::atomic<int> old_calls{0}, new_calls{0};
        p.run(1, 1, [&](int, int) { ++old_calls; });
        p.run(1, 1, [&](int, int) { ++new_calls; });
        p.shutdown();
        printf("сценарий аудита: old_calls=%d new_calls=%d (нужно 1 и 1)\n", old_calls.load(), new_calls.load());
        check(old_calls == 1 && new_calls == 1, "сценарий аудита");
    }
    // 2) рабочий засыпает до ожидания поколения: run() не должен вернуться, пока он не вышел из поколения
    {
        Pool p; p.start(3);
        mode = 2; delay_first = true;
        std::atomic<int> after{0};
        std::atomic<bool> returned{false};
        for (int r = 0; r < 5; r++) {
            returned = false;
            p.run(64, 1, [&](int, int) { if (returned) ++after; });
            returned = true;
        }
        p.shutdown();
        printf("засыпание: вызовов после возврата run(): %d (нужно 0)\n", after.load());
        check(after == 0, "вызов после возврата");
    }
    // 3) случайная нагрузка: каждая строка ровно один раз, никаких вызовов после возврата
    {
        Pool p; p.start(8);
        mode = 3;
        std::mt19937 g(1);
        std::vector<std::atomic<int>> hit(4096);
        int bad = 0, late = 0;
        for (int it = 0; it < 20000; it++) {
            int n = 1 + g() % 4096, ch = 1 + g() % 64;
            for (int i = 0; i < n; i++) hit[i] = 0;
            std::atomic<bool> returned{false};
            std::atomic<int> late_here{0};
            p.run(n, ch, [&](int a, int b) {
                if (returned) ++late_here;
                for (int i = a; i < b; i++) hit[i].fetch_add(1);
            });
            returned = true;
            for (int i = 0; i < n; i++) bad += hit[i] != 1;
            late += late_here;
        }
        p.shutdown();
        printf("нагрузка: 20000 задач, строк обработано не ровно один раз: %d, вызовов после возврата: %d\n", bad, late);
        check(bad == 0 && late == 0, "нагрузка");
    }
    printf(fails ? "ИТОГ: ПРОВАЛ\n" : "ИТОГ: OK\n");
    return fails ? 1 : 0;
}
