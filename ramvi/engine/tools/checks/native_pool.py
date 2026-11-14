"""Проверка пула потоков из qmm.cpp на управляемом расписании и под случайной нагрузкой (без AVX/CUDA).
Берётся настоящий код Pool; добавляются только крючки задержки рабочих. Сборка: clang++ (Mac/Linux).
python -m ramvi.engine.tools check native_pool   -> ненулевой код выхода при провале."""
import re
import tempfile
import subprocess
import sys
from pathlib import Path

here = Path(__file__).resolve().parents[2] / "ramvi_engine" / "native"
src = (here / "qmm.cpp").read_text(encoding="utf-8")
pool = src[src.index("struct Job {"):src.index("static std::atomic<int> nthreads")]
# крючки: задержка перед ожиданием поколения и после взятия задачи
pool = pool.replace("unsigned seen = 0;\n                for (;;) {",
                    "unsigned seen = 0;\n                for (;;) {\n                    hook_before_wait();")
pool = pool.replace("j = cur;\n                    }", "j = cur;\n                    }\n                    hook_after_take(j);")
assert "hook_before_wait" in pool and "hook_after_take" in pool, "крючки не вставились — код пула изменился"

pre = r'''
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
'''
main = r'''
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
'''
with tempfile.TemporaryDirectory(prefix="ramvi-pool-") as folder:
    out = Path(folder) / "pool_test.cpp"
    out.write_text(pre + pool + main)
    exe = Path(folder) / "pool_test_bin"
    subprocess.run(["clang++", "-std=c++17", "-O2", "-pthread", "-fsanitize=thread", str(out), "-o", str(exe)], check=True, timeout=120)
    r = subprocess.run([str(exe)], capture_output=True, text=True, timeout=600)
    print(r.stdout + r.stderr[-3000:])
    sys.exit(r.returncode)
