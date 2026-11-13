// Умножение на веса GGUF (Q4_K, Q6_K, F16) на процессоре: Y[t] (+)= W X[t], t < T, счёт fp32 (как в видеокарте,
// чтобы шаг на процессоре и проверка пачки в видеокарте давали одинаковые числа). Блок из 256 весов строки
// распаковывается один раз в буфер (в кэше процессора) и умножается на все слова пачки; AVX-512, собственный пул по строкам.
// Собирается для Windows кросс-компилятором: x86_64-w64-mingw32-g++ -O3 -march=x86-64-v4 -mf16c -shared -static
// Потоки — свой постоянный пул с барьером выхода из поколения (см. struct Pool).
#include <immintrin.h>
#include <stdint.h>
#include <string.h>

#include <algorithm>
#include <atomic>
#include <math.h>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

#ifdef _WIN32
#define EXPORT extern "C" __declspec(dllexport)
#else
#define EXPORT extern "C" __attribute__((visibility("default")))
#endif
EXPORT int qmm_abi_version() { return 2; }

// ---- постоянный пул потоков ----
// У каждого вызова run() свой объект задачи (Job) в стеке run(): свой счётчик кусков, своя функция.
// Рабочий читает указатель на текущую задачу только под замком и только при смене поколения.
// run() возвращается лишь после того, как КАЖДЫЙ рабочий подтвердил выход из этого поколения (барьер),
// даже если ему не досталось ни одного куска. Поэтому следующее поколение публикуется только когда все
// вышли из прошлого: рабочий не может увидеть устаревшую задачу, пропустить поколение или обратиться к
// уже уничтоженному Job. Внешние вызовы run() — по очереди (run_m).
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
                    Job* j;
                    {
                        std::unique_lock<std::mutex> lk(m);
                        cv.wait(lk, [&] { return stop || gen != seen; });
                        if (stop) return;
                        seen = gen;
                        j = cur;
                    }
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
static std::atomic<int> nthreads{8};
static std::once_flag pool_once;
static Pool* pool = nullptr;
static Pool& P() {  // создаётся один раз, потокобезопасно
    std::call_once(pool_once, [] { pool = new Pool(); pool->start(nthreads.load()); });
    return *pool;
}

static inline float h2f(uint16_t h) { return _cvtsh_ss(h); }

static inline float hsum(__m512 v) { return _mm512_reduce_add_ps(v); }

// dot 256 весов (буфер w) с T строками X (шаг K), к acc[t]
static inline void dot256(const float* w, const float* X, int K, int k0, int T, float* acc) {
    for (int t = 0; t < T; t++) {
        const float* x = X + (size_t)t * K + k0;
        __m512 s0 = _mm512_setzero_ps(), s1 = _mm512_setzero_ps(), s2 = _mm512_setzero_ps(), s3 = _mm512_setzero_ps();
        for (int i = 0; i < 256; i += 64) {
            s0 = _mm512_fmadd_ps(_mm512_loadu_ps(w + i), _mm512_loadu_ps(x + i), s0);
            s1 = _mm512_fmadd_ps(_mm512_loadu_ps(w + i + 16), _mm512_loadu_ps(x + i + 16), s1);
            s2 = _mm512_fmadd_ps(_mm512_loadu_ps(w + i + 32), _mm512_loadu_ps(x + i + 32), s2);
            s3 = _mm512_fmadd_ps(_mm512_loadu_ps(w + i + 48), _mm512_loadu_ps(x + i + 48), s3);
        }
        acc[t] += hsum(_mm512_add_ps(_mm512_add_ps(s0, s1), _mm512_add_ps(s2, s3)));
    }
}

static inline void store(float* Y, int ys, int r, int T, const float* acc, int add) {
    for (int t = 0; t < T; t++) {
        float* y = Y + (size_t)t * ys + r;
        *y = add ? *y + acc[t] : acc[t];
    }
}

// Q4_K: блок 144 байта: d, dmin (fp16), scales[12] (6-битные масштаб/сдвиг 8 подблоков), qs[128]
static inline void q4k_decode(const uint8_t* b, float* w) {
    float d = h2f(*(const uint16_t*)b), dm = h2f(*(const uint16_t*)(b + 2));
    const uint8_t* s = b + 4;
    const uint8_t* q = b + 16;
    for (int j = 0; j < 4; j++) {
        int i0 = 2 * j, i1 = 2 * j + 1;
        int sc0, m0, sc1, m1;
        if (i0 < 4) { sc0 = s[i0] & 63; m0 = s[i0 + 4] & 63; }
        else { sc0 = (s[i0 + 4] & 15) | ((s[i0 - 4] >> 6) << 4); m0 = (s[i0 + 4] >> 4) | ((s[i0] >> 6) << 4); }
        if (i1 < 4) { sc1 = s[i1] & 63; m1 = s[i1 + 4] & 63; }
        else { sc1 = (s[i1 + 4] & 15) | ((s[i1 - 4] >> 6) << 4); m1 = (s[i1 + 4] >> 4) | ((s[i1] >> 6) << 4); }
        __m512 a0 = _mm512_set1_ps(d * sc0), b0 = _mm512_set1_ps(dm * m0);
        __m512 a1 = _mm512_set1_ps(d * sc1), b1 = _mm512_set1_ps(dm * m1);
        for (int l = 0; l < 32; l += 16) {
            __m128i qb = _mm_loadu_si128((const __m128i*)(q + j * 32 + l));
            __m512i qi = _mm512_cvtepu8_epi32(qb);
            __m512 lo = _mm512_cvtepi32_ps(_mm512_and_si512(qi, _mm512_set1_epi32(15)));
            __m512 hi = _mm512_cvtepi32_ps(_mm512_srli_epi32(qi, 4));
            _mm512_storeu_ps(w + j * 64 + l, _mm512_fmsub_ps(a0, lo, b0));
            _mm512_storeu_ps(w + j * 64 + 32 + l, _mm512_fmsub_ps(a1, hi, b1));
        }
    }
}

// Q6_K: блок 210 байт: ql[128], qh[64], scales[16] (int8), d (fp16). Векторно по 16 чисел.
static inline void q6k_decode(const uint8_t* b, float* w) {
    const uint8_t* ql = b;
    const uint8_t* qh = b + 128;
    const int8_t* sc = (const int8_t*)(b + 192);
    float d = h2f(*(const uint16_t*)(b + 208));
    const __m512i m15 = _mm512_set1_epi32(15), m3 = _mm512_set1_epi32(3), c32 = _mm512_set1_epi32(32);
    for (int h = 0; h < 2; h++) {
        const int8_t* s = sc + h * 8;
        for (int l0 = 0; l0 < 32; l0 += 16) {
            int is = l0 / 16;
            __m512i L0 = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*)(ql + h * 64 + l0)));
            __m512i L1 = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*)(ql + h * 64 + 32 + l0)));
            __m512i H = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*)(qh + h * 32 + l0)));
            __m512i q1 = _mm512_sub_epi32(_mm512_or_si512(_mm512_and_si512(L0, m15), _mm512_slli_epi32(_mm512_and_si512(H, m3), 4)), c32);
            __m512i q2 = _mm512_sub_epi32(_mm512_or_si512(_mm512_and_si512(L1, m15), _mm512_slli_epi32(_mm512_and_si512(_mm512_srli_epi32(H, 2), m3), 4)), c32);
            __m512i q3 = _mm512_sub_epi32(_mm512_or_si512(_mm512_srli_epi32(L0, 4), _mm512_slli_epi32(_mm512_and_si512(_mm512_srli_epi32(H, 4), m3), 4)), c32);
            __m512i q4 = _mm512_sub_epi32(_mm512_or_si512(_mm512_srli_epi32(L1, 4), _mm512_slli_epi32(_mm512_and_si512(_mm512_srli_epi32(H, 6), m3), 4)), c32);
            float* o = w + h * 128 + l0;
            _mm512_storeu_ps(o, _mm512_mul_ps(_mm512_set1_ps(d * s[is]), _mm512_cvtepi32_ps(q1)));
            _mm512_storeu_ps(o + 32, _mm512_mul_ps(_mm512_set1_ps(d * s[is + 2]), _mm512_cvtepi32_ps(q2)));
            _mm512_storeu_ps(o + 64, _mm512_mul_ps(_mm512_set1_ps(d * s[is + 4]), _mm512_cvtepi32_ps(q3)));
            _mm512_storeu_ps(o + 96, _mm512_mul_ps(_mm512_set1_ps(d * s[is + 6]), _mm512_cvtepi32_ps(q4)));
        }
    }
}

template <int KIND>
static void mm(const uint8_t* W, int N, int K, const float* X, int T, float* Y, int ys, int add) {
    const int nb = K / 256;
    const int bs = KIND == 0 ? 144 : 210;
    P().run(N, 16, [=](int r0, int r1) {
        alignas(64) float w[256];
        for (int r = r0; r < r1; r++) {
            float acc[16] = {0};
            const uint8_t* row = W + (size_t)r * nb * bs;
            for (int b = 0; b < nb; b++) {
                if (KIND == 0) q4k_decode(row + (size_t)b * bs, w);
                else q6k_decode(row + (size_t)b * bs, w);
                dot256(w, X, K, b * 256, T, acc);
            }
            store(Y, ys, r, T, acc, add);
        }
    });
}

// Проверка входа: пачка 1..16 (массивы acc[16]), длина строки кратна 256, размеры положительны. 0 — успех.
static inline int bad_shape(int N, int K, int T, int ys) {
    return !(N > 0 && K > 0 && K % 256 == 0 && T >= 1 && T <= 16 && ys >= N);
}

EXPORT int q4k_mm(const uint8_t* W, int N, int K, const float* X, int T, float* Y, int ys, int add) {
    if (bad_shape(N, K, T, ys) || !W || !X || !Y) return -1;
    mm<0>(W, N, K, X, T, Y, ys, add);
    return 0;
}

EXPORT int q6k_mm(const uint8_t* W, int N, int K, const float* X, int T, float* Y, int ys, int add) {
    if (bad_shape(N, K, T, ys) || !W || !X || !Y) return -1;
    mm<1>(W, N, K, X, T, Y, ys, add);
    return 0;
}

EXPORT int f16_mm(const uint16_t* W, int N, int K, const float* X, int T, float* Y, int ys, int add) {
    if (bad_shape(N, K, T, ys) || !W || !X || !Y) return -1;
    P().run(N, 16, [=](int r0, int r1) {
        alignas(64) float w[256];
        for (int r = r0; r < r1; r++) {
            float acc[16] = {0};
            const uint16_t* row = W + (size_t)r * K;
            for (int k0 = 0; k0 < K; k0 += 256) {
                for (int i = 0; i < 256; i += 16)
                    _mm512_storeu_ps(w + i, _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*)(row + k0 + i))));
                dot256(w, X, K, k0, T, acc);
            }
            store(Y, ys, r, T, acc, add);
        }
    });
    return 0;
}

EXPORT void set_threads(int n) { if (n >= 1 && n <= 256) nthreads = n; }  // действует до первого вызова
EXPORT void pool_shutdown() { if (pool) pool->shutdown(); }
EXPORT int take_failed() { return P().failed.exchange(0); }  // 1 — в рабочем потоке было исключение
EXPORT int get_threads() { return nthreads; }

// ---------------- MoE: эксперты слоя для T слов (k экспертов на слово) ----------------
// G, U: gate/up всех экспертов [E][F строк][D] (Q4_K или Q6_K), Dn: down [E][D строк][F]. ids/wts: [T*k].
// H: рабочий буфер [T*k][F]; Z: выход [T*k][D] (вклад пары «слово-эксперт», уже умноженный на вес).
// Слова группируются по эксперту: строка эксперта распаковывается один раз на все слова группы.
static inline void decode_any(int typ, const uint8_t* b, float* w) {
    if (typ == 0) q4k_decode(b, w); else q6k_decode(b, w);
}
static inline int bsize(int typ) { return typ == 0 ? 144 : 210; }

EXPORT int moe_ffn(const uint8_t* G, const uint8_t* U, const uint8_t* Dn, int tg, int tu, int td, int D, int F,
                   int T, int k, const int* ids, const float* wts, const float* X, float* H, float* Z) {
    if (!(D > 0 && F > 0 && D % 256 == 0 && F % 256 == 0 && T >= 1 && k >= 1 && T * k <= 4096) || !G || !U || !Dn)
        return -1;
    for (int t = 0; t < 3; t++) {
        int ty = t == 0 ? tg : t == 1 ? tu : td;
        if (ty != 0 && ty != 1) return -1;
    }
    const int NP = T * k;
    // Одиночный decode: без сортировки экспертов, групп и цикла токенов.
    // Порядок FMA/hsum оставлен тем же, что у общего пути.
    bool unique = true;
    if (T == 1) {
        int active = 0;
        for (int i = 0; i < k && unique; ++i) {
            if (wts[i] == 0.f) continue;
            ++active;
            for (int j = 0; j < i; ++j)
                if (wts[j] != 0.f && ids[i] == ids[j]) { unique = false; break; }
        }
        if (active == 0) { memset(Z, 0, (size_t)k * D * sizeof(float)); return 0; }
    }
    if (T == 1 && unique) {
        const int RC = 32;
        const int fc = (F + RC - 1) / RC, dc = (D + RC - 1) / RC;
        const int bg = bsize(tg), bu = bsize(tu), bd = bsize(td);
        P().run(k * fc, 1, [&](int a, int b) {
            alignas(64) float wg[256], wu[256];
            for (int it = a; it < b; ++it) {
                const int p = it / fc, r0 = (it % fc) * RC;
                if (wts[p] == 0.f) continue;
                const int e = ids[p];
                for (int r = r0; r < std::min(r0 + RC, F); ++r) {
                    const uint8_t* rg = G + ((size_t)e * F + r) * (D / 256) * bg;
                    const uint8_t* ru = U + ((size_t)e * F + r) * (D / 256) * bu;
                    float ag = 0.f, au = 0.f;
                    for (int bk = 0; bk < D / 256; ++bk) {
                        decode_any(tg, rg + (size_t)bk * bg, wg);
                        decode_any(tu, ru + (size_t)bk * bu, wu);
                        const float* x = X + bk * 256;
                        __m512 sg = _mm512_setzero_ps(), su = _mm512_setzero_ps();
                        for (int i = 0; i < 256; i += 16) {
                            __m512 xv = _mm512_loadu_ps(x + i);
                            sg = _mm512_fmadd_ps(_mm512_loadu_ps(wg + i), xv, sg);
                            su = _mm512_fmadd_ps(_mm512_loadu_ps(wu + i), xv, su);
                        }
                        ag += hsum(sg); au += hsum(su);
                    }
                    H[(size_t)p * F + r] = ag / (1.f + expf(-ag)) * au;
                }
            }
        });
        P().run(k * dc, 1, [&](int a, int b) {
            alignas(64) float wd[256];
            for (int it = a; it < b; ++it) {
                const int p = it / dc, r0 = (it % dc) * RC;
                for (int r = r0; r < std::min(r0 + RC, D); ++r) {
                    if (wts[p] == 0.f) { Z[(size_t)p * D + r] = 0.f; continue; }
                    const uint8_t* rd = Dn + ((size_t)ids[p] * D + r) * (F / 256) * bd;
                    float ad = 0.f;
                    for (int bk = 0; bk < F / 256; ++bk) {
                        decode_any(td, rd + (size_t)bk * bd, wd);
                        const float* h = H + (size_t)p * F + bk * 256;
                        __m512 sum = _mm512_setzero_ps();
                        for (int i = 0; i < 256; i += 16)
                            sum = _mm512_fmadd_ps(_mm512_loadu_ps(wd + i), _mm512_loadu_ps(h + i), sum);
                        ad += hsum(sum);
                    }
                    Z[(size_t)p * D + r] = wts[p] * ad;
                }
            }
        });
        return 0;
    }
    // группы: эксперт -> список пар
    // пары с весом 0 (их эксперт считает видеокарта) пропускаем, их вклад — ноль
    std::vector<int> order;
    for (int i = 0; i < NP; i++) {
        if (wts[i] != 0.f) order.push_back(i);
        else memset(Z + (size_t)i * D, 0, sizeof(float) * D);
    }
    const int NQ = (int)order.size();
    std::sort(order.begin(), order.end(), [&](int a, int b) { return ids[a] < ids[b]; });
    std::vector<int> gs;  // начала групп в order
    for (int i = 0; i < NQ; i++) if (i == 0 || ids[order[i]] != ids[order[i - 1]]) gs.push_back(i);
    gs.push_back(NQ);
    if (NQ == 0) return 0;
    const int ng = (int)gs.size() - 1;
    const size_t rbg = (size_t)(D / 256) * bsize(tg), rbu = (size_t)(D / 256) * bsize(tu), rbd = (size_t)(F / 256) * bsize(td);
    const int RC = 32;  // строк в куске работы
    const int fch = (F + RC - 1) / RC, dch = (D + RC - 1) / RC;
    // 1) gate, up, silu(g)*u -> H
    P().run(ng * fch, 1, [&](int a, int b) {
        alignas(64) float wg[256], wu[256];
        for (int it = a; it < b; it++) {
            int g = it / fch, r0 = (it % fch) * RC, r1 = r0 + RC < F ? r0 + RC : F;
            int e = ids[order[gs[g]]];
            for (int r = r0; r < r1; r++) {
                const uint8_t* rg = G + ((size_t)e * F + r) * rbg;
                const uint8_t* ru = U + ((size_t)e * F + r) * rbu;
                float ag[16], au[16];
                int n = gs[g + 1] - gs[g];
                for (int c0 = 0; c0 < n; c0 += 16) {
                    int cn = n - c0 < 16 ? n - c0 : 16;
                    for (int j = 0; j < cn; j++) ag[j] = au[j] = 0.f;
                    for (int bk = 0; bk < D / 256; bk++) {
                        decode_any(tg, rg + (size_t)bk * bsize(tg), wg);
                        decode_any(tu, ru + (size_t)bk * bsize(tu), wu);
                        for (int j = 0; j < cn; j++) {
                            const float* x = X + (size_t)(order[gs[g] + c0 + j] / k) * D + bk * 256;
                            __m512 sg = _mm512_setzero_ps(), su = _mm512_setzero_ps();
                            for (int i = 0; i < 256; i += 16) {
                                __m512 xv = _mm512_loadu_ps(x + i);
                                sg = _mm512_fmadd_ps(_mm512_loadu_ps(wg + i), xv, sg);
                                su = _mm512_fmadd_ps(_mm512_loadu_ps(wu + i), xv, su);
                            }
                            ag[j] += hsum(sg);
                            au[j] += hsum(su);
                        }
                    }
                    for (int j = 0; j < cn; j++) {
                        float gv = ag[j];
                        H[(size_t)order[gs[g] + c0 + j] * F + r] = gv / (1.f + expf(-gv)) * au[j];
                    }
                }
            }
        }
    });
    // 2) down: Z[p] = w_p * Down_e H[p]
    P().run(ng * dch, 1, [&](int a, int b) {
        alignas(64) float wd[256];
        for (int it = a; it < b; it++) {
            int g = it / dch, r0 = (it % dch) * RC, r1 = r0 + RC < D ? r0 + RC : D;
            int e = ids[order[gs[g]]];
            int n = gs[g + 1] - gs[g];
            for (int r = r0; r < r1; r++) {
                const uint8_t* rd = Dn + ((size_t)e * D + r) * rbd;
                for (int c0 = 0; c0 < n; c0 += 16) {
                    int cn = n - c0 < 16 ? n - c0 : 16;
                    float ad[16] = {0};
                    for (int bk = 0; bk < F / 256; bk++) {
                        decode_any(td, rd + (size_t)bk * bsize(td), wd);
                        for (int j = 0; j < cn; j++) {
                            const float* h = H + (size_t)order[gs[g] + c0 + j] * F + bk * 256;
                            __m512 s = _mm512_setzero_ps();
                            for (int i = 0; i < 256; i += 16) s = _mm512_fmadd_ps(_mm512_loadu_ps(wd + i), _mm512_loadu_ps(h + i), s);
                            ad[j] += hsum(s);
                        }
                    }
                    for (int j = 0; j < cn; j++) {
                        int p = order[gs[g] + c0 + j];
                        Z[(size_t)p * D + r] = wts[p] * ad[j];
                    }
                }
            }
        }
    });
    return 0;
}
