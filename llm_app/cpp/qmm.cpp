// Умножение на веса GGUF (Q4_K, Q6_K, F16) на процессоре: Y[t] (+)= W X[t], t < T, счёт fp32 (как в видеокарте,
// чтобы шаг на процессоре и проверка пачки в видеокарте давали одинаковые числа). Блок из 256 весов строки
// распаковывается один раз в буфер (в кэше процессора) и умножается на все слова пачки; AVX-512, OpenMP по строкам.
// Собирается для Windows кросс-компилятором: x86_64-w64-mingw32-g++ -O3 -march=znver4 -shared -static
// Потоки — свой постоянный пул (строки раздаются кусками через атомарный счётчик).
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

#define EXPORT extern "C" __declspec(dllexport)

// ---- постоянный пул потоков ----
struct Pool {
    std::vector<std::thread> th;
    std::mutex m;
    std::condition_variable cv;
    std::function<void(int, int)> job;  // (начало, конец)
    std::atomic<int> next{0}, done{0};
    int total = 0, chunk = 1, gen = 0, nworkers = 0;
    bool stop = false;

    void work() {
        for (;;) {
            int a = next.fetch_add(chunk);
            if (a >= total) break;
            int b = a + chunk < total ? a + chunk : total;
            job(a, b);
            done.fetch_add(b - a);
        }
    }
    void start(int n) {
        nworkers = n - 1;
        for (int i = 0; i < nworkers; i++)
            th.emplace_back([this] {
                int seen = 0;
                for (;;) {
                    {
                        std::unique_lock<std::mutex> lk(m);
                        cv.wait(lk, [&] { return stop || gen != seen; });
                        if (stop) return;
                        seen = gen;
                    }
                    work();
                }
            });
    }
    void run(int n, int ch, std::function<void(int, int)> f) {
        {
            std::lock_guard<std::mutex> lk(m);
            job = f; total = n; chunk = ch; next = 0; done = 0; gen++;
        }
        cv.notify_all();
        work();
        while (done.load() < total) _mm_pause();
    }
};
static Pool* pool = nullptr;
static int nthreads = 8;
static Pool& P() {
    if (!pool) { pool = new Pool(); pool->start(nthreads); }
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

EXPORT void q4k_mm(const uint8_t* W, int N, int K, const float* X, int T, float* Y, int ys, int add) {
    mm<0>(W, N, K, X, T, Y, ys, add);
}

EXPORT void q6k_mm(const uint8_t* W, int N, int K, const float* X, int T, float* Y, int ys, int add) {
    mm<1>(W, N, K, X, T, Y, ys, add);
}

EXPORT void f16_mm(const uint16_t* W, int N, int K, const float* X, int T, float* Y, int ys, int add) {
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
}

EXPORT void set_threads(int n) { if (!pool) nthreads = n; }  // до первого вызова
EXPORT int get_threads() { return nthreads; }

// ---------------- MoE: эксперты слоя для T слов (k экспертов на слово) ----------------
// G, U: gate/up всех экспертов [E][F строк][D] (Q4_K или Q6_K), Dn: down [E][D строк][F]. ids/wts: [T*k].
// H: рабочий буфер [T*k][F]; Z: выход [T*k][D] (вклад пары «слово-эксперт», уже умноженный на вес).
// Слова группируются по эксперту: строка эксперта распаковывается один раз на все слова группы.
static inline void decode_any(int typ, const uint8_t* b, float* w) {
    if (typ == 0) q4k_decode(b, w); else q6k_decode(b, w);
}
static inline int bsize(int typ) { return typ == 0 ? 144 : 210; }

EXPORT void moe_ffn(const uint8_t* G, const uint8_t* U, const uint8_t* Dn, int tg, int tu, int td, int D, int F,
                    int T, int k, const int* ids, const float* wts, const float* X, float* H, float* Z) {
    const int NP = T * k;
    // группы: эксперт -> список пар
    std::vector<int> order(NP);
    for (int i = 0; i < NP; i++) order[i] = i;
    std::sort(order.begin(), order.end(), [&](int a, int b) { return ids[a] < ids[b]; });
    std::vector<int> gs;  // начала групп в order
    for (int i = 0; i < NP; i++) if (i == 0 || ids[order[i]] != ids[order[i - 1]]) gs.push_back(i);
    gs.push_back(NP);
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
}
