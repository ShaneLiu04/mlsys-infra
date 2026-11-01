// Re-implementation of the CS149 "fake vector intrinsics" used by asst1 prog2.
//
// Every operation takes a mask; the logger counts one vector instruction per call
// and how many lanes were enabled, so utilization = enabled lanes / (W * instructions).
// The API mirrors the starter (_cs149_vadd_float, _cs149_cntbits, ...).
#pragma once

#include <cmath>
#include <cstdio>

#ifndef VECTOR_WIDTH
#define VECTOR_WIDTH 4
#endif

template <typename T>
struct __cs149_vec {
    T value[VECTOR_WIDTH]{};
};
struct __cs149_mask : __cs149_vec<bool> {};
using __cs149_vec_float = __cs149_vec<float>;
using __cs149_vec_int = __cs149_vec<int>;

struct CS149Logger {
    long long instructions = 0;
    long long enabled_lanes = 0;
    long long total_lanes = 0;
    void reset() { instructions = enabled_lanes = total_lanes = 0; }
    void add(const __cs149_mask& m) {
        ++instructions;
        total_lanes += VECTOR_WIDTH;
        for (int i = 0; i < VECTOR_WIDTH; ++i) enabled_lanes += m.value[i];
    }
    double utilization() const { return total_lanes ? double(enabled_lanes) / total_lanes : 1.0; }
};
inline CS149Logger CS149_LOG;

inline __cs149_mask _cs149_init_ones(int first = VECTOR_WIDTH) {
    __cs149_mask m;
    for (int i = 0; i < VECTOR_WIDTH; ++i) m.value[i] = i < first;
    return m;
}
inline __cs149_mask _cs149_mask_not(const __cs149_mask& a) {
    __cs149_mask m;
    for (int i = 0; i < VECTOR_WIDTH; ++i) m.value[i] = !a.value[i];
    CS149_LOG.add(_cs149_init_ones());
    return m;
}
inline __cs149_mask _cs149_mask_and(const __cs149_mask& a, const __cs149_mask& b) {
    __cs149_mask m;
    for (int i = 0; i < VECTOR_WIDTH; ++i) m.value[i] = a.value[i] && b.value[i];
    CS149_LOG.add(_cs149_init_ones());
    return m;
}
inline __cs149_mask _cs149_mask_or(const __cs149_mask& a, const __cs149_mask& b) {
    __cs149_mask m;
    for (int i = 0; i < VECTOR_WIDTH; ++i) m.value[i] = a.value[i] || b.value[i];
    CS149_LOG.add(_cs149_init_ones());
    return m;
}
inline int _cs149_cntbits(const __cs149_mask& a) {
    int c = 0;
    for (int i = 0; i < VECTOR_WIDTH; ++i) c += a.value[i];
    CS149_LOG.add(_cs149_init_ones());
    return c;
}

#define CS149_LANEWISE(expr)                                \
    for (int i = 0; i < VECTOR_WIDTH; ++i)                  \
        if (mask.value[i]) { expr; }                        \
    CS149_LOG.add(mask)

template <typename T>
inline void _cs149_vset(__cs149_vec<T>& d, T v, const __cs149_mask& mask) { CS149_LANEWISE(d.value[i] = v); }
inline void _cs149_vset_float(__cs149_vec_float& d, float v, const __cs149_mask& m) { _cs149_vset<float>(d, v, m); }
inline void _cs149_vset_int(__cs149_vec_int& d, int v, const __cs149_mask& m) { _cs149_vset<int>(d, v, m); }

template <typename T>
inline void _cs149_vmove(__cs149_vec<T>& d, const __cs149_vec<T>& s, const __cs149_mask& mask) {
    CS149_LANEWISE(d.value[i] = s.value[i]);
}
inline void _cs149_vmove_float(__cs149_vec_float& d, const __cs149_vec_float& s, const __cs149_mask& m) { _cs149_vmove(d, s, m); }
inline void _cs149_vmove_int(__cs149_vec_int& d, const __cs149_vec_int& s, const __cs149_mask& m) { _cs149_vmove(d, s, m); }

template <typename T>
inline void _cs149_vload(__cs149_vec<T>& d, const T* src, const __cs149_mask& mask) { CS149_LANEWISE(d.value[i] = src[i]); }
inline void _cs149_vload_float(__cs149_vec_float& d, const float* s, const __cs149_mask& m) { _cs149_vload(d, s, m); }
inline void _cs149_vload_int(__cs149_vec_int& d, const int* s, const __cs149_mask& m) { _cs149_vload(d, s, m); }

template <typename T>
inline void _cs149_vstore(T* dst, const __cs149_vec<T>& s, const __cs149_mask& mask) { CS149_LANEWISE(dst[i] = s.value[i]); }
inline void _cs149_vstore_float(float* d, const __cs149_vec_float& s, const __cs149_mask& m) { _cs149_vstore(d, s, m); }
inline void _cs149_vstore_int(int* d, const __cs149_vec_int& s, const __cs149_mask& m) { _cs149_vstore(d, s, m); }

#define CS149_BINOP(name, T, op)                                                                   \
    inline void name(__cs149_vec<T>& d, const __cs149_vec<T>& a, const __cs149_vec<T>& b,          \
                     const __cs149_mask& mask) {                                                   \
        CS149_LANEWISE(d.value[i] = a.value[i] op b.value[i]);                                    \
    }
CS149_BINOP(_cs149_vadd_float, float, +)
CS149_BINOP(_cs149_vsub_float, float, -)
CS149_BINOP(_cs149_vmult_float, float, *)
CS149_BINOP(_cs149_vdiv_float, float, /)
CS149_BINOP(_cs149_vadd_int, int, +)
CS149_BINOP(_cs149_vsub_int, int, -)
CS149_BINOP(_cs149_vmult_int, int, *)
CS149_BINOP(_cs149_vdiv_int, int, /)

inline void _cs149_vabs_float(__cs149_vec_float& d, const __cs149_vec_float& a, const __cs149_mask& mask) {
    CS149_LANEWISE(d.value[i] = std::fabs(a.value[i]));
}
inline void _cs149_vabs_int(__cs149_vec_int& d, const __cs149_vec_int& a, const __cs149_mask& mask) {
    CS149_LANEWISE(d.value[i] = a.value[i] < 0 ? -a.value[i] : a.value[i]);
}

#define CS149_CMP(name, T, op)                                                                     \
    inline void name(__cs149_mask& d, const __cs149_vec<T>& a, const __cs149_vec<T>& b,            \
                     const __cs149_mask& mask) {                                                   \
        CS149_LANEWISE(d.value[i] = a.value[i] op b.value[i]);                                    \
    }
CS149_CMP(_cs149_vgt_float, float, >)
CS149_CMP(_cs149_vlt_float, float, <)
CS149_CMP(_cs149_veq_float, float, ==)
CS149_CMP(_cs149_vgt_int, int, >)
CS149_CMP(_cs149_vlt_int, int, <)
CS149_CMP(_cs149_veq_int, int, ==)

// Adjacent-pair add: [a b c d] -> [a+b a+b c+d c+d].
inline void _cs149_hadd_float(__cs149_vec_float& d, const __cs149_vec_float& s) {
    for (int i = 0; i < VECTOR_WIDTH / 2; ++i) {
        float r = s.value[2 * i] + s.value[2 * i + 1];
        d.value[2 * i] = d.value[2 * i + 1] = r;
    }
    CS149_LOG.add(_cs149_init_ones());
}
// Even lanes first, then odd lanes: [a b c d] -> [a c b d].
inline void _cs149_interleave_float(__cs149_vec_float& d, const __cs149_vec_float& s) {
    __cs149_vec_float t;
    for (int i = 0; i < VECTOR_WIDTH; ++i)
        t.value[i] = i < VECTOR_WIDTH / 2 ? s.value[2 * i] : s.value[2 * (i - VECTOR_WIDTH / 2) + 1];
    d = t;
    CS149_LOG.add(_cs149_init_ones());
}
