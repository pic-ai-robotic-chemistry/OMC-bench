// mat_mul_sym_kernel.cu
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>
#include <cstdint>

namespace {

inline void cuda_check_last_error(const char* msg) {
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, msg, ": ", cudaGetErrorString(err));
}

__device__ __forceinline__ int idx4(int e, int a, int b, int f, int F) {
    return (((e * 3 + a) * 3 + b) * F + f);
}
__device__ __forceinline__ int idx4_f4(int e, int a, int b, int f4, int F4) {
    return (((e * 3 + a) * 3 + b) * F4 + f4);
}

__device__ __forceinline__ float4 fma4(float4 acc, float4 x, float4 y) {
    acc.x = fmaf(x.x, y.x, acc.x);
    acc.y = fmaf(x.y, y.y, acc.y);
    acc.z = fmaf(x.z, y.z, acc.z);
    acc.w = fmaf(x.w, y.w, acc.w);
    return acc;
}
__device__ __forceinline__ float4 add4(float4 a, float4 b) {
    a.x += b.x; a.y += b.y; a.z += b.z; a.w += b.w;
    return a;
}
__device__ __forceinline__ float4 sub4(float4 a, float4 b) {
    a.x -= b.x; a.y -= b.y; a.z -= b.z; a.w -= b.w;
    return a;
}
__device__ __forceinline__ float4 mul4s(float4 a, float s) {
    a.x *= s; a.y *= s; a.z *= s; a.w *= s;
    return a;
}
__device__ __forceinline__ float4 zero4() { return float4{0.f,0.f,0.f,0.f}; }

// ------------------------------------
// forward (float4)
// out = 0.5*(H@G + (H@G)^T) - tr(H@G)/3 * I
// ------------------------------------
__global__ void mat_mul_sym_fwd_f4(
    const float* __restrict__ h,
    const float* __restrict__ g,
    float* __restrict__ out,
    int E, int F)
{
    int F4 = F >> 2;
    int e  = (int)blockIdx.x;
    int f4 = (int)blockIdx.y * (int)blockDim.x + (int)threadIdx.x;
    if (e >= E || f4 >= F4) return;

    const float4* __restrict__ h4 = reinterpret_cast<const float4*>(h);
    const float4* __restrict__ g4 = reinterpret_cast<const float4*>(g);
    float4* __restrict__ o4       = reinterpret_cast<float4*>(out);

    float4 r00=zero4(), r01=zero4(), r02=zero4();
    float4 r10=zero4(), r11=zero4(), r12=zero4();
    float4 r20=zero4(), r21=zero4(), r22=zero4();

    // k=0
    {
        float4 h00 = h4[idx4_f4(e,0,0,f4,F4)];
        float4 h10 = h4[idx4_f4(e,1,0,f4,F4)];
        float4 h20 = h4[idx4_f4(e,2,0,f4,F4)];

        float4 g00 = g4[idx4_f4(e,0,0,f4,F4)];
        float4 g01 = g4[idx4_f4(e,0,1,f4,F4)];
        float4 g02 = g4[idx4_f4(e,0,2,f4,F4)];

        r00 = fma4(r00, h00, g00); r01 = fma4(r01, h00, g01); r02 = fma4(r02, h00, g02);
        r10 = fma4(r10, h10, g00); r11 = fma4(r11, h10, g01); r12 = fma4(r12, h10, g02);
        r20 = fma4(r20, h20, g00); r21 = fma4(r21, h20, g01); r22 = fma4(r22, h20, g02);
    }
    // k=1
    {
        float4 h01 = h4[idx4_f4(e,0,1,f4,F4)];
        float4 h11 = h4[idx4_f4(e,1,1,f4,F4)];
        float4 h21 = h4[idx4_f4(e,2,1,f4,F4)];

        float4 g10 = g4[idx4_f4(e,1,0,f4,F4)];
        float4 g11 = g4[idx4_f4(e,1,1,f4,F4)];
        float4 g12 = g4[idx4_f4(e,1,2,f4,F4)];

        r00 = fma4(r00, h01, g10); r01 = fma4(r01, h01, g11); r02 = fma4(r02, h01, g12);
        r10 = fma4(r10, h11, g10); r11 = fma4(r11, h11, g11); r12 = fma4(r12, h11, g12);
        r20 = fma4(r20, h21, g10); r21 = fma4(r21, h21, g11); r22 = fma4(r22, h21, g12);
    }
    // k=2
    {
        float4 h02 = h4[idx4_f4(e,0,2,f4,F4)];
        float4 h12 = h4[idx4_f4(e,1,2,f4,F4)];
        float4 h22 = h4[idx4_f4(e,2,2,f4,F4)];

        float4 g20 = g4[idx4_f4(e,2,0,f4,F4)];
        float4 g21 = g4[idx4_f4(e,2,1,f4,F4)];
        float4 g22 = g4[idx4_f4(e,2,2,f4,F4)];

        r00 = fma4(r00, h02, g20); r01 = fma4(r01, h02, g21); r02 = fma4(r02, h02, g22);
        r10 = fma4(r10, h12, g20); r11 = fma4(r11, h12, g21); r12 = fma4(r12, h12, g22);
        r20 = fma4(r20, h22, g20); r21 = fma4(r21, h22, g21); r22 = fma4(r22, h22, g22);
    }

    // sym
    float4 s00 = r00;
    float4 s11 = r11;
    float4 s22 = r22;
    float4 s01 = mul4s(add4(r01, r10), 0.5f);
    float4 s02 = mul4s(add4(r02, r20), 0.5f);
    float4 s12 = mul4s(add4(r12, r21), 0.5f);

    // traceless (note: tr(sym)=tr(raw))
    float4 tr = add4(add4(s00, s11), s22);
    float4 t3 = mul4s(tr, 1.0f/3.0f);
    s00 = sub4(s00, t3);
    s11 = sub4(s11, t3);
    s22 = sub4(s22, t3);

    // write symmetric
    o4[idx4_f4(e,0,0,f4,F4)] = s00;
    o4[idx4_f4(e,0,1,f4,F4)] = s01;
    o4[idx4_f4(e,0,2,f4,F4)] = s02;

    o4[idx4_f4(e,1,0,f4,F4)] = s01;
    o4[idx4_f4(e,1,1,f4,F4)] = s11;
    o4[idx4_f4(e,1,2,f4,F4)] = s12;

    o4[idx4_f4(e,2,0,f4,F4)] = s02;
    o4[idx4_f4(e,2,1,f4,F4)] = s12;
    o4[idx4_f4(e,2,2,f4,F4)] = s22;
}

// ------------------------------------
// backward (float4) - 1st order only
// grad_raw = sym(grad_out) - tr(grad_out)/3 I
// grad_h   = grad_raw @ g^T
// grad_g   = h^T @ grad_raw
// ------------------------------------
__global__ void mat_mul_sym_bwd_f4(
    const float* __restrict__ h,
    const float* __restrict__ g,
    const float* __restrict__ grad_out,
    float* __restrict__ grad_h,
    float* __restrict__ grad_g,
    int E, int F)
{
    int F4 = F >> 2;
    int e  = (int)blockIdx.x;
    int f4 = (int)blockIdx.y * (int)blockDim.x + (int)threadIdx.x;
    if (e >= E || f4 >= F4) return;

    const float4* __restrict__ h4  = reinterpret_cast<const float4*>(h);
    const float4* __restrict__ g4  = reinterpret_cast<const float4*>(g);
    const float4* __restrict__ go4 = reinterpret_cast<const float4*>(grad_out);
    float4* __restrict__ gh4       = reinterpret_cast<float4*>(grad_h);
    float4* __restrict__ gg4       = reinterpret_cast<float4*>(grad_g);

    // load grad_out
    float4 go00 = go4[idx4_f4(e,0,0,f4,F4)];
    float4 go01 = go4[idx4_f4(e,0,1,f4,F4)];
    float4 go02 = go4[idx4_f4(e,0,2,f4,F4)];

    float4 go10 = go4[idx4_f4(e,1,0,f4,F4)];
    float4 go11 = go4[idx4_f4(e,1,1,f4,F4)];
    float4 go12 = go4[idx4_f4(e,1,2,f4,F4)];

    float4 go20 = go4[idx4_f4(e,2,0,f4,F4)];
    float4 go21 = go4[idx4_f4(e,2,1,f4,F4)];
    float4 go22 = go4[idx4_f4(e,2,2,f4,F4)];

    // grad_raw = sym(go) - tr(go)/3 I
    float4 tr = add4(add4(go00, go11), go22);
    float4 t3 = mul4s(tr, 1.0f/3.0f);

    float4 dr00 = sub4(go00, t3);
    float4 dr11 = sub4(go11, t3);
    float4 dr22 = sub4(go22, t3);

    float4 dr01 = mul4s(add4(go01, go10), 0.5f);
    float4 dr02 = mul4s(add4(go02, go20), 0.5f);
    float4 dr12 = mul4s(add4(go12, go21), 0.5f);

    float4 dr10 = dr01;
    float4 dr20 = dr02;
    float4 dr21 = dr12;

    // load h
    float4 h00 = h4[idx4_f4(e,0,0,f4,F4)];
    float4 h01 = h4[idx4_f4(e,0,1,f4,F4)];
    float4 h02 = h4[idx4_f4(e,0,2,f4,F4)];

    float4 h10 = h4[idx4_f4(e,1,0,f4,F4)];
    float4 h11 = h4[idx4_f4(e,1,1,f4,F4)];
    float4 h12 = h4[idx4_f4(e,1,2,f4,F4)];

    float4 h20 = h4[idx4_f4(e,2,0,f4,F4)];
    float4 h21 = h4[idx4_f4(e,2,1,f4,F4)];
    float4 h22 = h4[idx4_f4(e,2,2,f4,F4)];

    // load g
    float4 g00 = g4[idx4_f4(e,0,0,f4,F4)];
    float4 g01 = g4[idx4_f4(e,0,1,f4,F4)];
    float4 g02 = g4[idx4_f4(e,0,2,f4,F4)];

    float4 g10 = g4[idx4_f4(e,1,0,f4,F4)];
    float4 g11 = g4[idx4_f4(e,1,1,f4,F4)];
    float4 g12 = g4[idx4_f4(e,1,2,f4,F4)];

    float4 g20 = g4[idx4_f4(e,2,0,f4,F4)];
    float4 g21 = g4[idx4_f4(e,2,1,f4,F4)];
    float4 g22 = g4[idx4_f4(e,2,2,f4,F4)];

    // grad_h = dr @ g^T
    float4 gh00=zero4(), gh01=zero4(), gh02=zero4();
    float4 gh10=zero4(), gh11=zero4(), gh12=zero4();
    float4 gh20=zero4(), gh21=zero4(), gh22=zero4();

    // k=0 -> use g row0 (g00,g01,g02)
    gh00 = fma4(gh00, dr00, g00); gh00 = fma4(gh00, dr01, g01); gh00 = fma4(gh00, dr02, g02);
    gh10 = fma4(gh10, dr10, g00); gh10 = fma4(gh10, dr11, g01); gh10 = fma4(gh10, dr12, g02);
    gh20 = fma4(gh20, dr20, g00); gh20 = fma4(gh20, dr21, g01); gh20 = fma4(gh20, dr22, g02);

    // k=1 -> use g row1
    gh01 = fma4(gh01, dr00, g10); gh01 = fma4(gh01, dr01, g11); gh01 = fma4(gh01, dr02, g12);
    gh11 = fma4(gh11, dr10, g10); gh11 = fma4(gh11, dr11, g11); gh11 = fma4(gh11, dr12, g12);
    gh21 = fma4(gh21, dr20, g10); gh21 = fma4(gh21, dr21, g11); gh21 = fma4(gh21, dr22, g12);

    // k=2 -> use g row2
    gh02 = fma4(gh02, dr00, g20); gh02 = fma4(gh02, dr01, g21); gh02 = fma4(gh02, dr02, g22);
    gh12 = fma4(gh12, dr10, g20); gh12 = fma4(gh12, dr11, g21); gh12 = fma4(gh12, dr12, g22);
    gh22 = fma4(gh22, dr20, g20); gh22 = fma4(gh22, dr21, g21); gh22 = fma4(gh22, dr22, g22);

    // grad_g = h^T @ dr
    float4 gg00=zero4(), gg01=zero4(), gg02=zero4();
    float4 gg10=zero4(), gg11=zero4(), gg12=zero4();
    float4 gg20=zero4(), gg21=zero4(), gg22=zero4();

    // k=0 -> use h col0: (h00,h10,h20)
    gg00 = fma4(gg00, h00, dr00); gg00 = fma4(gg00, h10, dr10); gg00 = fma4(gg00, h20, dr20);
    gg01 = fma4(gg01, h00, dr01); gg01 = fma4(gg01, h10, dr11); gg01 = fma4(gg01, h20, dr21);
    gg02 = fma4(gg02, h00, dr02); gg02 = fma4(gg02, h10, dr12); gg02 = fma4(gg02, h20, dr22);

    // k=1 -> use h col1
    gg10 = fma4(gg10, h01, dr00); gg10 = fma4(gg10, h11, dr10); gg10 = fma4(gg10, h21, dr20);
    gg11 = fma4(gg11, h01, dr01); gg11 = fma4(gg11, h11, dr11); gg11 = fma4(gg11, h21, dr21);
    gg12 = fma4(gg12, h01, dr02); gg12 = fma4(gg12, h11, dr12); gg12 = fma4(gg12, h21, dr22);

    // k=2 -> use h col2
    gg20 = fma4(gg20, h02, dr00); gg20 = fma4(gg20, h12, dr10); gg20 = fma4(gg20, h22, dr20);
    gg21 = fma4(gg21, h02, dr01); gg21 = fma4(gg21, h12, dr11); gg21 = fma4(gg21, h22, dr21);
    gg22 = fma4(gg22, h02, dr02); gg22 = fma4(gg22, h12, dr12); gg22 = fma4(gg22, h22, dr22);

    // store grad_h
    gh4[idx4_f4(e,0,0,f4,F4)] = gh00;
    gh4[idx4_f4(e,0,1,f4,F4)] = gh01;
    gh4[idx4_f4(e,0,2,f4,F4)] = gh02;

    gh4[idx4_f4(e,1,0,f4,F4)] = gh10;
    gh4[idx4_f4(e,1,1,f4,F4)] = gh11;
    gh4[idx4_f4(e,1,2,f4,F4)] = gh12;

    gh4[idx4_f4(e,2,0,f4,F4)] = gh20;
    gh4[idx4_f4(e,2,1,f4,F4)] = gh21;
    gh4[idx4_f4(e,2,2,f4,F4)] = gh22;

    // store grad_g
    gg4[idx4_f4(e,0,0,f4,F4)] = gg00;
    gg4[idx4_f4(e,0,1,f4,F4)] = gg01;
    gg4[idx4_f4(e,0,2,f4,F4)] = gg02;

    gg4[idx4_f4(e,1,0,f4,F4)] = gg10;
    gg4[idx4_f4(e,1,1,f4,F4)] = gg11;
    gg4[idx4_f4(e,1,2,f4,F4)] = gg12;

    gg4[idx4_f4(e,2,0,f4,F4)] = gg20;
    gg4[idx4_f4(e,2,1,f4,F4)] = gg21;
    gg4[idx4_f4(e,2,2,f4,F4)] = gg22;
}

// ------------------------------------
// scalar fallback forward/backward (any F, no float4)
// each thread handles one feature f, computes 3x3
// ------------------------------------
__global__ void mat_mul_sym_fwd_f1(
    const float* __restrict__ h,
    const float* __restrict__ g,
    float* __restrict__ out,
    int E, int F)
{
    int e = (int)blockIdx.x;
    int f = (int)blockIdx.y * (int)blockDim.x + (int)threadIdx.x;
    if (e >= E || f >= F) return;

    // raw = H@G
    float r00 = 0, r01 = 0, r02 = 0;
    float r10 = 0, r11 = 0, r12 = 0;
    float r20 = 0, r21 = 0, r22 = 0;

    // k=0
    {
        float h00 = h[idx4(e,0,0,f,F)], h10 = h[idx4(e,1,0,f,F)], h20 = h[idx4(e,2,0,f,F)];
        float g00 = g[idx4(e,0,0,f,F)], g01 = g[idx4(e,0,1,f,F)], g02 = g[idx4(e,0,2,f,F)];
        r00 = fmaf(h00, g00, r00); r01 = fmaf(h00, g01, r01); r02 = fmaf(h00, g02, r02);
        r10 = fmaf(h10, g00, r10); r11 = fmaf(h10, g01, r11); r12 = fmaf(h10, g02, r12);
        r20 = fmaf(h20, g00, r20); r21 = fmaf(h20, g01, r21); r22 = fmaf(h20, g02, r22);
    }
    // k=1
    {
        float h01 = h[idx4(e,0,1,f,F)], h11 = h[idx4(e,1,1,f,F)], h21 = h[idx4(e,2,1,f,F)];
        float g10 = g[idx4(e,1,0,f,F)], g11 = g[idx4(e,1,1,f,F)], g12 = g[idx4(e,1,2,f,F)];
        r00 = fmaf(h01, g10, r00); r01 = fmaf(h01, g11, r01); r02 = fmaf(h01, g12, r02);
        r10 = fmaf(h11, g10, r10); r11 = fmaf(h11, g11, r11); r12 = fmaf(h11, g12, r12);
        r20 = fmaf(h21, g10, r20); r21 = fmaf(h21, g11, r21); r22 = fmaf(h21, g12, r22);
    }
    // k=2
    {
        float h02 = h[idx4(e,0,2,f,F)], h12 = h[idx4(e,1,2,f,F)], h22 = h[idx4(e,2,2,f,F)];
        float g20 = g[idx4(e,2,0,f,F)], g21 = g[idx4(e,2,1,f,F)], g22 = g[idx4(e,2,2,f,F)];
        r00 = fmaf(h02, g20, r00); r01 = fmaf(h02, g21, r01); r02 = fmaf(h02, g22, r02);
        r10 = fmaf(h12, g20, r10); r11 = fmaf(h12, g21, r11); r12 = fmaf(h12, g22, r12);
        r20 = fmaf(h22, g20, r20); r21 = fmaf(h22, g21, r21); r22 = fmaf(h22, g22, r22);
    }

    float s00 = r00, s11 = r11, s22 = r22;
    float s01 = 0.5f * (r01 + r10);
    float s02 = 0.5f * (r02 + r20);
    float s12 = 0.5f * (r12 + r21);

    float tr = s00 + s11 + s22;
    float t3 = tr * (1.0f/3.0f);
    s00 -= t3; s11 -= t3; s22 -= t3;

    out[idx4(e,0,0,f,F)] = s00;
    out[idx4(e,0,1,f,F)] = s01;
    out[idx4(e,0,2,f,F)] = s02;

    out[idx4(e,1,0,f,F)] = s01;
    out[idx4(e,1,1,f,F)] = s11;
    out[idx4(e,1,2,f,F)] = s12;

    out[idx4(e,2,0,f,F)] = s02;
    out[idx4(e,2,1,f,F)] = s12;
    out[idx4(e,2,2,f,F)] = s22;
}

__global__ void mat_mul_sym_bwd_f1(
    const float* __restrict__ h,
    const float* __restrict__ g,
    const float* __restrict__ grad_out,
    float* __restrict__ grad_h,
    float* __restrict__ grad_g,
    int E, int F)
{
    int e = (int)blockIdx.x;
    int f = (int)blockIdx.y * (int)blockDim.x + (int)threadIdx.x;
    if (e >= E || f >= F) return;

    float go00 = grad_out[idx4(e,0,0,f,F)];
    float go01 = grad_out[idx4(e,0,1,f,F)];
    float go02 = grad_out[idx4(e,0,2,f,F)];

    float go10 = grad_out[idx4(e,1,0,f,F)];
    float go11 = grad_out[idx4(e,1,1,f,F)];
    float go12 = grad_out[idx4(e,1,2,f,F)];

    float go20 = grad_out[idx4(e,2,0,f,F)];
    float go21 = grad_out[idx4(e,2,1,f,F)];
    float go22 = grad_out[idx4(e,2,2,f,F)];

    float tr = go00 + go11 + go22;
    float t3 = tr * (1.0f/3.0f);

    float dr00 = go00 - t3;
    float dr11 = go11 - t3;
    float dr22 = go22 - t3;

    float dr01 = 0.5f * (go01 + go10);
    float dr02 = 0.5f * (go02 + go20);
    float dr12 = 0.5f * (go12 + go21);

    float dr10 = dr01;
    float dr20 = dr02;
    float dr21 = dr12;

    float g00 = g[idx4(e,0,0,f,F)], g01 = g[idx4(e,0,1,f,F)], g02 = g[idx4(e,0,2,f,F)];
    float g10 = g[idx4(e,1,0,f,F)], g11 = g[idx4(e,1,1,f,F)], g12 = g[idx4(e,1,2,f,F)];
    float g20 = g[idx4(e,2,0,f,F)], g21 = g[idx4(e,2,1,f,F)], g22 = g[idx4(e,2,2,f,F)];

    float h00 = h[idx4(e,0,0,f,F)], h01 = h[idx4(e,0,1,f,F)], h02 = h[idx4(e,0,2,f,F)];
    float h10 = h[idx4(e,1,0,f,F)], h11 = h[idx4(e,1,1,f,F)], h12 = h[idx4(e,1,2,f,F)];
    float h20 = h[idx4(e,2,0,f,F)], h21 = h[idx4(e,2,1,f,F)], h22 = h[idx4(e,2,2,f,F)];

    // grad_h = dr @ g^T
    grad_h[idx4(e,0,0,f,F)] = dr00*g00 + dr01*g01 + dr02*g02;
    grad_h[idx4(e,1,0,f,F)] = dr10*g00 + dr11*g01 + dr12*g02;
    grad_h[idx4(e,2,0,f,F)] = dr20*g00 + dr21*g01 + dr22*g02;

    grad_h[idx4(e,0,1,f,F)] = dr00*g10 + dr01*g11 + dr02*g12;
    grad_h[idx4(e,1,1,f,F)] = dr10*g10 + dr11*g11 + dr12*g12;
    grad_h[idx4(e,2,1,f,F)] = dr20*g10 + dr21*g11 + dr22*g12;

    grad_h[idx4(e,0,2,f,F)] = dr00*g20 + dr01*g21 + dr02*g22;
    grad_h[idx4(e,1,2,f,F)] = dr10*g20 + dr11*g21 + dr12*g22;
    grad_h[idx4(e,2,2,f,F)] = dr20*g20 + dr21*g21 + dr22*g22;

    // grad_g = h^T @ dr
    grad_g[idx4(e,0,0,f,F)] = h00*dr00 + h10*dr10 + h20*dr20;
    grad_g[idx4(e,0,1,f,F)] = h00*dr01 + h10*dr11 + h20*dr21;
    grad_g[idx4(e,0,2,f,F)] = h00*dr02 + h10*dr12 + h20*dr22;

    grad_g[idx4(e,1,0,f,F)] = h01*dr00 + h11*dr10 + h21*dr20;
    grad_g[idx4(e,1,1,f,F)] = h01*dr01 + h11*dr11 + h21*dr21;
    grad_g[idx4(e,1,2,f,F)] = h01*dr02 + h11*dr12 + h21*dr22;

    grad_g[idx4(e,2,0,f,F)] = h02*dr00 + h12*dr10 + h22*dr20;
    grad_g[idx4(e,2,1,f,F)] = h02*dr01 + h12*dr11 + h22*dr21;
    grad_g[idx4(e,2,2,f,F)] = h02*dr02 + h12*dr12 + h22*dr22;
}

} // namespace

torch::Tensor mat_mul_sym_forward(torch::Tensor h, torch::Tensor geom) {
    TORCH_CHECK(h.is_cuda(), "h must be CUDA");
    TORCH_CHECK(geom.is_cuda(), "geom must be CUDA");
    TORCH_CHECK(h.dtype() == torch::kFloat32, "only float32 supported");
    TORCH_CHECK(geom.dtype() == torch::kFloat32, "only float32 supported");
    TORCH_CHECK(h.is_contiguous(), "h must be contiguous");
    TORCH_CHECK(geom.is_contiguous(), "geom must be contiguous");
    TORCH_CHECK(h.dim() == 4 && geom.dim() == 4, "expect [E,3,3,F]");
    TORCH_CHECK(h.size(1) == 3 && h.size(2) == 3, "h must be [E,3,3,F]");
    TORCH_CHECK(geom.size(1) == 3 && geom.size(2) == 3, "geom must be [E,3,3,F]");
    TORCH_CHECK(h.sizes() == geom.sizes(), "h and geom must have same shape");

    int E = (int)h.size(0);
    int F = (int)h.size(3);

    auto out = torch::empty_like(h);

    // choose float4 path if possible
    bool f4_ok = (F % 4 == 0);
    if (f4_ok) {
        const void* hp = h.data_ptr();
        const void* gp = geom.data_ptr();
        void* op = out.data_ptr();
        f4_ok = (((uintptr_t)hp % 16) == 0) && (((uintptr_t)gp % 16) == 0) && (((uintptr_t)op % 16) == 0);
    }

    dim3 block(128);
    if (f4_ok) {
        int F4 = F / 4;
        dim3 grid((unsigned)E, (unsigned)((F4 + block.x - 1) / block.x));
        mat_mul_sym_fwd_f4<<<grid, block>>>(
            h.data_ptr<float>(),
            geom.data_ptr<float>(),
            out.data_ptr<float>(),
            E, F
        );
        cuda_check_last_error("mat_mul_sym_forward (f4) launch failed");
    } else {
        dim3 grid((unsigned)E, (unsigned)((F + block.x - 1) / block.x));
        mat_mul_sym_fwd_f1<<<grid, block>>>(
            h.data_ptr<float>(),
            geom.data_ptr<float>(),
            out.data_ptr<float>(),
            E, F
        );
        cuda_check_last_error("mat_mul_sym_forward (f1) launch failed");
    }

    return out;
}

std::vector<torch::Tensor> mat_mul_sym_backward(torch::Tensor grad_out, torch::Tensor h, torch::Tensor geom) {
    TORCH_CHECK(grad_out.is_cuda(), "grad_out must be CUDA");
    TORCH_CHECK(h.is_cuda() && geom.is_cuda(), "h/geom must be CUDA");
    TORCH_CHECK(grad_out.dtype() == torch::kFloat32, "only float32 supported");
    TORCH_CHECK(h.dtype() == torch::kFloat32 && geom.dtype() == torch::kFloat32, "only float32 supported");
    TORCH_CHECK(grad_out.is_contiguous(), "grad_out must be contiguous");
    TORCH_CHECK(h.is_contiguous() && geom.is_contiguous(), "h/geom must be contiguous");
    TORCH_CHECK(grad_out.sizes() == h.sizes() && h.sizes() == geom.sizes(), "shapes must match");

    int E = (int)h.size(0);
    int F = (int)h.size(3);

    auto grad_h = torch::empty_like(h);
    auto grad_g = torch::empty_like(geom);

    bool f4_ok = (F % 4 == 0);
    if (f4_ok) {
        const void* hp  = h.data_ptr();
        const void* gp  = geom.data_ptr();
        const void* gop = grad_out.data_ptr();
        void* ghp = grad_h.data_ptr();
        void* ggp = grad_g.data_ptr();
        f4_ok = (((uintptr_t)hp % 16) == 0) && (((uintptr_t)gp % 16) == 0) &&
                (((uintptr_t)gop % 16) == 0) && (((uintptr_t)ghp % 16) == 0) &&
                (((uintptr_t)ggp % 16) == 0);
    }

    dim3 block(128);
    if (f4_ok) {
        int F4 = F / 4;
        dim3 grid((unsigned)E, (unsigned)((F4 + block.x - 1) / block.x));
        mat_mul_sym_bwd_f4<<<grid, block>>>(
            h.data_ptr<float>(),
            geom.data_ptr<float>(),
            grad_out.data_ptr<float>(),
            grad_h.data_ptr<float>(),
            grad_g.data_ptr<float>(),
            E, F
        );
        cuda_check_last_error("mat_mul_sym_backward (f4) launch failed");
    } else {
        dim3 grid((unsigned)E, (unsigned)((F + block.x - 1) / block.x));
        mat_mul_sym_bwd_f1<<<grid, block>>>(
            h.data_ptr<float>(),
            geom.data_ptr<float>(),
            grad_out.data_ptr<float>(),
            grad_h.data_ptr<float>(),
            grad_g.data_ptr<float>(),
            E, F
        );
        cuda_check_last_error("mat_mul_sym_backward (f1) launch failed");
    }

    return {grad_h, grad_g};
}
