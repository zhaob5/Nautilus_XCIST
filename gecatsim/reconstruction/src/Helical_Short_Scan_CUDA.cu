// Helical_Short_Scan_CUDA.cu
// Build (exe):
//   nvcc -O3 -std=c++14 Helical_Short_Scan_CUDA.cu -o Helical_Short_Scan_CUDA.exe
//
// If you later want a DLL for ctypes, remove main() and add:
//   extern "C" __declspec(dllexport) void fbp(TestStruct* t)

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>

#ifdef _WIN32
  #define EXPORT __declspec(dllexport)
#else
  #define EXPORT
#endif

#define pi 3.14159265358979f

// ------------------- your original struct (unchanged layout) -------------------
typedef struct TestStruct {
    float    ScanR;
    float    DistD;
    int      YL;
    int      ZL;
    float    dectorYoffset;
    float    dectorZoffset;
    float    XOffSet;
    float    YOffSet;
    float    ZOffSet;
    float    phantomXOffSet;
    float    phantomYOffSet;
    float    phantomZOffSet;
    float    DecFanAng;
    float    DecHeight;
    float    DecWidth;
    float    dx;
    float    dy;
    float    dz;
    float    h;
    float    BetaS;
    float    BetaE;
    int      AngleNumber;
    int      N_2pi;
    float    Radius;
    int      RecSize;
    int      RecSizeZ;
    float    delta;
    float    HSCoef;
    float    k1;
    float    ***GF;        // host: [YL][ZL][PN]
    float    ***RecIm;     // host: [YN][XN][ZN] where YN==XN==RecSize
} TestStruct;

// ------------------- error checks -------------------
static inline void ck(cudaError_t e, const char* msg) {
    if (e != cudaSuccess) {
        fprintf(stderr, "CUDA error at %s: %s\n", msg, cudaGetErrorString(e));
        std::exit(1);
    }
}

// ------------------- flatten index helpers -------------------
__host__ __device__ __forceinline__
int idxGF(int U, int V, int p, int ZL, int PN) {
    return (U * ZL + V) * PN + p; // GF[U][V][p]
}

__host__ __device__ __forceinline__
int idxRec(int xi, int yi, int zi, int XN, int ZN) {
    return (yi * XN + xi) * ZN + zi; // RecIm[yi][xi][zi]
}

// ------------------- flatten/unflatten -------------------
// Assumption: GF[u][v] is a contiguous array of length PN.
// Assumption: RecIm[yi][xi] is a contiguous array of length ZN.
static void flatten_GF(const TestStruct* t, float* outGF_flat) {
    int YL = t->YL, ZL = t->ZL, PN = t->AngleNumber;
    for (int u = 0; u < YL; u++) {
        for (int v = 0; v < ZL; v++) {
            const float* src = t->GF[u][v];
            float* dst = outGF_flat + idxGF(u, v, 0, ZL, PN);
            std::memcpy(dst, src, (size_t)PN * sizeof(float));
        }
    }
}

static void flatten_Rec(const TestStruct* t, float* outRec_flat) {
    int XN = t->RecSize;
    int YN = t->RecSize;
    int ZN = t->RecSizeZ;
    for (int yi = 0; yi < YN; yi++) {
        for (int xi = 0; xi < XN; xi++) {
            const float* src = t->RecIm[yi][xi];
            float* dst = outRec_flat + idxRec(xi, yi, 0, XN, ZN);
            std::memcpy(dst, src, (size_t)ZN * sizeof(float));
        }
    }
}

static void unflatten_Rec(TestStruct* t, const float* rec_flat) {
    int XN = t->RecSize;
    int YN = t->RecSize;
    int ZN = t->RecSizeZ;
    for (int yi = 0; yi < YN; yi++) {
        for (int xi = 0; xi < XN; xi++) {
            float* dst = t->RecIm[yi][xi];
            const float* src = rec_flat + idxRec(xi, yi, 0, XN, ZN);
            std::memcpy(dst, src, (size_t)ZN * sizeof(float));
        }
    }
}

// ------------------- GPU kernel: one thread per (xi, yi) for a fixed zi -------------------
__global__ void fbp_slice_kernel(
    const float* __restrict__ GF,   // [YL*ZL*PN]
    float* __restrict__ Rec,        // [YN*XN*ZN]
    const float* __restrict__ w,    // [N_2pi]
    // parameters
    float ScanR, float DistD,
    int YL, int ZL, int PN,
    int XN, int YN, int ZN,
    float dx, float dy, float dz,
    float XNC, float YNC,
    float XOffSet, float YOffSet, float ZOffSet,
    float dYL, float dZL, float YLC, float ZLC,
    float h,
    float BetaS, float DeltaFai,
    int N_2pi, int N_pi,
    float k1,
    float z, int zi,
    int s0, int s1, int s2
){
    int xi = blockIdx.x * blockDim.x + threadIdx.x;
    int yi = blockIdx.y * blockDim.y + threadIdx.y;
    if (xi >= XN || yi >= YN) return;

    float y = -(yi - YNC) * dy - YOffSet;
    float x = -(xi - XNC) * dx - XOffSet;

    float accum = 0.0f;

    for (int ProjInd = s1; ProjInd <= s2; ProjInd++) {
        float View = BetaS + ProjInd * DeltaFai;

        int d1 = N_pi - (s0 - ProjInd);
        int d2 = (ProjInd < s0) ? (d1 + N_pi) : (d1 - N_pi);

        d1 %= N_2pi; if (d1 < 0) d1 += N_2pi;
        d2 %= N_2pi; if (d2 < 0) d2 += N_2pi;

        float c = cosf(View);
        float s = sinf(View);

        float UU = -x*c - y*s;
        float Yr = -x*s + y*c;

        float ScanR2 = ScanR * ScanR;
        float Yr2 = Yr * Yr;
        if (Yr2 >= ScanR2) continue;

        float root = sqrtf(ScanR2 - Yr2);
        float denom = root + UU;
        if (fabsf(denom) < 1e-12f) continue;

        float Zr = (z - h*(View + asinf(Yr/ScanR))/(2.0f*pi)) * (DistD) / denom;

        float U1 = Yr / dYL + YLC;
        int   U  = (int)ceilf(U1);

        float V1 = Zr / dZL + ZLC;
        int   V  = (int)ceilf(V1);

        float Dey = (float)U - U1;
        float Dez = (float)V - V1;

        if ((U > 0) && (U < YL) && (V > 0) && (V < ZL)) {
            float g00 = GF[idxGF(U-1, V-1, ProjInd, ZL, PN)];
            float g01 = GF[idxGF(U-1, V  , ProjInd, ZL, PN)];
            float g10 = GF[idxGF(U  , V-1, ProjInd, ZL, PN)];
            float g11 = GF[idxGF(U  , V  , ProjInd, ZL, PN)];

            float touying =
                Dey*Dez         * g00 +
                Dey*(1.0f-Dez)  * g01 +
                (1.0f-Dey)*Dez  * g10 +
                (1.0f-Dey)*(1.0f-Dez) * g11;

            float weight1 = w[d1];
            float weight2 = w[d2];

            float Gama = fabsf((z - h*View/(2.0f*pi)) / (root + UU));
            float Gama_C = (ProjInd < s0)
                ? fabsf((z - h*(View + pi)/(2.0f*pi)) / (root - UU))
                : fabsf((z - h*(View - pi)/(2.0f*pi)) / (root - UU));

            float m1 = powf(Gama,   k1);
            float m2 = powf(Gama_C, k1);

            float weight = 1.0f; // (weight1*m2) / (weight2*m1 + weight1*m2 + 1e-20f);

            accum += weight * touying * DeltaFai;
        }
    }

    Rec[idxRec(xi, yi, zi, XN, ZN)] += accum;
}

// ------------------- Your fbp() host wrapper (GPU accelerated) -------------------
extern "C" EXPORT void fbp(TestStruct *t)
{
    // ---- unpack parameters (same idea as your C code) ----
    float ScanR, DistD, DecL, DecHeight, DecWidth, ObjR, dectorYoffset, dectorZoffset;
    float dx, dy, dz, dYL, dZL, DeltaFai, YLC, ZLC, RadiusSquare, XOffSet, YOffSet, ZOffSet;
    float XNC, YNC, ZNC;
    float h, h1, BetaS, BetaE, delta, HSCoef, k1;
    int YL, ZL, PN, RecSize, RecSizeZ, N_2pi, N_pi, XN, YN, ZN;

    ScanR = t->ScanR;
    DistD = t->DistD;
    DecL  = t->DecFanAng;
    YL    = t->YL;
    ZL    = t->ZL;
    DecHeight = t->DecHeight;
    DecWidth  = t->DecWidth;
    h1 = t->h;
    ObjR = t->Radius;
    RecSize  = t->RecSize;
    RecSizeZ = t->RecSizeZ;
    delta = t->delta;
    HSCoef = t->HSCoef;
    k1 = t->k1;

    BetaS = t->BetaS;
    BetaE = t->BetaE;
    N_2pi = t->N_2pi;
    PN    = t->AngleNumber;

    dx = t->dx;
    dy = t->dy;
    dz = t->dz;

    dectorYoffset = t->dectorYoffset;
    dectorZoffset = t->dectorZoffset;

    XOffSet = t->XOffSet;
    YOffSet = t->YOffSet;
    ZOffSet = t->ZOffSet;

    // ---- derived constants (same as your C code) ----
    XN = RecSize;
    XNC = (XN - 1) * 0.5f;
    YN = RecSize;
    YNC = (YN - 1) * 0.5f;
    ZN = RecSizeZ;
    ZNC = (ZN - 1) * 0.5f;

    h = h1 * DecHeight;

    dYL = DecL / (float)YL;
    dZL = DecHeight / (float)ZL;
    YLC = (YL - 1) * 0.5f;
    ZLC = (ZL - 1) * 0.5f + dectorZoffset;

    RadiusSquare = ObjR * ObjR; (void)RadiusSquare; // not used later here, but keep

    DeltaFai = 2.0f * pi / (float)N_2pi;
    N_pi = N_2pi / 2;

    // overwritten in your code:
    dYL = DecWidth  / (float)YL;
    dZL = DecHeight / (float)ZL;
    DeltaFai = 2.0f * pi / (float)N_2pi;

    // ---- host w ----
    float* h_w = (float*)malloc((size_t)N_2pi * sizeof(float));
    if (!h_w) { fprintf(stderr, "malloc failed for h_w\n"); return; }

    // ---- flatten host GF & RecIm ----
    size_t nGF  = (size_t)YL * (size_t)ZL * (size_t)PN;
    size_t nRec = (size_t)YN * (size_t)XN * (size_t)ZN;

    float* h_GF_flat  = (float*)malloc(nGF  * sizeof(float));
    float* h_Rec_flat = (float*)malloc(nRec * sizeof(float));
    if (!h_GF_flat || !h_Rec_flat) {
        fprintf(stderr, "malloc failed for flat buffers\n");
        free(h_w);
        free(h_GF_flat);
        free(h_Rec_flat);
        return;
    }

    flatten_GF(t, h_GF_flat);
    flatten_Rec(t, h_Rec_flat);

    // ---- device allocations ----
    float *d_GF = nullptr, *d_Rec = nullptr, *d_w = nullptr;
    ck(cudaMalloc(&d_GF,  nGF  * sizeof(float)), "cudaMalloc d_GF");
    ck(cudaMalloc(&d_Rec, nRec * sizeof(float)), "cudaMalloc d_Rec");
    ck(cudaMalloc(&d_w,   (size_t)N_2pi * sizeof(float)), "cudaMalloc d_w");

    ck(cudaMemcpy(d_GF,  h_GF_flat,  nGF  * sizeof(float), cudaMemcpyHostToDevice), "H2D GF");
    ck(cudaMemcpy(d_Rec, h_Rec_flat, nRec * sizeof(float), cudaMemcpyHostToDevice), "H2D Rec");

    // ---- main loop over slices (keep on CPU, minimal change) ----
    for (int zi = 0; zi < ZN; zi++) {
        printf("   recon slice %d/%d...\n", zi, ZN);

        float z = (zi - ZNC) * dz + ZOffSet;

        float Beta0 = 2.0f * pi * z / h;
        int s0 = (int)ceil((Beta0 - BetaS) / DeltaFai - 0.5);
        int s1 = s0 - (int)ceil(N_pi * HSCoef);
        int s2 = s0 + (int)ceil(N_pi * HSCoef) - 1;

        if ((s1 < PN) || (s2 > 0)) {
            if (s1 < 0) s1 = 0;
            if (s2 > PN - 1) s2 = PN - 1;

            // ---- build weighting function w[] on CPU (same logic) ----
            for (int k = 0; k < N_2pi; k++) h_w[k] = 0.0f;

            int L = s2 - s1 + 1;
            int Shift = N_pi - (s0 - s1);

            if (L < (int)(2.0f * delta)) {
                for (int k = 0; k < L; k++) {
                    float val = cosf((pi/2.0f) * (2.0f*k - L + 1.0f) / (float)L);
                    h_w[k + Shift] = val * val;
                }
            } else {
                for (int k = 0; k < L; k++) {
                    if (0 <= k && k < (int)delta) {
                        float val = cosf((pi/2.0f) * (delta - k - 0.5f) / delta);
                        h_w[k + Shift] = val * val;
                    } else if ((L - (int)delta) <= k && k < L) {
                        float val = cosf((pi/2.0f) * (k - (L - (int)delta) + 0.5f) / delta);
                        h_w[k + Shift] = val * val;
                    } else {
                        h_w[k + Shift] = 1.0f;
                    }
                }
            }

            // copy w to device
            ck(cudaMemcpy(d_w, h_w, (size_t)N_2pi * sizeof(float), cudaMemcpyHostToDevice), "H2D w");

            // launch kernel over (xi, yi) for this slice
            dim3 block(16, 16);
            dim3 grid((XN + block.x - 1) / block.x,
                      (YN + block.y - 1) / block.y);

            fbp_slice_kernel<<<grid, block>>>(
                d_GF, d_Rec, d_w,
                ScanR, DistD,
                YL, ZL, PN,
                XN, YN, ZN,
                dx, dy, dz,
                XNC, YNC,
                XOffSet, YOffSet, ZOffSet,
                dYL, dZL, YLC, ZLC,
                h,
                BetaS, DeltaFai,
                N_2pi, N_pi,
                k1,
                z, zi,
                s0, s1, s2
            );
            ck(cudaGetLastError(), "kernel launch");
            ck(cudaDeviceSynchronize(), "kernel sync");
        }
    }

    // ---- copy back result ----
    ck(cudaMemcpy(h_Rec_flat, d_Rec, nRec * sizeof(float), cudaMemcpyDeviceToHost), "D2H Rec");

    unflatten_Rec(t, h_Rec_flat);

    // ---- cleanup ----
    cudaFree(d_GF);
    cudaFree(d_Rec);
    cudaFree(d_w);

    free(h_w);
    free(h_GF_flat);
    free(h_Rec_flat);
}
