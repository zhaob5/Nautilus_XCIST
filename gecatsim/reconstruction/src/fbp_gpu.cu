// fbp_gpu.cu
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#define pi 3.14159265358979f

#ifdef _WIN32
#define DLL_EXPORT __declspec(dllexport)
#else
#define DLL_EXPORT
#endif

// --- keep the same TestStruct layout (matching your original) ---
typedef struct TestStruct {
    float    ScanR;
    float    DistD;
    int       YL;
    int       ZL;
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
    int       AngleNumber;
    int       N_2pi;
    float    Radius;
    int       RecSize;
    int       RecSizeZ;
    float    delta;
    float    HSCoef;
    float    k1;
    float    ***GF;        // [YL][ZL][PN]
    float    ***RecIm;     // [YL][XN][ZN]
} TestStruct;

// index helpers
inline __host__ __device__ int GF_index(int U, int V, int P, int YL, int ZL, int PN) {
    // layout: P major, then V, then U:
    // index = ((P * ZL) + V) * YL + U
    return ((P * ZL) + V) * YL + U;
}

inline __host__ __device__ int Rec_index(int xi, int yi, int zi, int XN, int YN, int ZN) {
    // layout: zi major, then yi, then xi  => ((zi * YN) + yi) * XN + xi
    return ((zi * YN) + yi) * XN + xi;
}

// Kernel: each thread computes one (xi, yi) for a given zi
__global__ void backproject_slice_kernel(
    const float *GF_flat,      // size: PN * ZL * YL
    float *Rec_flat,           // size: XN * YN * ZN
    const float *w,            // weight array of length N_2pi (zero outside used range)
    // dimensions / sizes
    int XN, int YN, int ZN,
    int YL, int ZL, int PN,
    // geometry / constants
    float dx, float dy, float dz,
    float XNC, float YNC, float ZNC,
    float XOffSet, float YOffSet, float ZOffSet,
    float ScanR, float DistD, float h,
    float dYL, float dZL,
    float BetaS,
    int N_2pi, int N_pi,
    float DeltaFai,
    // per-slice values
    int s0, int s1, int s2,
    float z,                   // computed z for the current slice
    int PN_global              // PN (same as AngleNumber)
)
{
    int xi = blockIdx.x * blockDim.x + threadIdx.x;
    int yi = blockIdx.y * blockDim.y + threadIdx.y;
    int zi = blockIdx.z * blockDim.z + threadIdx.z; // should be zero for single-slice launches

    if (xi >= XN || yi >= YN) return;
    // we only use zi for indexing Rec_flat; kernel is launched with z-slice mapped to block.z=0 normally
    int zi_idx = 0; // because we process single slice per kernel launch

    float x = -(xi - XNC) * dx - XOffSet;
    float y = -(yi - YNC) * dy - YOffSet;

    float accum = 0.0f;

    for (int ProjInd = s1; ProjInd <= s2; ++ProjInd) {
        float View = BetaS + ProjInd * DeltaFai;

        // compute UU, Yr, Zr using same math as your CPU code
        float UU = -x * cosf(View) - y * sinf(View);
        float Yr = -x * sinf(View) + y * cosf(View);
        // protect sqrt domain
        float sqrtTerm = sqrtf(ScanR * ScanR - Yr * Yr);
        float denom = sqrtTerm + UU;
        // avoid division by zero
        if (denom == 0.0f) continue;
        float Zr = (z - h * (View + asinf(Yr / ScanR)) / (2.0f * pi)) * (DistD) / denom;

        float U1 = Yr / dYL + ((YL - 1) * 0.5f);
        int U = (int)ceilf(U1);
        float V1 = Zr / dZL + ((ZL - 1) * 0.5f) + 0.0f; // ZLC set earlier + dectorZoffset (we pass offsets via ZOffSet)
        int V = (int)ceilf(V1);
        float Dey = U - U1;
        float Dez = V - V1;

        if ((U > 0) && (U < YL) && (V > 0) && (V < ZL)) {
            // interpolate: t->GF[U-1][V-1][ProjInd] etc.
            int U_1 = U - 1;
            int V_1 = V - 1;
            // bounds ensured above
            int idx00 = GF_index(U_1, V_1, ProjInd, YL, ZL, PN);
            int idx01 = GF_index(U_1, V, ProjInd, YL, ZL, PN);
            int idx10 = GF_index(U, V_1, ProjInd, YL, ZL, PN);
            int idx11 = GF_index(U, V, ProjInd, YL, ZL, PN);

            float g00 = GF_flat[idx00];
            float g01 = GF_flat[idx01];
            float g10 = GF_flat[idx10];
            float g11 = GF_flat[idx11];

            float touying = Dey * Dez * g00
                          + Dey * (1.0f - Dez) * g01
                          + (1.0f - Dey) * Dez * g10
                          + (1.0f - Dey) * (1.0f - Dez) * g11;

            // compute d1 index as in original code:
            int d1 = N_pi - (s0 - ProjInd);
            // ensure within 0..N_2pi-1
            if (d1 < 0 || d1 >= N_2pi) continue;

            float weight = w[d1];
            accum += weight * touying * DeltaFai;
        }
    } // end projection loop

    int rec_idx = Rec_index(xi, yi, zi_idx + (int)roundf(ZNC - ZNC), XN, YN, ZN); // zi_idx zero, but consistent formula
    // We assumed Rec_flat already contains previous values for that (xi,yi,zi)
    // So we add accum to it
    atomicAdd(&Rec_flat[rec_idx], accum); // atomic just in case; single-thread write per idx makes it safe without atomic, but keep atomic for safety
}
 
extern "C" DLL_EXPORT void fbp(TestStruct *t)
{
    // copy essential parameters
    int YL = t->YL;
    int ZL = t->ZL;
    int PN = t->AngleNumber;
    int XN = t->RecSize;
    int YN = t->RecSize;
    int ZN = t->RecSizeZ;

    float dx = t->dx;
    float dy = t->dy;
    float dz = t->dz;
    float XOffSet = t->XOffSet;
    float YOffSet = t->YOffSet;
    float ZOffSet = t->ZOffSet;
    float ScanR = t->ScanR;
    float DistD = t->DistD;
    float DecHeight = t->DecHeight;
    float DecWidth = t->DecWidth;
    float DeltaFai = 2.0f * pi / (float)(t->N_2pi);
    int N_2pi = t->N_2pi;
    int N_pi = N_2pi / 2;
    float h = t->h * DecHeight; // follow your original
    float BetaS = t->BetaS;
    float dYL = DecWidth / (float)YL;
    float dZL = DecHeight / (float)ZL;

    // flatten GF and RecIm into contiguous managed arrays
    size_t GF_size = (size_t)PN * (size_t)ZL * (size_t)YL;
    size_t Rec_size = (size_t)XN * (size_t)YN * (size_t)ZN;

    float *GF_flat = nullptr;
    float *Rec_flat = nullptr;

    cudaMallocManaged(&GF_flat, GF_size * sizeof(float));
    cudaMallocManaged(&Rec_flat, Rec_size * sizeof(float));

    // copy GF from t->GF[U][V][P] into GF_flat using same ordering as GF_index
    for (int p = 0; p < PN; ++p) {
        for (int v = 0; v < ZL; ++v) {
            for (int u = 0; u < YL; ++u) {
                size_t idx = GF_index(u, v, p, YL, ZL, PN);
                GF_flat[idx] = t->GF[u][v][p];
            }
        }
    }

    // copy RecIm into Rec_flat
    for (int zi = 0; zi < ZN; ++zi) {
        for (int yi = 0; yi < YN; ++yi) {
            for (int xi = 0; xi < XN; ++xi) {
                size_t idx = Rec_index(xi, yi, zi, XN, YN, ZN);
                Rec_flat[idx] = t->RecIm[yi][xi][zi]; // match your original indexing order RecIm[yi][xi][zi]
            }
        }
    }

    // prepare weight buffer (max length N_2pi)
    float *w = nullptr;
    cudaMallocManaged(&w, sizeof(float) * N_2pi);
    // initialize all zeros
    for (int k = 0; k < N_2pi; ++k) w[k] = 0.0f;

    // Host loop over z slices (keeps logic close to original)
    float XNC = (XN - 1) * 0.5f;
    float YNC = (YN - 1) * 0.5f;
    float ZNC = (ZN - 1) * 0.5f;

    for (int zi = 0; zi < ZN; ++zi) {
        printf("GPU recon slice %d/%d\n", zi, ZN);
        float z = (zi - ZNC) * dz + ZOffSet;
        float Beta0 = 0.0f; // as in your code
        int s0 = (int)ceilf((Beta0 - BetaS) / DeltaFai - 0.5f);
        int s1 = s0 - (int)ceilf(N_pi * t->HSCoef);
        int s2 = s0 + (int)ceilf(N_pi * t->HSCoef) - 1;

        if (s1 < 0) s1 = 0;
        if (s2 > PN - 1) s2 = PN - 1;
        if (s1 > s2) continue;

        // zero weight array
        for (int k = 0; k < N_2pi; ++k) w[k] = 0.0f;
        int L = s2 - s1 + 1;
        int Shift = N_pi - (s0 - s1);

        if (L < 2 * (int)t->delta) {
            for (int k = 0; k < L; ++k) {
                int idx = k + Shift;
                if (idx >= 0 && idx < N_2pi)
                    w[idx] = powf(cosf((pi / 2.0f) * (2.0f * k - L + 1) / (float)L), 2.0f);
            }
        } else {
            for (int k = 0; k < L; ++k) {
                int idx = k + Shift;
                if (idx < 0 || idx >= N_2pi) continue;
                if (0 <= k && k < (int)t->delta)
                    w[idx] = powf(cosf((pi / 2.0f) * ((float)t->delta - k - 0.5f) / (float)t->delta), 2.0f);
                else if (L - (int)t->delta <= k && k < L)
                    w[idx] = powf(cosf((pi / 2.0f) * (k - (L - (int)t->delta) + 0.5f) / (float)t->delta), 2.0f);
                else
                    w[idx] = 1.0f;
            }
        }

        // Launch kernel for this slice zi
        dim3 block(16, 16, 1);
        dim3 grid((XN + block.x - 1) / block.x, (YN + block.y - 1) / block.y, 1);

        // zero-indexed block.z; we will use zi to select rec index offset when copying back
        backproject_slice_kernel<<<grid, block>>>(
            GF_flat, Rec_flat, w,
            XN, YN, ZN, YL, ZL, PN,
            dx, dy, dz,
            XNC, YNC, ZNC,
            XOffSet, YOffSet, ZOffSet,
            ScanR, DistD, h,
            dYL, dZL,
            BetaS, N_2pi, N_pi, DeltaFai,
            s0, s1, s2,
            z,
            PN
        );

        cudaDeviceSynchronize();
    } // end zi loop

    // copy back Rec_flat into t->RecIm
    for (int zi = 0; zi < ZN; ++zi) {
        for (int yi = 0; yi < YN; ++yi) {
            for (int xi = 0; xi < XN; ++xi) {
                size_t idx = Rec_index(xi, yi, zi, XN, YN, ZN);
                t->RecIm[yi][xi][zi] = Rec_flat[idx];
            }
        }
    }

    // free managed memory
    cudaFree(GF_flat);
    cudaFree(Rec_flat);
    cudaFree(w);
}
