#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define PI 3.14159265358979f

#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t err = (call);                                           \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error %s at %s:%d\n",                     \
                    cudaGetErrorString(err), __FILE__, __LINE__);           \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    } while (0)

typedef struct TestStruct {
    float ScanR;
    float DistD;
    int   YL;
    int   ZL;
    float dectorYoffset;
    float dectorZoffset;
    float XOffSet;
    float YOffSet;
    float ZOffSet;
    float phantomXOffSet;
    float phantomYOffSet;
    float phantomZOffSet;
    float DecFanAng;
    float DecHeight;
    float DecWidth;
    float dx;
    float dy;
    float dz;
    float h;
    float BetaS;
    float BetaE;
    int   AngleNumber;
    int   N_2pi;
    float Radius;
    int   RecSize;
    int   RecSizeZ;
    float delta;
    float HSCoef;
    float k1;

    // NOTE:
    // These triple pointers are NOT used directly on GPU.
    // Before calling the CUDA routine, flatten them into contiguous arrays.
    float*** GF;
    float*** RecIm;
    
    // NEW: per-view geometry
    float* SIDArray;
    float* SDDArray;
    float* ViewAngleArray;    
} TestStruct;

// -------------------------
// Index helpers for flat arrays
// -------------------------
__host__ __device__ inline int idxGF(int u, int v, int p, int ZL, int PN)
{
    // GF[U][V][ProjInd]
    return (u * ZL + v) * PN + p;
}

__host__ __device__ inline int idxRec(int y, int x, int z, int XN, int ZN)
{
    // RecIm[yi][xi][zi]
    return (y * XN + x) * ZN + z;
}

// -------------------------
// Kernel 1: backprojection
// one thread = one voxel (xi, yi, zi)
// -------------------------
__global__ void backprojectKernel(
    const float* GF,
    float* RecIm,
    const float* SIDArray,
    const float* SDDArray,
    const float* ViewAngleArray,
    int   YL,
    int   ZL,
    float dectorZoffset,
    float XOffSet,
    float YOffSet,
    float ZOffSet,
    float DecFanAng,
    float DecHeight,
    float DecWidth,
    float dx,
    float dy,
    float dz,
    float h_norm,
    int   PN,
    int   RecSize,
    int   RecSizeZ)
{
    int xi = blockIdx.x * blockDim.x + threadIdx.x;
    int yi = blockIdx.y * blockDim.y + threadIdx.y;
    int zi = blockIdx.z * blockDim.z + threadIdx.z;

    int XN = RecSize;
    int YN = RecSize;
    int ZN = RecSizeZ;

    if (xi >= XN || yi >= YN || zi >= ZN) return;

    float XNC = 0.5f * (XN - 1);
    float YNC = 0.5f * (YN - 1);
    float ZNC = 0.5f * (ZN - 1);

    float h = h_norm * DecHeight;
    float dDecL = DecFanAng / YL;
    float dYL = DecWidth / YL;
    float dZL = DecHeight / ZL;
    // float DeltaFai = 2.0f * PI / N_2pi;
    float YLC = 0.5f * (YL - 1);
    float ZLC = 0.5f * (ZL - 1) + dectorZoffset;

    float x = -(xi - XNC) * dx - XOffSet;
    float y = -(yi - YNC) * dy - YOffSet;
    float z = (zi - ZNC) * dz + ZOffSet;

    float accum = 0.0f;

    int s1 = 0;
    int s2 = PN - 1;

    for (int ProjInd = s1; ProjInd <= s2; ProjInd++)
    {
        float View  = ViewAngleArray[ProjInd];
        float ScanR = SIDArray[ProjInd];
        float DistD = SDDArray[ProjInd];
    
        float c = cosf(View);
        float s = sinf(View);
    
        float UU = -x * c - y * s;
        float Yr = -x * s + y * c;
    
        float ratio = Yr / ScanR;
        ratio = fminf(1.0f, fmaxf(-1.0f, ratio));
    
        float denom_sqrt = ScanR * ScanR - Yr * Yr;
        if (denom_sqrt <= 0.0f) continue;
    
        float denom = sqrtf(denom_sqrt) + UU;
        if (fabsf(denom) < 1e-8f) continue;
    
        float Zr = (z - h * (View + asinf(ratio)) / (2.0f * PI)) * DistD / denom;

        float U1 = Yr / dYL + YLC;
        float V1 = Zr / dZL + ZLC;

        int U = (int)ceilf(U1);
        int V = (int)ceilf(V1);

        float Dey = U - U1;
        // float Dez = V - V1;   // not used in your current interpolation

        // int idx_d = (int)ceilf(-UU / dYL + YLC);
        // if (idx_d < 0) idx_d = 0;
        // if (idx_d >= YL) idx_d = YL - 1;

        // float weight1 = w[idxW(ProjInd, idx_d, YL)];
        // (void)weight1; // weighting already applied outside in your note

        // same interpolation logic as your current code:
        // projection = Dey * GF[U-1][V][ProjInd] + (1-Dey) * GF[U][V][ProjInd];
        if (U > 0 && U < YL && V > 0 && V < ZL)
        {
            float p0 = GF[idxGF(U - 1, V, ProjInd, ZL, PN)];
            float p1 = GF[idxGF(U,     V, ProjInd, ZL, PN)];
            float projection = Dey * p0 + (1.0f - Dey) * p1;

            float dView;
            if (ProjInd < PN - 1)
                dView = ViewAngleArray[ProjInd + 1] - ViewAngleArray[ProjInd];
            else
                dView = ViewAngleArray[ProjInd] - ViewAngleArray[ProjInd - 1];
            
            accum += projection; //* dView;
            // If later you want weighting inside:
            // accum += weight1 * projection * DeltaFai;
        }
    }

    RecIm[idxRec(yi, xi, zi, XN, ZN)] += accum;
}

// -------------------------
// Helper: flatten GF from float*** to contiguous float*
// Expected original layout: GF[YL][ZL][PN]
// -------------------------
float* flattenGF(float*** GF, int YL, int ZL, int PN)
{
    size_t total = (size_t)YL * ZL * PN;
    float* out = (float*)malloc(total * sizeof(float));
    if (!out) return NULL;

    for (int u = 0; u < YL; u++)
    {
        for (int v = 0; v < ZL; v++)
        {
            for (int p = 0; p < PN; p++)
            {
                out[idxGF(u, v, p, ZL, PN)] = GF[u][v][p];
            }
        }
    }
    return out;
}

// -------------------------
// Helper: flatten RecIm from float*** to contiguous float*
// Expected original layout: RecIm[YN][XN][ZN], with YN = XN = RecSize
// -------------------------
float* flattenRecIm(float*** RecIm, int YN, int XN, int ZN)
{
    size_t total = (size_t)YN * XN * ZN;
    float* out = (float*)malloc(total * sizeof(float));
    if (!out) return NULL;

    for (int y = 0; y < YN; y++)
    {
        for (int x = 0; x < XN; x++)
        {
            for (int z = 0; z < ZN; z++)
            {
                out[idxRec(y, x, z, XN, ZN)] = RecIm[y][x][z];
            }
        }
    }
    return out;
}

// -------------------------
// Helper: copy contiguous RecIm back to float***
// -------------------------
void unflattenRecIm(float*** RecIm, const float* in, int YN, int XN, int ZN)
{
    for (int y = 0; y < YN; y++)
    {
        for (int x = 0; x < XN; x++)
        {
            for (int z = 0; z < ZN; z++)
            {
                RecIm[y][x][z] = in[idxRec(y, x, z, XN, ZN)];
            }
        }
    }
}

// -------------------------
// Main CUDA wrapper
// -------------------------
extern "C" __declspec(dllexport) void fbp(TestStruct* t)
{
    int YL = t->YL;
    int ZL = t->ZL;
    int PN = t->AngleNumber;
    int XN = t->RecSize;
    int YN = t->RecSize;
    int ZN = t->RecSizeZ;
    int N_2pi = t->N_2pi;

    size_t gfCount  = (size_t)YL * ZL * PN;
    size_t recCount = (size_t)YN * XN * ZN;
    // size_t wCount   = (size_t)PN * YL;

    float* h_GF    = flattenGF(t->GF, YL, ZL, PN);
    float* h_RecIm = flattenRecIm(t->RecIm, YN, XN, ZN);

    if (!h_GF || !h_RecIm)
    {
        fprintf(stderr, "Host memory allocation failed.\n");
        exit(EXIT_FAILURE);
    }

    float* d_GF = NULL;
    float* d_RecIm = NULL;
    // float* d_w = NULL;
    
    float* d_SID = NULL;
    float* d_SDD = NULL;
    float* d_ViewAngle = NULL;
    
    CUDA_CHECK(cudaMalloc((void**)&d_GF,    gfCount * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_RecIm, recCount * sizeof(float)));
    // CUDA_CHECK(cudaMalloc((void**)&d_w,     wCount * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_GF, h_GF, gfCount * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_RecIm, h_RecIm, recCount * sizeof(float), cudaMemcpyHostToDevice));
    
    CUDA_CHECK(cudaMalloc((void**)&d_SID,       PN * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_SDD,       PN * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_ViewAngle, PN * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_SID,       t->SIDArray,       PN * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_SDD,       t->SDDArray,       PN * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_ViewAngle, t->ViewAngleArray, PN * sizeof(float), cudaMemcpyHostToDevice));
    
    dim3 block(8, 8, 4);
    dim3 grid((XN + block.x - 1) / block.x,
              (YN + block.y - 1) / block.y,
              (ZN + block.z - 1) / block.z);

    backprojectKernel<<<grid, block>>>(
        d_GF,
        d_RecIm,
        d_SID,
        d_SDD,
        d_ViewAngle,
        t->YL,
        t->ZL,
        t->dectorZoffset,
        t->XOffSet,
        t->YOffSet,
        t->ZOffSet,
        t->DecFanAng,
        t->DecHeight,
        t->DecWidth,
        t->dx,
        t->dy,
        t->dz,
        t->h,
        t->AngleNumber,
        t->RecSize,
        t->RecSizeZ
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_RecIm, d_RecIm, recCount * sizeof(float), cudaMemcpyDeviceToHost));
    unflattenRecIm(t->RecIm, h_RecIm, YN, XN, ZN);

    cudaFree(d_GF);
    cudaFree(d_RecIm);
    // cudaFree(d_w);
    cudaFree(d_SID);
    cudaFree(d_SDD);
    cudaFree(d_ViewAngle);
    
    free(h_GF);
    free(h_RecIm);
}