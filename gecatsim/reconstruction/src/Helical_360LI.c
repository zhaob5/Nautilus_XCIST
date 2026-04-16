#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define PI 3.14159265358979f

typedef struct TestStruct {
    float    ScanR;        // UNUSED (kept for compatibility)
    float    DistD;        // UNUSED
    int      YL;           // number of parallel bins (t direction)
    int      ZL;           // number of detector rows / slices
    float    dectorYoffset;
    float    dectorZoffset;
    float    XOffSet;
    float    YOffSet;
    float    ZOffSet;
    float    phantomXOffSet;
    float    phantomYOffSet;
    float    phantomZOffSet;
    float    DecFanAng;    // UNUSED
    float    DecHeight;
    float    DecWidth;
    float    dx;
    float    dy;
    float    dz;
    float    h;            // normalized pitch (rev distance / detector height)
    float    BetaS;
    float    BetaE;
    int      AngleNumber;  // total number of views
    int      N_2pi;        // views per rotation
    float    Radius;
    int      RecSize;
    int      RecSizeZ;
    float    delta;        // UNUSED
    float    HSCoef;       // UNUSED
    float    k1;           // UNUSED
    float    ***GF;        // [t][z][view]  PARALLEL + FILTERED
    float    ***RecIm;     // [y][x][z]
} TestStruct;


extern void fbp(TestStruct *t)
{
    const int YL = t->YL;
    const int ZL = t->ZL;
    const int PN = t->AngleNumber;
    const int N_2pi = t->N_2pi;

    const int XN = t->RecSize;
    const int YN = t->RecSize;
    const int ZN = t->RecSizeZ;

    const float dx = t->dx;
    const float dy = t->dy;
    const float dz = t->dz;

    const float XOff = t->XOffSet;
    const float YOff = t->YOffSet;
    const float ZOff = t->ZOffSet;

    const float DecHeight = t->DecHeight;
    const float h_mm_per_rev = t->h * DecHeight;

    const float DeltaTheta = 2.0f * PI / (float)N_2pi;
    const float BetaS = t->BetaS;

    const float dT = t->DecWidth / (float)YL;   // SAME DeltaT used in Python
    const float YLCW = (YL - 1) * 0.5f;

    const float XNC = (XN - 1) * 0.5f;
    const float YNC = (YN - 1) * 0.5f;
    const float ZNC = (ZN - 1) * 0.5f;

    int   *p0 = (int*)malloc(sizeof(int) * N_2pi);
    float *tt = (float*)malloc(sizeof(float) * N_2pi);

    for (int zi = 0; zi < ZN; zi++)
    {
        printf("Recon slice %d / %d\n", zi, ZN);

        float z = (zi - ZNC) * dz + ZOff;

        /* --- 360LI z-interpolation index setup --- */
        for (int i = 0; i < N_2pi; i++)
        {
            float beta = BetaS + i * DeltaTheta;
            float z0   = (h_mm_per_rev / (2.0f * PI)) * beta;

            int k = (int)floorf((z - z0) / h_mm_per_rev);
            int p = i + k * N_2pi;

            if (p < 0) p = 0;
            if (p + N_2pi >= PN) p = PN - N_2pi - 1;

            float beta_p = BetaS + p * DeltaTheta;
            float z_p = (h_mm_per_rev / (2.0f * PI)) * beta_p;

            float tfrac = (z - z_p) / h_mm_per_rev;
            if (tfrac < 0.0f) tfrac = 0.0f;
            if (tfrac > 1.0f) tfrac = 1.0f;

            p0[i] = p;
            tt[i] = tfrac;
        }

        /* --- Parallel backprojection --- */
        for (int yi = 0; yi < YN; yi++)
        {
            float y = -(yi - YNC) * dy - YOff;

            for (int xi = 0; xi < XN; xi++)
            {
                float x = -(xi - XNC) * dx - XOff;
                float sum = 0.0f;

                for (int i = 0; i < N_2pi; i++)
                {
                    float theta = BetaS + i * DeltaTheta;

                    /* PARALLEL ray coordinate */
                    float t_mm = x * cosf(theta) + y * sinf(theta);
                    float U1   = t_mm / dT + YLCW;

                    int U = (int)floorf(U1);
                    if (U < 0 || U >= YL - 1) continue;

                    float a = U1 - U;

                    int p = p0[i];
                    float w = tt[i];

                    int V = 0;  // single-row detector //int V = zi;        // Z axis already rebinned
                    if (V < 0 || V >= ZL) continue;

                    float g0 =
                        (1.0f - a) * t->GF[U][V][p] +
                        a           * t->GF[U+1][V][p];

                    float g1 =
                        (1.0f - a) * t->GF[U][V][p + N_2pi] +
                        a           * t->GF[U+1][V][p + N_2pi];

                    sum += ((1.0f - w) * g0 + w * g1);
                }

                t->RecIm[yi][xi][zi] += sum * DeltaTheta;
            }
        }
    }

    free(p0);
    free(tt);
}
