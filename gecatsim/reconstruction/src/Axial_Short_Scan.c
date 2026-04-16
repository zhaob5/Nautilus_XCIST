#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define pi 3.14159265358979

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
//    int       FOILength;
//    int       FOIWidth;
//    int       FOIHeight;
    float    delta;
    float    HSCoef;
    float    k1;
    float    ***GF;
    float    ***RecIm;
} TestStruct;

extern void fbp(TestStruct *t) {

	float ScanR, DistD,DecL,DecHeight,DecWidth,ObjR,dectorYoffset,dectorZoffset;
	float dx,dy,dz,dYL,dZL,DeltaFai,YLC,ZLC,RadiusSquare,XOffSet,YOffSet,ZOffSet, dDecL;
	float XNC, YNC, ZNC,startangle;
	float h,h1, BetaE, BetaS, delta, HSCoef, k1, Gamma_m, alpha;
	int YL,ZL,PN,RecSize, RecSizeZ, N_2pi,N_pi,XN, YN, ZN;
//	int FOILength,FOIWidth,FOIHeight;

	ScanR = t->ScanR;        /*source object distance*/
    DistD = t->DistD;
	DecL = t->DecFanAng;     /*Project Angle*/
	YL = t->YL;              /*Detection Number */
    ZL = t->ZL;
    DecHeight = t->DecHeight; // Detector length along Z = detectorRowSize * detectorRowsPerMod
    DecWidth = t->DecWidth;
    h1 = t->h; // normalized pitch h = H/L, distance per Rev (mm) / Detector length along Z (mm)
	ObjR = t->Radius;      /*Radius of the field of view*/
	RecSize = t->RecSize;  // RecSize = imageSize /*reconstruction size x*/
	RecSizeZ = t->RecSizeZ;  // RecSizeZ = sliceCount /*reconstruction size x*/
	delta = t->delta;
	HSCoef = t->HSCoef; // % of a full scan
	k1 = t->k1;

    BetaS = t->BetaS; // Start Angle of the Short Scan
    BetaE = t->BetaE; // End Angle of the Short Scan
    N_2pi = t->N_2pi; // viewsPerRotation
    PN = t->AngleNumber; // viewCount, total number of views in scan
    dx = t->dx;   
	dy = t->dy;
    dz = t->dz; // dz = sliceThickness
    dectorYoffset = t->dectorYoffset;
    dectorZoffset = t->dectorZoffset;
//    startangle = t->startangle;
    XOffSet = t->XOffSet;
    YOffSet = t->YOffSet;
    ZOffSet = t->ZOffSet;
    XN = RecSize;
    XNC = (XN-1)*0.5;
    YN = RecSize;
    YNC = (YN-1)*0.5;
    ZN = RecSizeZ; // sliceCount
    ZNC = (ZN-1)*0.5; // slice brfore Mid slice
    h = h1*DecHeight; // h = distance per Rev (mm), normalized pitch * Detector length along Z (mm) = distance per Rev (mm)
	dDecL = DecL/YL;       /*Each move Angle of projection*/
	dZL= DecHeight/(ZL);       /*Each move Angle of projection*/
	YLC    = (YL-1)*0.5;
	ZLC    = (ZL-1)*0.5 + dectorZoffset;

	RadiusSquare= ObjR*ObjR;
	//DeltaFai = 2*pi/N_2pi;
    N_pi = N_2pi/2; // half viewsPerRotation
    
    //Gamma_m = (BetaE - BetaS - pi)/2;
    Gamma_m = DecL/2; // maximum half fan angle

    dYL = DecWidth/YL;
    dZL = DecHeight/ZL; // detector row size along Z
    DeltaFai = 2*pi/N_2pi; // angluar step/increment

    //float *w;
    //w     = (float*)malloc(sizeof(float)*N_2pi);

    // allocate 2D array w[PN][YL] as a contiguous block with row pointers
    int S = PN;            // number of views
    int D = YL;            // number of detector channels

    float w[S][D];

    //float** w = (float**)malloc(sizeof(float*) * S);
    //if (w == NULL) { perror("malloc w"); exit(EXIT_FAILURE); }

    //float* w_data = (float*)malloc(sizeof(float) * S * D);
    //if (w_data == NULL) { perror("malloc w_data"); free(w); exit(EXIT_FAILURE); }

    ///* point rows into contiguous block */
    //for (int i = 0; i < S; ++i) {
    //    w[i] = w_data + i * D;
    //}

    ///* optional: initialize to zero */
    //memset(w_data, 0, sizeof(float) * S * D);

     //////begin of the  main code
     float x,y,z,Dey,Dez, projection,UU,U1,V1,Beta0,Yr,Zr,View,weight,weight1,weight2,Gama,Gama_C,m1,m2,theta, beta;
     int ProjInd,xi,yi,zi,U,V,s0,s1,s2,d1,d2,L,Shift, idx_d;

	 for(zi = 0; zi<ZN; zi++)  // For each z location: // create two slices for testing
	 {
         printf("   recon slice %d/%d...\n", zi, ZN);
		 ///compute the projection position for every grid on the image plane
         z = (zi-ZNC) * dz+ZOffSet; // Distance from current recon plane to center plane = (currentSliceN - slicebeforemidN) * sliceThickness
         //Beta0 = 2 * pi * z / h; // How many Revs from current plane to center plane, in rad
         Beta0 = 0;
         //s0 = ceil((Beta0-BetaS) / DeltaFai-0.5); // Start View, from vertical y-axis
         //s1 = s0-ceil(N_pi*HSCoef);      // min view idx
         //s2 = s0+ceil(N_pi*HSCoef)-1;    // max view idx s2 = ceil((Beta0 + N_Circle * pi + pi ) /DeltaFai);
         
         //s0 = PN/2; // center of rotation (mid projection)
         s1 = 0; // first view index
         s2 = PN;// ceil(N_pi * (1 + DecL / pi));   // last view index

         if ((s1<PN)||(s2>0))
         {
           if (s1 < 0)  {s1 = 0;} // min view idx cannot be less than 0 (must start at y-axis)
           if (s2 > PN-1) {s2 = PN-1;} // max view idx cannot larger than totoal view number
          //////////////////////////////////////////

            for (int s = 0; s < PN; s++)  // PN = total number of views
            {
                beta = BetaS + s * DeltaFai;

                for (int d = 0; d < YL; d++)  // YL = number of detector channels
                {
                    alpha = (YLC-d) * dDecL;  // fan angle of each detector element
                    //float weight = 1.0f;

                    if (0 <= beta && beta <= 2*(Gamma_m - alpha))
                    {
                        //weight = pow(sin(pi/4 * beta/(Gamma_m-alpha)), 2);
                        theta = beta / (2 * (Gamma_m - alpha));
                        weight = 3*pow(theta,2) - 2*pow(theta,3);
                    }
                    else if ((pi-2*alpha) <= beta && beta <= (pi+2*Gamma_m))
                    {
                        //weight = pow(sin(pi/4 * (pi + 2*Gamma_m - beta) / (alpha)), 2);
                        theta = (pi + 2* Gamma_m - beta) / (2 * (Gamma_m + alpha));
                        weight = 3 * pow(theta, 2) - 2 * pow(theta, 3);
                    }
                    else
                    {
                        weight = 1.0;
                    }

                    //w[s * YL + d] = weight;
					w[s][d] = weight;
                }
            }
            //printf("wi = %f", w[0][250]);


         ///////////////////////////////////////////
         for (ProjInd = s1; ProjInd <= s2; ProjInd++ )
         //for (ProjInd = s1; ProjInd <= N_2pi; ProjInd++)
         {
			 View = BetaS + ProjInd * DeltaFai; // Current projection angle
             //d1   = N_pi-(s0-ProjInd); //d1 = ProjInd;
             //if (ProjInd < s0)
             //{
             //    d2 = d1+N_pi;
             //}
             //else //(ProjInd >= s0)
             //{
             //    d2 = d1-N_pi;
             //}

             for(yi=0;yi<YN;yi++)
             {
                 y = -(yi-YNC)*dy-YOffSet;
                 //#pragma omp parallel for private(xi,x, UU, Yr, Zr, U1, U,V1,V, Dey,Dez, projection,weight1,weight2,Gama,Gama_C,m1,m2,weight)
                 for(xi=0;xi<XN;xi++)
                 {
                    x  = -(xi-XNC)*dx-XOffSet; 
                    UU = -x*cos(View)-y*sin(View); // Signed distance from rotation axis to ray passing through (x, y) — corresponds to fan-beam geometry
				    Yr = -x*sin(View)+y*cos(View); // y coordinate on the detector plane
                    Zr = (z-h*(View+asin(Yr/ScanR))/(2.0*pi))*(DistD)/(sqrt(ScanR*ScanR-Yr*Yr)+UU);///03/05/23 Yu
                    //Zr = 0;

                    // maps from world coordinates (x,y,z) to detector coordinates (U,V).
                    U1 = Yr/dYL+YLC;
                    U  = ceil(U1);
                    V1 = Zr/dZL+ZLC;
                    V  = ceil(V1);
                    Dey = U-U1;
                    Dez = V-V1;
                    //Dez = 0;

					//alpha = pi/2 - View - atan2((cos(View)*ScanR-y), (x+sin(View)*ScanR)); // fan angle corresponding to detector position U
                    //idx_d = ceil(-UU/dYL + YLC); // detector index corresponding to alpha
                    //if (idx_d < 0) { idx_d = 0; }
                    //if (idx_d > YL) { idx_d = YL;}
                    //weight1 = w[ProjInd][idx_d];
                    //weight1 = 1;

                    //Linear interploate
                    if ((U>0)&&(U<YL)&&(V>0)&&(V<ZL))
                    {
                        //projection = Dey*Dez*t->GF[U-1][V-1][ProjInd]
                        //            +Dey*(1-Dez)*t->GF[U-1][V][ProjInd]
                        //            +(1-Dey)*Dez*t->GF[U][V-1][ProjInd]
                        //            +(1-Dey)*(1-Dez)*t->GF[U][V][ProjInd];

                        //projection = t->GF[U][V][ProjInd];
                        projection = Dey * t->GF[U - 1][V][ProjInd] + (1 - Dey) * t->GF[U][V][ProjInd];


                          //weight1 = w[d1]; // current
                          //weight2 = w[d2]; // conj
                       
                          // Gama   = fabs((z-h*View/(2.0*pi))/(sqrt(ScanR*ScanR-Yr*Yr)+UU)); // Gamma = tan(alpha)
                          // if (ProjInd < s0)
                          // {
                          //     Gama_C = fabs((z-h*(View+pi)/(2.0*pi))/(sqrt(ScanR*ScanR-Yr*Yr)-UU));
                          // }
                          // else
                          // {
                          //     Gama_C = fabs((z-h*(View-pi)/(2.0*pi))/(sqrt(ScanR*ScanR-Yr*Yr)-UU));
                          // }
                          // m1 = pow(Gama,  k1);    // g(a, p(h)) = tan^(kh)(alpha), current m1     = std::real(std::pow(Gama,k1*h));
                          // m2 = pow(Gama_C, k1);  // g(a_c, p(h)) = tan^(kh)(alpha_c), conj m2     = std::real(std::pow(Gama_C,k1*h));
                           // weight = (weight1*m2)/(weight2*m1+weight1*m2); // Eq.16 in Tang_2006 paper
                           //weight = weight1/(weight2+weight1);
                         

                          //t->RecIm[yi][xi][zi]=t->RecIm[yi][xi][zi] + weight * projection * DeltaFai; // Eq.9 in Tang_2006 paper
                          //t->RecIm[yi][xi][zi] = t->RecIm[yi][xi][zi] + weight1 * projection * DeltaFai;
                          t->RecIm[yi][xi][zi] = t->RecIm[yi][xi][zi] + projection * DeltaFai; // Did Weighting outside already
 
                    }
                 }	//xi
			 }//yi
         }//ProjInd
         //printf("wf = %f", w[0][250]);
         }//  if ((s1<PN)||(s2>0))
	 } //zi
     //////end of the main code
     /* free when done */
     //free(w_data);
     //free(w);
 }
