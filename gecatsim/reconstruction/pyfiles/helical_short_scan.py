import ctypes as ct
import numpy as np
import math
from gecatsim.reconstruction.pyfiles.createHSP import createHSP
import scipy.io as scio
import matplotlib.pyplot as plt
from gecatsim.reconstruction.pyfiles.mapConfigVariablesToHelical import mapConfigVariablesToHelical
import os
import time

import gecatsim as xc

# Init ctypes types
FLOAT = ct.c_float
PtrFLOAT = ct.POINTER(FLOAT)
PtrPtrFLOAT = ct.POINTER(PtrFLOAT)
PtrPtrPtrFLOAT = ct.POINTER(PtrPtrFLOAT)
PI =3.14159265358979

class TestStruct(ct.Structure):
    _fields_ = [
                ("ScanR", ct.c_float),
                ("DistD", ct.c_float),
                ("YL", ct.c_int),
                ("ZL", ct.c_int),
                ("dectorYoffset", ct.c_float),  # Detector along the horizontal direction (pixel, e.g. quarter pixel)
                ("dectorZoffset", ct.c_float),  # Detector offset along the vertical direcion (pixel, e.g. quarter pixel)
                ("XOffSet", ct.c_float),  # recon offset along the x axis(mm)
                ("YOffSet", ct.c_float),  # recon offset along the y axis(mm)
                ("ZOffSet", ct.c_float),  # recon offset along the z axis(mm)
                ("phantomXOffSet", ct.c_float),  # phantom offset along the x axis(mm)
                ("phantomYOffSet", ct.c_float),  # phantom offset along the y axis(mm)
                ("phantomZOffSet", ct.c_float),  # phantom offset along the z axis(mm)
                ("DecFanAng", ct.c_float),
                ("DecHeight", ct.c_float),
                ("DecWidth", ct.c_float),
                ("dx", ct.c_float),
                ("dy", ct.c_float),
                ("dz", ct.c_float),
                ("h", ct.c_float),
                ("BetaS", ct.c_float),
                ("BetaE", ct.c_float),
                ("AngleNumber", ct.c_int),
                ("N_2pi", ct.c_int),
                ("Radius", ct.c_float),
                ("RecSize", ct.c_int),
                ("RecSizeZ", ct.c_int),
                ("delta", ct.c_float),
                ("HSCoef", ct.c_float),
                ("k1", ct.c_float),
                ("GF", PtrPtrPtrFLOAT),
                ("RecIm", PtrPtrPtrFLOAT)
                ]


def double3darray2pointer(arr):
    # Converts a 3D numpy to ctypes 3D array.
    arr_dimx = FLOAT * arr.shape[2]
    arr_dimy = PtrFLOAT * arr.shape[1]
    arr_dimz = PtrPtrFLOAT * arr.shape[0]

    arr_ptr = arr_dimz()

    for i, row in enumerate(arr):
        arr_ptr[i] = arr_dimy()
        for j, col in enumerate(row):
            arr_ptr[i][j] = arr_dimx()
            for k, val in enumerate(col):
                arr_ptr[i][j][k] = val
    return arr_ptr


def double3dpointer2array(ptr, n, m, o):
    # Converts ctypes 3D array into a 3D numpy array.
    arr = np.zeros(shape=(n, m, o))

    for i in range(n):
        for j in range(m):
            for k in range(o):
                arr[i, j, k] = ptr[i][j][k]

    return arr


def load_C_recon_lib():
    # add recon lib path to environment value "PATH" for depending DLLs
    # # # # recon_lib = my_path.find_dir("top", os.path.join("reconstruction", "lib"))
    # # # # my_path.add_dir_to_path(recon_lib)

    #  my_path.find_dir doesn't have the key "reconstruction", use the temp solution below:
    recon_lib = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../lib")

    # load C/C++ lib
    ll = ct.cdll.LoadLibrary
    lib_file = "Helical_Short_Scan_CUDA.dll"
    #lib_file = "helicalrecon.dll"
    clib = ll(os.path.join(recon_lib, lib_file))

    return clib

def parkers_weighting(ViewN, YL, BetaS, DeltaFai, YLC, dDecL, Gamma_m):
    """
    Compute Parker's weighting function for short-scan CT reconstruction.
    
    Parameters:
        ViewN (int): total number of short-scan projection views
        YL (int): number of detector channels
        BetaS (float): starting projection angle (radians)
        DeltaFai (float): projection angle increment (radians)
        YLC (float): detector center index
        dDecL (float): detector angular pitch (radians)
        Gamma_virtual (float):  angle exceed 180deg (radians)
    
    Returns:
        w (ndarray): Parker weighting matrix of shape (PN, YL)
    """
    ViewNF = ViewN  # extend views for virtual projections
    w = np.ones((ViewNF, YL), dtype=np.float32)
    pi = np.pi
    #GammaN = int(Gamma_virtual/DeltaFai)

    for s in range(ViewNF):
        beta = BetaS + s * DeltaFai
        for d in range(YL):
            alpha = (YLC - d) * dDecL

            if 0 <= beta < (2 * (Gamma_m - alpha)):              
                #w[s, d] = np.sin(pi / 4 * beta / (Gamma_m - alpha)) ** 2
                theta = beta / (2 * (Gamma_m - alpha))
                w[s, d] = 3*(theta**2) - 2*(theta**3)
            elif (pi - 2 * alpha) <= beta < (pi + 2 * Gamma_m):
                #w[s, d] = np.sin(pi / 4 * (pi + 2 * Gamma_m - beta) / alpha) ** 2
                theta = (pi + 2* Gamma_m - beta) / (2 * (Gamma_m + alpha))
                w[s, d] = 3*(theta**2) - 2*(theta**3)
            else:
                w[s, d] = 1.0
    return w

def silver_weighting(ViewN, YL, BetaS, DeltaFai, YLC, dDecL, Gamma_m):
    """
    Generalized Parker/Silver weighting for arbitrary short-scan (>180° + fan angle).
    """
    pi = np.pi
    ScanRange = ViewN * DeltaFai  # total scan angle coverage
    # Extra angular coverage beyond half-scan
    # Delta = 2Gamma_virtual
    Delta = ScanRange - (pi + 2 * Gamma_m)
    Delta = max(0.0, Delta)  # ensure non-negative

    w = np.ones((ViewN, YL), dtype=np.float32)

    for s in range(ViewN):
        beta = BetaS + s * DeltaFai
        for d in range(YL):
            alpha = (YLC - d) * dDecL
            # alpha = (d - YLC) * dDecL
            
            # Ramp-up region
            beta1 = 0
            beta2 = 2 * (Gamma_m + Delta/2 - alpha)

            # Ramp-down region
            beta3 = pi - 2 * alpha
            beta4 = pi + 2 * Gamma_m + Delta

            if beta1 <= beta < beta2:
                theta = (beta - beta1) / (beta2 - beta1)
                w[s, d] = 3*(theta**2) - 2*(theta**3)   # smooth cubic ramp up
            elif beta3 <= beta < beta4:
                theta = (beta4 - beta) / (beta4 - beta3)
                w[s, d] = 3*(theta**2) - 2*(theta**3)   # smooth cubic ramp down
            # else:
            #     w[s, d] = 1.0
    return w


def helical_short_scan(cfg, prep):
    prep = prep[:,:,::-1]

    if cfg.protocol.rotationDirection == -1:	
        prep = prep[::-1,:,:]
    Proj = prep.transpose(0,2,1) # Sinogram

    # scanner & recon geometry
    SO, DD, YL, ZL, ViewN, N_Turn, N_2pi, h, startAngle, dectorYoffset, dectorZoffset, \
    k1, delta, HSCoef, rowSize, ColSize, imageSize, sliceCount, sliceThickness, centerOffset, ObjR,kernelType \
        = mapConfigVariablesToHelical(cfg)
        
    ####################  IMPORTANT!!!!! "h" above is not table speed!!!! h = tableSpeed * rotationTime = distance per Rev (mm) #####################
    #print("k1 = ", k1)
    #print("h = ", h)
    #print("delta = ", delta)
    
    DecAngle = math.atan(ColSize/2/DD)*2*YL # Fan Angle, DD = source-to-detector distance, YL = total number of detector columns
    #print("DecAngle = ", DecAngle)
    HSCoef = (DecAngle/PI+0.52) # Half Scan Coe. % of Fan Angle in half plus .52?
    # HSCoef = (DecAngle/(2*PI)+0.5) # Correction for the Half Scan Coefficient
    #print("HSCoef", HSCoef)
    #print(SO, DD, YL, ZL, ViewN, N_Turn, N_2pi, h, startAngle, dectorYoffset, dectorZoffset)
    DecHeight = rowSize * ZL # Detector length along Z = detectorRowSize * detectorRowsPerMod
    h = h/DecHeight # distance per Rev (mm) / Detector length along Z (mm) = normalized pitch h = H/L
    dx = 2*ObjR/imageSize
    dy = 2*ObjR/imageSize
    dz = sliceThickness

    YLCW = (YL-1)*0.5
    YLC= (YL-1)*0.5 - dectorYoffset  # Detector center along the horizontal direction of detector array
    ZLC= (ZL-1)*0.5 - dectorZoffset # Detector center along the vertical direction of detector array
    
    ###### Note "startAngle" is at the middle between BetaS and BetaE ######
    ###### The angle between BetaS and BetaE is the Scan Angle Coverange ######
    BetaS = -N_Turn*PI  + PI*startAngle/180 # Start Angle, could smaller than 0deg, N_Turn = viewCount/viewsPerRotation
    BetaE =  N_Turn*PI  + PI*startAngle/180 # End Angle, could larger than 360deg
    # BetaS = 0 # Start Angle, could smaller than 0deg, N_Turn = viewCount/viewsPerRotation
    # BetaE = 2*PI*N_Turn  #+ PI*startAngle/180 # End Angle, could larger than 360deg

    # True Start Angle and End Angle
    #betaStart = PI*startAngle/180 # Start Angle
    #betaEnd = BetaE - BetaS + betaStart # End Angle

    DecWidth = math.tan(DecAngle*0.5)*(SO)*2
    dYL   =  DecWidth/YL
    dZL   =  DecHeight/ZL
    DeltaFai= (BetaE-BetaS)/(ViewN-1)
    DeltaTheta = DeltaFai
    DeltaT     = dYL  
    dYA = DecAngle/YL

    PProj    = np.zeros((ViewN,YL,ZL))
    
    
    ########### Trim Sinogram #################### 
    # define allowed angular sector per rotation (in degrees) 
    def compute_beta_array(cfg): # Extract parameters 
        startAngle = cfg.protocol.startAngle * math.pi/180.0 # radians 
        viewCount = cfg.protocol.viewCount 
        viewsPerRotation = cfg.protocol.viewsPerRotation 
        rotDir = cfg.protocol.rotationDirection # +1 or -1 
        # Compute beta per view 
        beta_array = startAngle + (np.arange(viewCount) / viewsPerRotation) * rotDir * math.tau 
        return beta_array 
    
    # ############ Hard Cut Sinogram #############
    # allowed_start = 0 # e.g., 0 deg 
    # allowed_end = cfg.protocol.coverage # e.g., 230 deg 
    # angle_mod = np.degrees(np.mod(compute_beta_array(cfg), 2*np.pi)) # mask views outside allowed sector 
    # mask_allowed = (angle_mod >= allowed_start) & (angle_mod <= allowed_end) 
    # Proj[~mask_allowed, :, :] = 0.0 # or np.nan if you want NaNs
    # ############ End Hard Cut Sinogram #############
    
    
    
    ############ Try Graduately Cut/Trim Sinogram ####################
    fade_deg = 20.0  # width of fade near start/end in degrees (tune 5~30)
    def smoothstep(x):
        # x in [0,1]
        return x*x*(3 - 2*x)
    allowed_start = 0.0
    allowed_end   = float(cfg.protocol.coverage)  # e.g. 230
    
    angle_deg = np.degrees(np.mod(compute_beta_array(cfg), 2*np.pi))
    
    w_ang = np.zeros_like(angle_deg, dtype=np.float32)
    
    # fully-on region
    core = (angle_deg >= (allowed_start + fade_deg)) & (angle_deg <= (allowed_end - fade_deg))
    w_ang[core] = 1.0
    
    # ramp up near start
    ramp_up = (angle_deg >= allowed_start) & (angle_deg < (allowed_start + fade_deg))
    x = (angle_deg[ramp_up] - allowed_start) / fade_deg  # 0..1
    w_ang[ramp_up] = smoothstep(x).astype(np.float32)
    
    # ramp down near end
    ramp_dn = (angle_deg > (allowed_end - fade_deg)) & (angle_deg <= allowed_end)
    x = (allowed_end - angle_deg[ramp_dn]) / fade_deg     # 0..1
    w_ang[ramp_dn] = smoothstep(x).astype(np.float32)
    
    # apply to Proj (Proj is [view, YL, ZL] in your code at that moment)
    Proj *= w_ang[:, None, None]
    ############ End Try Graduate Trim Sinogram #################### 
    
    
    ############### Plot Entire Sinogram ################
    det_row = ZL // 2   # middle row
    Sino = Proj[:, :, det_row]   # shape (totalViews, YL)
    alpha = (YLC - np.arange(YL)) * dYA * 180/np.pi
    beta = np.degrees(np.mod(compute_beta_array(cfg), 2*np.pi))
    plt.figure(figsize=(8,6))
    plt.imshow(
        Sino,
        extent=[alpha[0], alpha[-1], beta[0], beta[-1]],
        origin='lower',
        aspect='auto',
        cmap='gray',
        interpolation='none'
    )
    plt.colorbar(label='Projection value')
    plt.xlabel('Fan angle α (deg)')
    plt.ylabel('Projection angle β (deg)')
    plt.title('Full Sinogram (After Angular Fading)')
    plt.show()
    ############### End Plot Entire Sinogram ################
    
    
    #################### Apply Short Scan Weighting #####################
    viewsPerRot = cfg.protocol.viewsPerRotation
    totalViews  = cfg.protocol.viewCount
    viewsShort = int((cfg.protocol.coverage/360) * cfg.protocol.viewsPerRotation)
    
    numRots = totalViews // viewsPerRot
    #print(f"Detected {numRots} full rotations")

    Gamma_m = DecAngle / 2
    #print("* Applying helical short-scan weighting...")
    
    # allocate full weighting map
    W_full = np.zeros((totalViews, YL), dtype=np.float32)
    
    for rot in range(numRots):
        v_start = rot * viewsPerRot
        v_end   = v_start + viewsPerRot # viewsShort
    
        # Extract one-rotation sinogram block
        Proj_block = Proj[v_start:v_end, :, :]   # shape: [viewsPerRot, YL, ZL]
    
        # Define angular parameters FOR THIS ROTATION
        # ViewN = viewsPerRot
        BetaS_rot = 0.0                 # reset start angle per rotation
        DeltaFai  = 2*np.pi / viewsPerRot
    
        # --- Compute weighting for this rotation ---
        W = silver_weighting(
            viewsShort, #viewsPerRot,
            YL,
            BetaS_rot,
            DeltaFai,
            YLC,
            dYA,
            Gamma_m
        )
    
        # --- Apply weighting ---
        for i in range(viewsShort): # (viewsPerRot):
            Proj_block[i, :, :] *= W[i, :, None]
    
        # Write back
        Proj[v_start:v_end, :, :] = Proj_block
        
        # place into full array
        W_full[v_start:v_start+viewsShort, :] = W
      #################### End Apply Short Scan Weighting #####################
     
        
    ############### Plot Weighing Function on Entire Sinogram ################
    alpha = (YLC - np.arange(YL)) * dYA * 180/np.pi
    beta = np.arange(totalViews) * DeltaFai * 180/np.pi
    plt.figure(figsize=(8,6))
    plt.imshow(
        W_full,
        extent=[alpha[0], alpha[-1], beta[0], beta[-1]],
        origin='lower',
        aspect='auto',
        cmap='viridis',
        interpolation='none'
    )
    plt.colorbar(label='Weight')
    plt.xlabel('Fan angle α (deg)')
    plt.ylabel('Projection angle β (deg)')
    plt.title('Short-Scan Weighting Map (Full Scan)')
    plt.show()
    ############### End Plot Weighing Function on Entire Sinogram ################
    
    
    # ############### Plot Weighing Function ################
    # # Suppose W.shape = (ViewN, YL)
    # beta = BetaS + np.arange(viewsShort) * DeltaFai * 180/PI  # full projection angles in deg
    # alpha = -(YLC - np.arange(YL)) * dYA * 180/PI         # fan angles in deg

    # plt.figure(figsize=(7, 6))
    # plt.imshow(
    #     W,
    #     extent=[alpha[0], alpha[-1], beta[0], beta[-1]],
    #     origin='lower',        # makes β increase upward (more intuitive)
    #     aspect='auto',
    #     cmap='viridis',
    #     interpolation='none'
    # )
    # plt.colorbar(label='Weight')
    # plt.xlabel('Fan angle α (deg)')
    # plt.ylabel('Projection angle β (deg)')
    # plt.title("Parker/Silver Weighting Function")
    # plt.show()


    # ############### Weight the Projections ###############
    # for i in range(ViewN):
    #     for j in range(YL):
    #         for k in range(ZL):
    #             Proj[i,j,k] = Proj[i,j,k]*W[i,j]




    ## rebinning the projection
    print("* Rebinning the projection...")
    for i in range(ViewN):   
        Theta=(i)*DeltaTheta                   # the view for the parallel projection
        for j in range(YL):
            t      = (j-YLCW)*DeltaT     # the distance from origin to ray for parallel beam
            Beta   = math.asin(t/(SO))            # the fan_angle for cone_beam projection
            Fai    = Theta+Beta              # the view for cone_beam projecton
            a      = Beta  # the position of this ray on the flat detector
            FaiIndex        =  (Fai/DeltaFai)
            UIndex          =  (a/dYA)+YLC
            FI              =  math.ceil(FaiIndex)
            UI              =  math.ceil(UIndex)
            coeXB           =  FI-FaiIndex
            coeXU           =  1-coeXB
            coeYB           =  UI-UIndex
            coeYU           =  1-coeYB
            if (FI<=0):
                IndexXU = 0
                IndexXB = 0
            elif(FI > ViewN-1):
                IndexXU = ViewN-1
                IndexXB = ViewN-1
            else:
                IndexXU = FI
                IndexXB = FI-1

            if (UI<=0):
                IndexYU = 0
                IndexYB = 0
            elif(UI>YL-1):
                IndexYU = YL-1
                IndexYB = YL-1
            else:
                IndexYU=UI
                IndexYB=UI-1
            PProj[i,j,:]=coeXB*coeYB*Proj[IndexXB,IndexYB,:]+coeXU*coeYB*Proj[IndexXU,IndexYB,:]+coeXB*coeYU*Proj[IndexXB,IndexYU,:]+coeXU*coeYU*Proj[IndexXU,IndexYU,:]
    
    
    # https://github.com/xcist/main/issues/122
    Projflip = np.flip(PProj, axis=2)
    Proj = Projflip.transpose(1, 2, 0) # final re-binned sinogram (detector, slice, angle)
    #print('Proj max = ', np.max(Proj))
    #scio.savemat('testrebin.mat', {'rebin': PProj})
    
    ##################### Perform Ramp filtering ######################
    print("* Applying the filter...")
    Dg=Proj
    nn = int(math.pow(2, (math.ceil(math.log2(abs(YL))) + 1)))
    nn2 = nn*2    
    FFT_F = createHSP(nn, kernelType)
    
    GF = Proj
    
    for ProjIndex in range(0, ViewN):
        for j in range(ZL):
            TempData = np.ones(YL)
            for k in range(YL):
                TempData[k] = Dg[k, j, ProjIndex]
            FFT_S = np.fft.fft(TempData, nn2)
            TempData = np.fft.ifft(FFT_S * FFT_F).imag
            for k in range(YL):
                GF[k, j, ProjIndex] = TempData[k]
    GF = GF/dYL # Filtered Rebinned Sinogram
    # print('GF max, min = ', np.max(GF), np.min(GF))
    # print('GF shape = ', np.shape(GF))
    
    
    # print("Proj shape (after transpose to Sinogram) =", Proj.shape)  # expect (View?, YL, ZL)
    # print("PProj shape =", PProj.shape)                              # expect (View?, YL, ZL)
    # print("GF shape =", GF.shape)                                    # expect (YL, ZL, View?)
    # print("cfg viewCount =", cfg.protocol.viewCount)
    # print("cfg viewsPerRotation =", cfg.protocol.viewsPerRotation)

    
    # ########### DO 360LI RECON ############### This is to run recon in Python, not in C
    # RecIm = np.zeros(shape=(imageSize, imageSize, sliceCount), dtype=np.float32)

    # # ---- 360LI parameters ----
    # H_mm_per_rev = h * DecHeight              # NOTE: your `h` has been normalized earlier (H/L), so H = h*L
    # DeltaTheta   = 2*np.pi / N_2pi            # one full rotation sampled by N_2pi views
    # dt_mm        = DeltaT                     # parallel-bin spacing in mm (your rebin uses DeltaT = dYL)
    # t0           = (YL - 1) * 0.5             # detector center index in t

    # # view angles for the base 0..2π set
    # theta_base = np.arange(N_2pi, dtype=np.float32) * DeltaTheta

    # # assume table z at BetaS is 0 (consistent with your BetaS = 0 usage above)
    # # z(beta) = (H/(2π)) * beta
    # # (if you have an absolute z-start, add it here)
    # z_start = 0.0

    # # precompute z_view for ALL views in GF (ViewN total)
    # beta_all = BetaS + np.arange(ViewN, dtype=np.float32) * DeltaFai # DeltaFai = DeltaTheta
    # z_view_all = z_start + (H_mm_per_rev / (2.0*np.pi)) * beta_all  # mm

    # # image grid coordinates (mm)
    # xc = (imageSize - 1) * 0.5
    # yc = (imageSize - 1) * 0.5
    # x_grid = (np.arange(imageSize, dtype=np.float32) - xc) * dx
    # y_grid = (np.arange(imageSize, dtype=np.float32) - yc) * dy

    # # choose which detector row to use (single-row case -> 0)
    # # if your data truly has one row, ZL should be 1.
    # det_row = 0

    # # ---- reconstruct each requested slice z ----
    # cos_t = np.cos(theta_base).astype(np.float32)
    # sin_t = np.sin(theta_base).astype(np.float32)
    
    # # Precompute X,Y grids once per slice (or move outside zi-loop if you like)
    # X, Y = np.meshgrid(x_grid, y_grid)  # shape (imageSize, imageSize)
    
    # # detector coordinate samples in mm for np.interp
    # t_det = (np.arange(YL, dtype=np.float32) - t0) * dt_mm
    
    # # zc = (sliceCount - 1) * 0.5
    # for zi in range(sliceCount):
    #     # desired slice z location (mm)
    #     # z = (zi - zc) * dz
    #     z_recon = z_view_all[N_2pi]/2

    #     # Build the synthetic axial sinogram for this z using 360LI:
    #     # G_axial[t, i] for i in [0..N_2pi-1]
    #     G_axial = np.zeros((YL, N_2pi), dtype=np.float32)
        
    #     # compute W once (outside the i-loop)
    #     W = silver_weighting(
    #         #viewsPerRot,
    #         viewsShort,
    #         YL,
    #         0.0,
    #         2*np.pi / viewsPerRot,
    #         YLC,
    #         dYA,
    #         Gamma_m
    #     )  # (viewsPerRot, YL)

    #     for i in range(int(cfg.protocol.coverage/360 * N_2pi)):
    #     #for i in range(N_2pi):
    #         # Find p such that z_view[p] <= z <= z_view[p + N_2pi]
    #         # We start from the i-th angle within the base rotation.
    #         # We'll shift by k rotations until we bracket z.
    #         # k = floor((z - z_view[i]) / H)
    #         # z_i = z_view_all[i]
    #         #k = int(np.floor((z - z_i) / H_mm_per_rev))
    #         # p = i + k * N_2pi
    #         p = i

    #         z_p  = z_view_all[p]
    #         z_p2 = z_view_all[p + N_2pi]
            
    #         z = z_recon - z_p

    #         # interpolation weight in z
    #         denom = (z_p2 - z_p)
    #         # if denom == 0:
    #         #     w = 0.0
    #         # else:
    #         #     # w = (z - z_p) / denom
    #         #     w = (z - z_p) / denom
    #         # if w < 0.0: w = 0.0
    #         # if w > 1.0: w = 1.0
            
    #         w = z / denom
    #         # 360LI interpolation between the two rotations (already filtered, rebinned)
    #         g0 = GF[:, det_row, p]
    #         g1 = GF[:, det_row, p + N_2pi]
    #         G_axial[:, i] = (1.0 - w) * g0 + w * g1
            
    #         # apply weighting for this angle i
    #         G_axial[:, i] *= W[i, :]


                
    #             # # Write back
    #             # Proj[v_start:v_end, :, :] = Proj_block
                
            
    #     # print("Recon Finished")
    #     # print("imageSize = ", imageSize)
    #     # print("N_2pi = ", N_2pi)
    #     print("z_p, z_p2 = ", z_p, z_p2)
    #     print("z = ", z)
    #     print("p, p2 = ", p, p + N_2pi)
    #     print("ViewN =", ViewN, "N_2pi =", N_2pi)
    #     print("Rotations in data =", ViewN / N_2pi)
    #     print("np.shape(G_axial), np.shape(W) =", np.shape(G_axial), np.shape(W))
        
    #     img = np.zeros((imageSize, imageSize), dtype=np.float32)
        
    #     for i in range(N_2pi):
    #         # t(x,y) for this angle
    #         t_img = X * cos_t[i] + Y * sin_t[i]  # shape (512,512)
        
    #         # interpolate G_axial(:, i) at these t locations
    #         # np.interp works on 1D, so flatten and reshape
    #         vals = np.interp(t_img.ravel(), t_det, G_axial[:, i], left=0.0, right=0.0)
    #         img += vals.reshape(imageSize, imageSize)
        
    #     RecIm[:, :, zi] = img * DeltaTheta

    # # --- Plot setup ---
    # # fig, ax = plt.subplots(figsize=(6, 6))

    # # ax.imshow(RecIm, cmap='gray')
    
    # return RecIm
    #  ########### END DO 360LI RECON ############### This is to run recon in Python, not in C







    ################# All below are Doing Recon in C ##################
    
    #Backproject the filtered data into the 3D space
    # Load the compiled library
    recon = load_C_recon_lib()
    # Define arguments of the C function
    recon.fbp.argtypes = [ct.POINTER(TestStruct)]
    # Define the return type of the C function
    recon.fbp.restype = None

    # init the struct
    t = TestStruct()

    t.ScanR = SO
    t.DistD = DD
    t.YL = YL
    t.ZL = ZL
    t.DecFanAng = DecAngle
    t.DecHeight = DecHeight
    t.DecWidth = DecWidth
    t.dx = dx
    t.dy = dy 
    t.dz = dz
    t.h = h
    t.BetaS = BetaS
    t.BetaE = BetaE
    t.dectorYoffset = dectorYoffset
    t.dectorZoffset = dectorZoffset
    t.AngleNumber = ViewN
    t.N_2pi = N_2pi
    t.Radius = ObjR
    t.RecSize = imageSize
    t.RecSizeZ = sliceCount
    t.delta = delta
    t.HSCoef = HSCoef
    t.k1 = k1

    t.XOffSet = centerOffset[0]
    t.YOffSet = centerOffset[1]
    t.ZOffSet = centerOffset[2]
    t.phantomXOffSet = 0
    t.phantomYOffSet = 0
    t.phantomZOffSet = 0

    if cfg.recon.printReconParameters:
        print("* Reconstruction parameters:")
        print("* SID: {} mm".format(t.ScanR))
        print("* SDD: {} mm".format(t.DistD))
        print("* Fan angle: {} degrees".format(t.DecFanAng))
        # print("* Start view: {}".format(t.startAngle))
        print("* Number of detector cols: {}".format(t.YL))
        print("* Number of detector rows: {}".format(t.ZL))
        print("* Detector height: {} mm".format(t.DecHeight))
        print("* Detector X offset: {} mm".format(t.dectorYoffset))
        print("* Detector Z offset: {} mm".format(t.dectorZoffset))
        print("* Scan number of views: {} ".format(t.AngleNumber))
        print("* Recon FOV: {} mm".format(2 * t.Radius))
        print("* Recon XY pixel size: {} mm".format(t.RecSize))
        print("* Recon Slice thickness: {} mm".format(t.sliceThickness))
        print("* Recon X offset: {} mm".format(t.XOffSet))
        print("* Recon Y offset: {} mm".format(t.YOffSet))
        print("* Recon Z offset: {} mm".format(t.ZOffSet))
    # Generate a 3D ctypes array from numpy array
    print("* Converting projection data from a numpy array to a C array...")
    GF_ptr = double3darray2pointer(GF)
    t.GF = GF_ptr

    # RecIm = np.zeros(shape=(t.RecSize, t.RecSize, t.RecSize))
    print("* Allocating a C array for the recon results...")
    RecIm = np.zeros(shape=(t.RecSize, t.RecSize, t.RecSizeZ))
    RecIm_ptr = double3darray2pointer(RecIm)
    t.RecIm = RecIm_ptr

    # interface with C function
    print("* In C...")
    
    start_time = time.perf_counter()
    
    recon.fbp(ct.byref(t))
    
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    print(f"Recon time: {elapsed_time:.4f} seconds")

    # Convert ctypes 3D arrays to numpy arrays
    rec = double3dpointer2array(RecIm_ptr, *RecIm.shape)
    rec = rec.transpose(1,0,2).astype(np.float32)
    rec = rec[::-1,::-1]
    return rec



######## Removed from .cu #########
# // ------------------- Helpers to allocate host 3D pointer arrays -------------------
# static float*** alloc3D(int A, int B, int C) {
#     float*** p = (float***)malloc((size_t)A * sizeof(float**));
#     for (int a = 0; a < A; a++) {
#         p[a] = (float**)malloc((size_t)B * sizeof(float*));
#         for (int b = 0; b < B; b++) {
#             p[a][b] = (float*)malloc((size_t)C * sizeof(float));
#             std::memset(p[a][b], 0, (size_t)C * sizeof(float));
#         }
#     }
#     return p;
# }
# static void free3D(float*** p, int A, int B) {
#     for (int a = 0; a < A; a++) {
#         for (int b = 0; b < B; b++) free(p[a][b]);
#         free(p[a]);
#     }
#     free(p);
# }

# // ------------------- main(): simple runnable test -------------------
# int main() {
#     TestStruct t = {};
#     // Small sizes for a quick test (you can change these)
#     t.YL = 32;
#     t.ZL = 16;
#     t.AngleNumber = 64;   // PN
#     t.N_2pi = 128;

#     t.RecSize = 64;       // XN==YN
#     t.RecSizeZ = 8;       // ZN

#     // Some reasonable-ish numbers (not physically perfect, but good for test)
#     t.ScanR = 500.0f;
#     t.DistD = 800.0f;
#     t.DecFanAng = 1.0f;
#     t.DecHeight = 200.0f;
#     t.DecWidth  = 200.0f;

#     t.dx = 1.0f; t.dy = 1.0f; t.dz = 2.0f;
#     t.h = 1.0f;               // will be multiplied by DecHeight inside fbp()
#     t.BetaS = 0.0f;
#     t.BetaE = 2.0f*pi;

#     t.Radius = 100.0f;
#     t.delta = 10.0f;
#     t.HSCoef = 1.0f;
#     t.k1 = 1.0f;

#     t.XOffSet = 0.0f;
#     t.YOffSet = 0.0f;
#     t.ZOffSet = 0.0f;
#     t.dectorYoffset = 0.0f;
#     t.dectorZoffset = 0.0f;

#     // Allocate host arrays
#     t.GF    = alloc3D(t.YL, t.ZL, t.AngleNumber);
#     t.RecIm = alloc3D(t.RecSize, t.RecSize, t.RecSizeZ); // [YN][XN][ZN]

#     // Fill GF with some dummy pattern
#     for (int u = 0; u < t.YL; u++) {
#         for (int v = 0; v < t.ZL; v++) {
#             for (int p = 0; p < t.AngleNumber; p++) {
#                 t.GF[u][v][p] = 0.001f * (u + 1) + 0.0001f * (v + 1) + 0.00001f * (p + 1);
#             }
#         }
#     }

#     printf("Running fbp() on GPU...\n");
#     fbp(&t);

#     // Print a few output voxels
#     printf("RecIm[0][0][0] = %f\n", t.RecIm[0][0][0]);
#     printf("RecIm[10][10][0] = %f\n", t.RecIm[10][10][0]);
#     printf("RecIm[10][10][7] = %f\n", t.RecIm[10][10][7]);

#     // Cleanup
#     free3D(t.GF, t.YL, t.ZL);
#     free3D(t.RecIm, t.RecSize, t.RecSize);

#     return 0;
# }