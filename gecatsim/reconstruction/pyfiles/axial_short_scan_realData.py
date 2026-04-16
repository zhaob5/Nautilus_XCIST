import ctypes as ct
import numpy as np
import math
from gecatsim.reconstruction.pyfiles.createHSP import createHSP
import scipy.io as scio
import matplotlib.pyplot as plt
from gecatsim.reconstruction.pyfiles.mapConfigVariablesToHelical import mapConfigVariablesToHelical
import os

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
                ("RecIm", PtrPtrPtrFLOAT),
                # NEW: per-view arrays
                ("SIDArray", PtrFLOAT),
                ("SDDArray", PtrFLOAT),
                ("ViewAngleArray", PtrFLOAT),
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
    if os.name == "nt":
        lib_file = "Axial_Short_Scan_CUDA_realData.dll"
    # else:
    #     lib_file = lib_file = "helicalrecon.so"
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
    Delta = ScanRange - (pi + 2 * Gamma_m) + BetaS
    Delta = max(0.0, Delta)  # ensure non-negative

    w = np.ones((ViewN, YL), dtype=np.float32)

    for s in range(ViewN):
        beta = BetaS + s * DeltaFai
        for d in range(YL):
            alpha = (YLC - d) * dDecL
            
            # Ramp-up region
            beta1 = 0
            #beta2 = 2 * (Gamma_m - alpha) + Delta / 2
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
            else:
                w[s, d] = 1.0
    return w


def axial_short_scan_realData(cfg, prep, SID=None, SDD=None, view_angle=None):
    prep = prep[:,:,::-1]

    if cfg.protocol.rotationDirection == -1:	
        prep = prep[::-1,:,:]
    Proj = prep.transpose(0,2,1) # Sinogram

    # scanner & recon geometry
    SO, DD, YL, ZL, ViewN, N_Turn, N_2pi, h, startAngle, dectorYoffset, dectorZoffset, \
    k1, delta, HSCoef, rowSize, ColSize, imageSize, sliceCount, sliceThickness, centerOffset, ObjR,kernelType \
        = mapConfigVariablesToHelical(cfg)
        
    ####################  Use Real trajectory Data #####################
    if SID is not None:
        SID = np.asarray(SID, dtype=np.float32)
        if len(SID) != ViewN:
            raise ValueError("SID length must match ViewN")
        SO = float(np.mean(SID))
    
    if SDD is not None:
        SDD = np.asarray(SDD, dtype=np.float32)
        if len(SDD) != ViewN:
            raise ValueError("SDD length must match ViewN")
        DD = float(np.mean(SDD))
    
    if view_angle is not None:
        view_angle = np.asarray(view_angle, dtype=np.float32)
        if len(view_angle) != ViewN:
            raise ValueError("view_angle length must match ViewN")
    
        view_angle_rad = view_angle
        BetaS = float(view_angle_rad[0])
        BetaE = float(view_angle_rad[-1])
        DeltaFai = float(np.mean(np.diff(view_angle_rad)))
        # BetaS = PI * startAngle / 180
        # BetaE = BetaS + 2 * PI * N_Turn
        # DeltaFai = (BetaE - BetaS) / ViewN
    else:
        BetaS = PI * startAngle / 180
        BetaE = BetaS + 2 * PI * N_Turn
        DeltaFai = (BetaE - BetaS) / ViewN
        
    DecAngle = math.atan(ColSize/2/DD)*2*YL # Fan Angle, DD = source-to-detector distance, YL = total number of detector columns
    DecHeight = rowSize * ZL # Detector length along Z = detectorRowSize * detectorRowsPerMod
    h = h/DecHeight # distance per Rev (mm) / Detector length along Z (mm) = normalized pitch h = H/L
    dx = 2*ObjR/imageSize
    dy = 2*ObjR/imageSize
    dz = sliceThickness

    YLCW = (YL-1)*0.5
    YLC= (YL-1)*0.5 - dectorYoffset  # Detector center along the horizontal direction of detector array
    ZLC= (ZL-1)*0.5 - dectorZoffset # Detector center along the vertical direction of detector array
    
    ###### The angle between BetaS and BetaE is the Scan Angle Coverange ######
    # BetaS = PI*startAngle/180 # Start Angle, could smaller than 0deg, N_Turn = viewCount/viewsPerRotation
    # BetaE = BetaS + 2*PI*N_Turn #BetaS + (ViewN/N_2pi*2*PI)

    DecWidth = math.tan(DecAngle*0.5)*(SO)*2
    dYL   =  DecWidth/YL
    dZL   =  DecHeight/ZL
    #DeltaFai= (BetaE-BetaS)/(ViewN-1)
    # DeltaFai= (BetaE-BetaS)/(ViewN)
    DeltaTheta = DeltaFai
    DeltaT     = dYL  
    dYA = DecAngle/YL

    PProj    = np.zeros((ViewN,YL,ZL))
    
    
    ############### Applying Parker's Weighing Function ################
    Gamma_m = DecAngle/2
    print("* Applying weighting function...")
    #W = parkers_weighting(ViewN, YL, BetaS, DeltaFai, YLC, dYA, Gamma_m) # For Half-Scan only
    W = silver_weighting(ViewN, YL, BetaS, DeltaFai, YLC, dYA, Gamma_m) # For General Short-Scan
    
    # ############### Plot Weighing Function ################
    # # Suppose W.shape = (ViewN, YL)
    # beta = BetaS + np.arange(ViewN) * DeltaFai * 180/PI  # full projection angles in deg
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
    # #plt.show()


    ############### Weight the Projections ###############
    for i in range(ViewN):
        for j in range(YL):
            for k in range(ZL):
                Proj[i,j,k] = Proj[i,j,k]*W[i,j]


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
            UIndex          =  (a/dYA)+YLC     ########## = -(a/dYA)+YLC for simulation traj
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
    
    # PProj = Proj.copy()
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
    #print('GF max, min = ', np.max(GF), np.min(GF))
    
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
    
    t.SIDArray = SID.ctypes.data_as(PtrFLOAT)
    t.SDDArray = SDD.ctypes.data_as(PtrFLOAT)
    t.ViewAngleArray = view_angle.ctypes.data_as(PtrFLOAT)

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
    recon.fbp(ct.byref(t))

    # Convert ctypes 3D arrays to numpy arrays
    rec = double3dpointer2array(RecIm_ptr, *RecIm.shape)
    rec = rec.transpose(1,0,2).astype(np.float32)
    rec = rec[::-1,::-1]
    return rec
