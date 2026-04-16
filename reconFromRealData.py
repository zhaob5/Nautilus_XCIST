# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 14:35:50 2026

@author: Botao Zhao
"""
import os
import ctypes
import numpy as np
import matplotlib.pyplot as plt
import gecatsim as xc
from gecatsim.reconstruction.pyfiles.axial_short_scan_realData import axial_short_scan_realData

######## For importing real trajectory data #########
def load_trajectory(npz_path, src_key, det_key, geo_key=None):
    """
    Load and select source/detector trajectories from npz file.

    Args:
        npz_path (str): path to .npz file
        src_key (str): key name for source trajectory
        det_key (str): key name for detector trajectory
        geo_key (str, optional): key for phantom/geometry points

    Returns:
        src (ViewN, 2)
        det (ViewN, 2)
        geo (optional)
    """
    data = np.load(npz_path)

    print("Trajectory arrays:", data.files)

    src = data[src_key]
    det = data[det_key]

    # sanity check
    assert src.shape[1] == 2, f"{src_key} is not (N,2)"
    assert det.shape[1] == 2, f"{det_key} is not (N,2)"

    if geo_key is not None:
        geo = data[geo_key]
        return src, det, geo

    return src, det
##################################################


############# Convert (x,y) → view angle + SID + SDD per view ################
def compute_view_geometry(src, det, iso_center=(0.0, 0.0)):
    """
    Compute per-view angle (deg), SID, and SDD from source/detector trajectories.

    Args:
        src: ndarray, shape (ViewN, 2)
        det: ndarray, shape (ViewN, 2)
        iso_center: tuple/list/array-like, (x0, y0)

    Returns:
        view_angle_deg: ndarray, shape (ViewN,)
        SID: ndarray, shape (ViewN,)
        SDD: ndarray, shape (ViewN,)
    """
    src = np.asarray(src, dtype=np.float64)
    det = np.asarray(det, dtype=np.float64)
    iso = np.asarray(iso_center, dtype=np.float64)

    src_rel = src - iso

    # source-to-isocenter distance
    SID = np.linalg.norm(src_rel, axis=1)

    # source-to-detector distance
    SDD = np.linalg.norm(det - src, axis=1)

    # source angle around isocenter, in radians
    view_angle_rad = np.arctan2(src_rel[:, 1], src_rel[:, 0])

    # unwrap to make angle continuous
    view_angle_rad = np.unwrap(view_angle_rad)

    # # convert to degrees
    # view_angle_deg = np.degrees(view_angle_rad)

    return view_angle_rad, SID, SDD
#########################################################

########## For multi-threaded CPU Speedup ###############
dll_dir = os.path.join(os.path.dirname(xc.__file__), "lib")
mingw_dir = r"C:\Program Files\mingw64\bin"   # change if your MinGW is elsewhere 

os.add_dll_directory(dll_dir)
os.add_dll_directory(mingw_dir)

ctypes.CDLL(os.path.join(dll_dir, "libcatsim64.dll"))
########## For multi-threaded CPU Speedup ###############

########## Load Sinogram ###########
# file_path = r"F:\Shared drives\Nautilus Engineering\XCIST\Nautilus_XCIST\realData\SinovisionMTF1_sweep0_sinogram.npy"
file_path = r"F:\Shared drives\Nautilus Engineering\XCIST\Nautilus_XCIST\realData\ImatronLP1_sweep0_sinogram.npy"
# file_path = r"F:\Shared drives\Nautilus Engineering\XCIST\Nautilus_XCIST\realData\Mpphan1_sweep0_sinogram.npy"
# file_path = r"F:\Shared drives\Nautilus Engineering\XCIST\Nautilus_XCIST\realData\SinovisionLP1_sweep0_sinogram.npy"

file_name = os.path.splitext(os.path.basename(file_path))[0]
##########################################################

######### Load Source and Detector Trajctory ######
traj_path = r"F:\Shared drives\Nautilus Engineering\XCIST\Nautilus_XCIST\realData\pin_results_Mpphan1.npz"

src, det, geo = load_trajectory(
    traj_path,
    src_key='array2',
    det_key='array3',
    geo_key='array1'
)

print("src shape:", src.shape)
print("det shape:", det.shape)
######################################################

############ Convert (x,y) → view angle + SID + SDD #############
view_angle, SID, SDD = compute_view_geometry(src, det)
#######################################################

##--------- Initialize 
ct = xc.CatSim("phantom", "protocol", "scanner", "recon", "physics")  # initialization "example_physics"  will add more reality
cfg = ct.get_current_cfg()

prep = np.load(file_path).astype(np.float32)

# optional center crop
use_crop = True
if use_crop:
    target_cols = cfg.scanner.detectorColCount
    total_cols = prep.shape[2]
    start = (total_cols - target_cols) // 2
    end = start + target_cols
    prep = prep[:, :, start:end]
    
    # make sure cfg matches cropped geometry
    # cfg.scanner.detectorColCount = target_cols
    # cfg.scanner.detectorColSize = original_col_size
    # or update total detector width consistently

print("input prep shape:", prep.shape)

rec = axial_short_scan_realData(cfg, prep, SID, SDD, view_angle)

print("recon done, shape:", rec.shape)
# np.save("recon_real_shortscan.npy", rec)

img_sum = np.sum(rec[:, :, 1:-1], axis=2)
img_avg = np.mean(rec[:, :, 1:-1], axis=2)
img = img_avg

########## Plot sino and recon vertically ##########
mid_row = prep.shape[1] // 2
sino_2d = prep[:, mid_row, :]

##----------Plot Sinogram
plt.figure() 
plt.imshow(sino_2d, aspect='auto') 
plt.xlabel("Detector Column") 
plt.ylabel("View (Angle Index)") 
plt.title("Sinogram (Middle Detector Row)") 
plt.colorbar() 
plt.show()

##----------Plot Recon
from scipy.ndimage import rotate
angle_deg = -90  # your desired rotation
img_rot = rotate(img, angle_deg, reshape=False)
img_flip = np.fliplr(img_rot)
plt.figure()
plt.imshow(img_flip, cmap='gray')
plt.title(f"Reconstruction (rotated {angle_deg}°)\n{file_name}")
plt.show()