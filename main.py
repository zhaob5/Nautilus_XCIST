# Copyright 2024, GE Precision HealthCare. All rights reserved. See https://github.com/xcist/main/tree/master/license


import os
import ctypes

###------------ import XCIST-CatSim
import numpy as np
import gecatsim as xc
import gecatsim.reconstruction.pyfiles.recon as recon
import time
import math

########## For multi-threaded CPU Speedup ###############
dll_dir = os.path.join(os.path.dirname(xc.__file__), "lib")
mingw_dir = r"C:\Program Files\mingw64\bin"   # change if your MinGW is elsewhere 

os.add_dll_directory(dll_dir)
os.add_dll_directory(mingw_dir)

ctypes.CDLL(os.path.join(dll_dir, "libcatsim64.dll"))
########## For multi-threaded CPU Speedup ###############



##--------- Initialize 
ct = xc.CatSim("phantom", "protocol", "scanner", "recon", "physics")  # initialization "example_physics"  will add more reality

# ##--------- Make changes to parameters (optional)
ct.resultsName = "out"

##--------- Run simulation
ct.run_all()  # run the scans defined by protocol.scanTypes

# ##--------- Reconstruction
cfg = ct.get_current_cfg()
cfg.do_Recon = 1
cfg.waitForKeypress = 0

start_time = time.time()        # record start time
recon.recon(cfg)                # run your reconstruction
end_time = time.time()          # record end time
elapsed = end_time - start_time
print(f"Reconstruction completed in {elapsed:.2f} seconds.")

##--------- Show results
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

################ Plot Sinogram ###############
def compute_triangle_edges(YL, YLC, dDecL, ViewN, BetaS, DeltaFai, Gamma_m):
    ScanRange = ViewN * DeltaFai  # total scan angle coverage
    # Extra angular coverage beyond half-scan
    # Delta = 2Gamma_virtual
    Delta = ScanRange - (np.pi + 2 * Gamma_m)
    Delta = max(0.0, Delta)  # ensure non-negative
    
    edges_up = []     # (d, s_idx) for ramp-up boundary beta2
    edges_down = []   # (d, s_idx) for ramp-down boundary beta3

    for d in range(YL):
        alpha = (d - YLC) * dDecL

        # ---- same formulas you used ----
        beta2 = 2 * (Gamma_m + Delta/2 - alpha)
        #beta2 = 2 * (Gamma_m + (0) / 2 - alpha)   # your formula
        beta3 = np.pi - 2 * alpha                 # your formula

        # Convert beta → view index (s index)
        s2 = int(round((beta2 - BetaS) / DeltaFai))
        s3 = int(round((beta3 - BetaS) / DeltaFai))

        # Only keep edges that fall inside valid range
        if 0 <= s2 < ViewN:
            edges_up.append((d, s2))
        if 0 <= s3 < ViewN:
            edges_down.append((d, s3))

    return np.array(edges_up), np.array(edges_down)


sino = xc.rawread(ct.resultsName+'.prep', [ct.protocol.viewCount, ct.scanner.detectorRowCount, ct.scanner.detectorColCount], 'float')
sino = np.squeeze(sino[:, 0, :])
xc.rawwrite(ct.resultsName+"_sino_%dx%d.raw"%(ct.scanner.detectorColCount, ct.protocol.viewCount), sino.copy(order="C"))

sinoname = ct.resultsName+"_sino_%dx%d.raw"%(ct.scanner.detectorColCount, ct.protocol.viewCount)
sinogram = xc.rawread(sinoname, [ct.protocol.viewCount, ct.scanner.detectorColCount], 'float')

plt.figure(figsize=(8,6))

# Plot the sinogram with equal aspect
ax = plt.gca()
im = ax.imshow(sinogram, cmap='gray', aspect='equal')  # << fix here
#im = ax.imshow(np.flipud(sinogram), cmap='gray', aspect='equal')
ax.invert_yaxis()
plt.title("Sinogram, Rebinned")

# Compute edges
dDecL = math.atan(ct.scanner.detectorColSize/2/ct.scanner.sdd)*2
YL = int(ct.scanner.detectorColCount)
YLC = ct.scanner.detectorColCount // 2
DecAngle = dDecL * YL
N_Turn = (ct.protocol.viewCount-1) / ct.protocol.viewsPerRotation
BetaE = 2 * np.pi * N_Turn
BetaS = ct.protocol.startAngle
ViewN = ct.protocol.viewCount
DeltaFai = (BetaE - BetaS) / (ViewN - 1)
Gamma_m = DecAngle / 2

edges_up, edges_down = compute_triangle_edges(
    YL,
    YLC,
    dDecL,
    ViewN,
    BetaS,
    DeltaFai,
    Gamma_m
)

# Plot edges
# ax.plot(edges_up[:,0], edges_up[:,1], 'r')
# ax.plot(edges_down[:,0], edges_down[:,1], 'r')

plt.xlabel("Detector Index")
plt.ylabel("Projection Angle (deg)")
plt.show()





# ############### Plot Recon Result ###############
# # Window Width (WW) and Level (WL) settings
# WW1, WL1 = 200, 40
# WW2, WL2 = 2000, -500

# # Scan parameters
# views, viewsPerRot = cfg.protocol.viewCount, cfg.protocol.viewsPerRotation
# ScanRange = round(views / viewsPerRot * 360, 1)

# # --- Helper function for windowing ---
# def window_image(img, window_width, window_level):
#     low = window_level - window_width / 2
#     high = window_level + window_width / 2
#     img_windowed = np.clip(img, low, high)
#     return (img_windowed - low) / (high - low)  # scale to [0,1]

# # --- Read reconstructed image ---
# imgFname = f"{ct.resultsName}_{ct.recon.imageSize}x{ct.recon.imageSize}x{ct.recon.sliceCount}.raw"
# img = xc.rawread(imgFname, [ct.recon.sliceCount, ct.recon.imageSize, ct.recon.imageSize], 'float')
# recon_img = img[ct.recon.sliceCount // 2, :, :]

# if cfg.recon.reconType == 'axial_short_scan':
#     recon_img = np.flip(recon_img, axis=(0, 1))

# # --- Plot with two different WW/WL settings ---
# plt.figure(figsize=(6, 10))

# # First (top) plot
# plt.subplot(2, 1, 1)
# plt.imshow(window_image(recon_img, WW1, WL1), cmap='gray')
# plt.title(f"Recon Type: {cfg.recon.reconType}\n(WW={WW1}, WL={WL1})\n(views={views}, viewsPerRot={viewsPerRot}, ScanRange={ScanRange}°)")
# #plt.axis('off')

# # Second (bottom) plot
# plt.subplot(2, 1, 2)
# plt.imshow(window_image(recon_img, WW2, WL2), cmap='gray')
# plt.title(f"Recon Type: {cfg.recon.reconType}\n(WW={WW2}, WL={WL2})\n(views={views}, viewsPerRot={viewsPerRot}, ScanRange={ScanRange}°)")
# #plt.axis('off')

# plt.tight_layout()
# plt.show()

############### Plot Recon Result ###############
# Window Width (WW) and Level (WL):
# WW = 200 #
# WL = 40 #    
# WW = 200 #
# WL = -1000 #
# WW = 2000 #
# WL = -500 #

# Line Pair:
WW = 1100 #
WL = -500 #
# # SinoVision:
# WW = 1474 #
# WL = 250 #

# # MTF:
# WL = -550 #
# WW = 400 #

views, viewsPerRot = cfg.protocol.viewCount, cfg.protocol.viewsPerRotation
ScanRange = round(views/viewsPerRot * 360, 1)
FanAngle = np.arctan(cfg.scanner.detectorColSize/2/cfg.scanner.sdd)*2*cfg.scanner.detectorColCount * 180/np.pi
FanAngle = round(FanAngle,1)

def window_image(img, window_width, window_level):
    low = window_level - window_width / 2
    high = window_level + window_width / 2
    img_windowed = np.clip(img, low, high)
    return (img_windowed - low) / (high - low)  # scale to [0,1]

imgFname = "%s_%dx%dx%d.raw" %(ct.resultsName, ct.recon.imageSize, ct.recon.imageSize, ct.recon.sliceCount)
img = xc.rawread(imgFname, [ct.recon.sliceCount, ct.recon.imageSize, ct.recon.imageSize], 'float')

# Suppose recon_img_stack has shape [num_slices, height, width]
recon_img_stack = img  # e.g., img[ct.recon.sliceCount, :, :]

# Initial slice
slice_idx = ct.recon.sliceCount // 2  # middle slice
recon_img = recon_img_stack[slice_idx, :, :]
if cfg.recon.reconType == 'axial_short_scan':
    recon_img = np.flip(recon_img, axis=(0, 1))

# --- Plot setup ---
fig, ax = plt.subplots(figsize=(6, 6))
plt.subplots_adjust(bottom=0.15)  # leave space for slider

#im = ax.imshow(window_image(recon_img, WW, WL), cmap='gray')
im = ax.imshow(recon_img, cmap='gray')
title_text = f"Recon Type: {cfg.recon.reconType}\n(FanAngle={FanAngle}°, WW={WW}, WL={WL})\n(views={views}, viewsPerRot={viewsPerRot}, ScanRange={ScanRange}°)"
#ax.set_title(f"Slice: {slice_idx}")
ax.set_title(title_text)
ax.axis('off')

############## Save Image to .npy file for Post processing #################
spotSize = cfg.scanner.focalspotWidth
detectorSize = cfg.scanner.detectorColSize
spot_str = f"{spotSize:.2f}".replace('.', '_')
det_str  = f"{detectorSize:.2f}".replace('.', '_')
ScanRange_str = f"{ScanRange}".replace('.', '_')
filename = f"{cfg.recon.kernelType}_spot{spot_str}_detector{det_str}.npy"
np.save(filename, recon_img_stack)
############## Save Image to .npy file for Post processing #################

# --- Slider axis ---
ax_slider = plt.axes([0.2, 0.05, 0.6, 0.03])
slider = Slider(ax_slider, 'Slice', 0, ct.recon.sliceCount-1, valinit=slice_idx, valstep=1)

# --- Update function ---
def update(val):
    slice_idx = int(slider.val)
    recon_img = recon_img_stack[slice_idx, :, :]
    if cfg.recon.reconType == 'axial_short_scan':
        recon_img = np.flip(recon_img, axis=(0, 1))
    im.set_data(window_image(recon_img, WW, WL))
    ax.set_title(title_text)
    fig.canvas.draw_idle()

slider.on_changed(update)
plt.show()



###################### Calculate Mean and STD in ROI ######################
# Define ROI coordinates (y1:y2, x1:x2)
roi_coords = (220, 230, 295, 305)
# roi_coords = (190, 210, 180, 200)
#roi_coords = (150, 200, 150, 200)  # example 50x50 pixel region
#roi_coords = (200, 225, 200, 225)
#roi_coords = (110, 135, 250, 275)
y1, y2, x1, x2 = roi_coords
roi = recon_img[y1:y2, x1:x2]

mean_ct = np.mean(roi)
std_ct = np.std(roi)

# Plot with annotation
fig, ax = plt.subplots(figsize=(6, 6))

# Zoom in
# zoom_img = recon_img[230:496, 230:496] # Line Pair
# zoom_img = recon_img[270:410, 260:400] # Sinovision geometry
zoom_img = recon_img[200:760, 155:715] # Sinovision geometry, 1024 size
ax.imshow(window_image(zoom_img, WW, WL), cmap='gray')
#ax.imshow(window_image(recon_img, WW, WL), cmap='gray')

# #zoom_img = recon_img[100:624, 100:624] # Cylinder
# zoom_img = recon_img # Cylinder 2048
# ax.imshow(zoom_img, cmap='gray')

# rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, edgecolor='red', facecolor='none', lw=1.5)
# ax.add_patch(rect)

# # Mean and std:
# ax.text(10, 20, f"Mean CT = {mean_ct:.2f} HU\nSTD = {std_ct:.2f}", color='yellow',
#         fontsize=12, fontweight='bold', bbox=dict(facecolor='black', alpha=0.5))

# Label focal spot and voxel size
# ax.text(10, 20, f"Mean CT = {mean_ct:.2f} HU\nSTD = {std_ct:.2f}", color='yellow',
#         fontsize=12, fontweight='bold', bbox=dict(facecolor='black', alpha=0.5))


plt.title(
    'Recon Type: {}, Filter: {}\n(FocalSpot={}x{} mm, voxel={:.1f} mm), (WW={}, WL={})\n(views={}, viewsPerRot={}, ScanRange={}deg)'.format(
        cfg.recon.reconType,
        cfg.recon.kernelType,
        cfg.scanner.focalspotWidth, 
        cfg.scanner.focalspotLength, 
        cfg.recon.fov/cfg.recon.imageSize, 
        WW, WL, views, viewsPerRot, ScanRange))
ax.axis('off')

plt.tight_layout()
plt.show()


########## Plot Line across line pairs to see dege change#########
from scipy.ndimage import map_coordinates
# line endpoints in zoom_img coordinates
# x0, y0 = 98, 200
# x1, y1 = 205, 93
# # x0, y0 = 45, 157
# # x1, y1 = 151, 51

# # SinoVision, 1024 size
x0, y0 = 50, 470
x1, y1 = 490, 30

# # SinoVision, 1024 size
# x0, y0 = 20, 180
# x1, y1 = 185, 15

# draw line on image
ax.plot([x0, x1], [y0, y1], 'r-', linewidth=1.5)

plt.show()

# sample points along the line
num = int(np.hypot(x1 - x0, y1 - y0)) + 1
x = np.linspace(x0, x1, num)
y = np.linspace(y0, y1, num)

# extract grayscale values from displayed image
img_disp = window_image(zoom_img, WW, WL)
profile = map_coordinates(img_disp, [y, x], order=1)

# distance axis
distance = np.linspace(0, np.hypot(x1 - x0, y1 - y0), num)

# plot cross-section
plt.figure(figsize=(7, 4))
plt.plot(distance, profile, color='red')

# ax = plt.gca()
# ax.invert_xaxis()

plt.xlabel("Distance (pixels)")
plt.ylabel("Gray scale value")
plt.title("Cross-section Profile")
plt.grid(True)
plt.show()
########## Plot Line across line pairs to see edege change#########


# ### --- Plot histogram of ROI values ---
# plt.figure(figsize=(5, 4))
# plt.hist(roi.flatten(), bins=40, color='gray', edgecolor='black')
# plt.title("CT Number Distribution in ROI")
# plt.xlabel("CT Number (HU)")
# plt.ylabel("Pixel Count")
# plt.grid(True, linestyle='--', alpha=0.6)
# plt.tight_layout()
# plt.show()

