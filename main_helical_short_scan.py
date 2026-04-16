# Copyright 2024, GE Precision HealthCare. All rights reserved. See https://github.com/xcist/main/tree/master/license

###------------ import XCIST-CatSim
import numpy as np
import gecatsim as xc
import gecatsim.reconstruction.pyfiles.recon as recon
import time
import math

##--------- Initialize 
ct = xc.CatSim("phantom", "protocol", "scanner", "recon", "physics")  # initialization "example_physics"  will add more reality

##--------- Make changes to parameters (optional)
ct.resultsName = "out"

##--------- Run simulation
ct.run_all()  # run the scans defined by protocol.scanTypes



# ##--------- Reconstruction
cfg = ct.get_current_cfg()
# cfg.do_Recon = 1
# cfg.waitForKeypress = 0

# start_time = time.time()        # record start time

# recon.recon(cfg)                # run your reconstruction

# end_time = time.time()          # record end time

# elapsed = end_time - start_time
# print(f"Reconstruction completed in {elapsed:.2f} seconds.")

##--------- Show results
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider


# ################ Plot Sinogram ###############
# def compute_triangle_edges(YL, YLC, dDecL, ViewN, BetaS, DeltaFai, Gamma_m):
#     ScanRange = ViewN * DeltaFai  # total scan angle coverage
#     # Extra angular coverage beyond half-scan
#     # Delta = 2Gamma_virtual
#     Delta = ScanRange - (np.pi + 2 * Gamma_m)
#     Delta = max(0.0, Delta)  # ensure non-negative
    
#     edges_up = []     # (d, s_idx) for ramp-up boundary beta2
#     edges_down = []   # (d, s_idx) for ramp-down boundary beta3

#     for d in range(YL):
#         alpha = (d - YLC) * dDecL

#         # ---- same formulas you used ----
#         beta2 = 2 * (Gamma_m + Delta/2 - alpha)
#         #beta2 = 2 * (Gamma_m + (0) / 2 - alpha)   # your formula
#         beta3 = np.pi - 2 * alpha                 # your formula

#         # Convert beta → view index (s index)
#         s2 = int(round((beta2 - BetaS) / DeltaFai))
#         s3 = int(round((beta3 - BetaS) / DeltaFai))

#         # Only keep edges that fall inside valid range
#         if 0 <= s2 < ViewN:
#             edges_up.append((d, s2))
#         if 0 <= s3 < ViewN:
#             edges_down.append((d, s3))

#     return np.array(edges_up), np.array(edges_down)


# sino = xc.rawread(ct.resultsName+'.prep', [ct.protocol.viewCount, ct.scanner.detectorRowCount, ct.scanner.detectorColCount], 'float')
# sino = np.squeeze(sino[:, 0, :])
# xc.rawwrite(ct.resultsName+"_sino_%dx%d.raw"%(ct.scanner.detectorColCount, ct.protocol.viewCount), sino.copy(order="C"))

# sinoname = ct.resultsName+"_sino_%dx%d.raw"%(ct.scanner.detectorColCount, ct.protocol.viewCount)
# sinogram = xc.rawread(sinoname, [ct.protocol.viewCount, ct.scanner.detectorColCount], 'float')


# # define allowed angular sector per rotation (in degrees)
# def compute_beta_array(cfg):
#     # Extract parameters
#     startAngle = cfg.protocol.startAngle * math.pi/180.0      # radians
#     viewCount = cfg.protocol.viewCount
#     viewsPerRotation = cfg.protocol.viewsPerRotation
#     rotDir = cfg.protocol.rotationDirection                  # +1 or -1

#     # Compute beta per view
#     beta_array = startAngle + (np.arange(viewCount) / viewsPerRotation) * rotDir * math.tau
#     return beta_array


# # allowed_start = 0          # e.g., 0 deg
# # allowed_end   = cfg.protocol.coverage      # e.g., 230 deg

# # angle_mod = np.degrees(np.mod(compute_beta_array(cfg), 2*np.pi))

# # # mask views outside allowed sector
# # mask_allowed = (angle_mod >= allowed_start) & (angle_mod <= allowed_end)

# # sino_masked = sinogram.copy()
# # sino_masked[~mask_allowed, :] = 0.0      # or np.nan if you want NaNs

# ############ Try Graduate Trim Sinogram ####################
# fade_deg = 30.0  # width of fade near start/end in degrees (tune 5~30)
# def smoothstep(x):
#     # x in [0,1]
#     return x*x*(3 - 2*x)
# allowed_start = 0.0
# allowed_end   = float(cfg.protocol.coverage)  # e.g. 230

# angle_deg = np.degrees(np.mod(compute_beta_array(cfg), 2*np.pi))

# w_ang = np.zeros_like(angle_deg, dtype=np.float32)

# # fully-on region
# core = (angle_deg >= (allowed_start + fade_deg)) & (angle_deg <= (allowed_end - fade_deg))
# w_ang[core] = 1.0

# # ramp up near start
# ramp_up = (angle_deg >= allowed_start) & (angle_deg < (allowed_start + fade_deg))
# x = (angle_deg[ramp_up] - allowed_start) / fade_deg  # 0..1
# w_ang[ramp_up] = smoothstep(x).astype(np.float32)

# # ramp down near end
# ramp_dn = (angle_deg > (allowed_end - fade_deg)) & (angle_deg <= allowed_end)
# x = (allowed_end - angle_deg[ramp_dn]) / fade_deg     # 0..1
# w_ang[ramp_dn] = smoothstep(x).astype(np.float32)

# # apply to Proj (Proj is [view, YL, ZL] in your code at that moment)
# #Proj *= w_ang[:, None, None]
# ############ End Try Graduate Trim Sinogram #################### 
# sinogram*= w_ang[:, None]

# # Write back (overwrite)
# sinogram.astype(np.float32).tofile(sinoname)

# # write and plot
# plt.figure(figsize=(8,6))
# plt.imshow(sinogram, cmap='gray', aspect='auto')
# plt.title('Sinogram with masked/unavailable angle views (by view index)')
# plt.xlabel('Detector index')
# plt.ylabel('View index')
# plt.show()

cfg.do_Recon = 1
cfg.waitForKeypress = 0

recon.recon(cfg)                # run your reconstruction

############### Plot Recon Result ###############
# Window Width (WW) and Level (WL):
# WW = 200 #
# WL = 40 #    
# WW = 200 #
# WL = -1000 #
WW = 2000 #
WL = -500 #

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

# --- Plot setup ---
fig, ax = plt.subplots(figsize=(6, 6))
plt.subplots_adjust(bottom=0.15)  # leave space for slider

im = ax.imshow(window_image(recon_img, WW, WL), cmap='gray')
ax.set_title(f"Slice: {slice_idx}")
ax.axis('off')

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
    ax.set_title(f"Slice: {slice_idx}")
    fig.canvas.draw_idle()

slider.on_changed(update)
plt.show()
