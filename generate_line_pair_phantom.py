# # -*- coding: utf-8 -*-
# """
# Created on Mon Mar  2 16:37:11 2026

# @author: Botao Zhao
# """

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import rotate

pixel_size = 0.05  # mm
size_mm = 125
N = int(size_mm / pixel_size)

# LDPE background = 1 everywhere
ldpe = np.ones((N, N), dtype=np.float32)
air  = np.zeros((N, N), dtype=np.float32)

def add_linepair_block(img, center_x, center_y,
                       block_width_mm, block_height_mm,
                       lp_width, n_lines, pixel_size):
    period_mm = block_width_mm / n_lines
    if lp_width > period_mm:
        raise ValueError("lp_width is larger than the inferred period (block_width_mm / n_lines).")

    half_w = int(block_width_mm / 2 / pixel_size)
    half_h = int(block_height_mm / 2 / pixel_size)

    cx = int(center_x / pixel_size)
    cy = int(center_y / pixel_size)

    x0 = cx - half_w
    x1 = cx + half_w
    y0 = cy - half_h
    y1 = cy + half_h

    for x in range(x0, x1):
        local_x_mm = (x - x0) * pixel_size
        if (local_x_mm % period_mm) < lp_width:
            for y in range(y0, y1):
                img[y, x] = 1.0   # mark as "air slit"

def add_air_square(img, center_x_mm, center_y_mm, side_mm, pixel_size):
    half = side_mm / 2.0
    cx = int(round(center_x_mm / pixel_size))
    cy = int(round(center_y_mm / pixel_size))
    half_px = int(round(half / pixel_size))

    x0 = cx - half_px
    x1 = cx + half_px
    y0 = cy - half_px
    y1 = cy + half_px

    img[y0:y1, x0:x1] = 1.0

x_init = 30
y_init = 50

# draw slits into the AIR mask
add_linepair_block(air, x_init,      y_init, 11.5, 12.7, lp_width=1.5,  n_lines=5, pixel_size=pixel_size)
add_linepair_block(air, x_init+18.6, y_init,  9.0, 10.0, lp_width=1.0,  n_lines=5, pixel_size=pixel_size)
add_linepair_block(air, x_init+35.6, y_init,  7.5,  8.0, lp_width=0.9,  n_lines=5, pixel_size=pixel_size)
add_linepair_block(air, x_init+51.0, y_init,  5.5,  6.4, lp_width=0.65, n_lines=5, pixel_size=pixel_size)
add_linepair_block(air, x_init+64.7, y_init,  4.5,  5.0, lp_width=0.5,  n_lines=5, pixel_size=pixel_size)

# add 1x1 mm reference point
add_air_square(air, center_x_mm=x_init, center_y_mm=y_init+10, side_mm=1.0, pixel_size=pixel_size)
add_air_square(air, center_x_mm=x_init+18.6, center_y_mm=y_init+10, side_mm=1.0, pixel_size=pixel_size)
add_air_square(air, center_x_mm=x_init+35.6, center_y_mm=y_init+10, side_mm=1.0, pixel_size=pixel_size)
add_air_square(air, center_x_mm=x_init+51.0, center_y_mm=y_init+10, side_mm=1.0, pixel_size=pixel_size)
add_air_square(air, center_x_mm=x_init+64.7, center_y_mm=y_init+10, side_mm=1.0, pixel_size=pixel_size)

# rotate both masks the same way
air  = rotate(air,  angle=45, reshape=False, order=0)   # use order=1 if you want partial volume
air  = (air > 0.5).astype(np.float32)                   # keep binary after rotation (remove if order=1)
ldpe = 1.0 - air

# #preview (air should be white if you plot it)
# plt.imshow(air, cmap='gray', origin='lower')
# plt.title("Digital Line Pair Phantom")
# plt.colorbar()
# plt.show()
plt.imshow(1 - np.fliplr(air), cmap='gray', origin='lower',
           extent=[0, size_mm, 0, size_mm])
plt.xlabel("mm")
plt.ylabel("mm")
plt.title("Digital Line Pair Phantom")
#plt.colorbar()
plt.show()

# save raw files (update names to match materials!)
# air.tofile("lp_air_2500x2500x1.raw")
# ldpe.tofile("lp_polyethylene_LDPE_2500x2500x1.raw")