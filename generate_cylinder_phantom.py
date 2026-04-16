# -*- coding: utf-8 -*-
"""
Simple solid cylinder phantom
Material: polyethylene_LDPE
Dia = 240 mm
Length = 10 mm
"""

import numpy as np
import matplotlib.pyplot as plt

pixel_size = 0.1   # mm
diameter_mm = 240.0
radius_mm = diameter_mm / 2.0
size_mm = 260.0

N = int(round(size_mm / pixel_size))

x = (np.arange(N) + 0.5) * pixel_size
y = (np.arange(N) + 0.5) * pixel_size
X, Y = np.meshgrid(x, y)

cx = size_mm / 2.0
cy = size_mm / 2.0

r = np.sqrt((X - cx)**2 + (Y - cy)**2)

ldpe = (r <= radius_mm).astype(np.float32)

plt.figure(figsize=(6, 6))
plt.imshow(np.flipud(ldpe), cmap='gray', extent=[0, size_mm, 0, size_mm])
plt.xlabel("mm")
plt.ylabel("mm")
plt.title("Solid LDPE Cylinder Phantom")
plt.axis("equal")
plt.show()

air = 1.0 - ldpe
air = air.astype(np.float32)
air.tofile("cylinder_air_2600x2600x1.raw")
ldpe.tofile("cylinder_polyethylene_LDPE_2600x2600x1.raw")