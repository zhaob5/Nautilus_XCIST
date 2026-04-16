# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 09:37:08 2026

@author: Botao Zhao
"""

import numpy as np
import matplotlib.pyplot as plt

data = np.load(r"F:\Shared drives\Nautilus Engineering\XCIST\Nautilus_XCIST\realData\pin_results_Mpphan1.npz")

print(data.files)
print(data['array1'].shape)
print(data['array2'].shape)
print(data['array3'].shape)

plt.figure()

# Phantom pins
geo = data['array1']
plt.scatter(geo[:,0], geo[:,1], label="Phantom pins")

# Source trajectory
src = data['array2']
plt.plot(src[:,0], src[:,1], '-o', label="Source")

# Detector trajectory
det = data['array3']
plt.plot(det[:,0], det[:,1], '-o', label="Detector")

plt.axis('equal')
plt.title("CT Geometry Overview")
plt.legend()
plt.show()