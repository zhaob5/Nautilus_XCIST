# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 08:40:04 2026

@author: Botao Zhao
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import math

def createHSP(Length, kernelType):
    HS = np.zeros(Length)
    Center = int((Length) / 2)
    PI = 3.14159265358979
    nn = Length
    nn2 = nn * 2
    
    if kernelType == "R-L":
        HS[0] = 0
        HS[Center] = 0.25
        for i in range(1, Center):
            HS[i] = -np.power((math.sin(PI * (i - Center) / 2)), 2) / (PI * PI * (i - Center) * (i - Center))

        for k in range((Center + 1), Length):
            HS[k] = -np.power((math.sin(PI * (k - Center) / 2)), 2) / (PI * PI * (k - Center) * (k - Center))
        k = int(nn / 2)
        TempF = np.zeros(nn2)
        TempF[0:k] = HS[k:nn]
        TempF[k + nn:nn2] = HS[0:k]
        HS = TempF * complex(0, 1)
        FFT_F = np.fft.fft(HS)

    elif kernelType == "S-L":
        for i in range(Length):
            HS[i] = -2 / (PI * PI * (4 * (i - Center) * (i - Center) - 1))
        k = int(nn / 2)
        TempF = np.zeros(nn2)
        TempF[0:k] = HS[k:nn]
        TempF[k + nn:nn2] = HS[0:k]
        HS = TempF * complex(0, 1)
        FFT_F = np.fft.fft(HS)

    elif kernelType == "soft":
        x = np.array([0, 0.25, 0.5, 0.75, 1])
        y = np.array([1, 0.815, 0.4564, 0.1636, 0])
        # y = np.array([1, 1.0485, 1.17, 1.2202, 0.9201])
        # y= np.array([1, 0.9338, 0.7441, 0.4425, 0.0531])
        f = interp1d(x, y, kind='quadratic')
        FFT_F = np.zeros(nn2)
        for i in range(nn):
            FFT_F[i] = f((i)/nn)*0.997 * (i+0.003) / nn2
            FFT_F[nn2 - i - 1] = f((i)/nn)* 0.997*(i + 1 + 0.003) / nn2
        FFT_F= FFT_F * complex(0,1)

    elif kernelType == "standard":
        x = np.array([0, 0.25, 0.5, 0.75, 1])
        # y = np.array([1, 0.815, 0.4564, 0.1636, 0])
        # y = np.array([1, 1.0485, 1.17, 1.2202, 0.9201])
        y = np.array([1, 0.9338, 0.7441, 0.4425, 0.0531])
        f = interp1d(x, y, kind='quadratic')
        FFT_F = np.zeros(nn2)
        for i in range(nn):
            FFT_F[i] = f((i) / nn) * 0.997 * (i + 0.003) / nn2
            FFT_F[nn2 - i - 1] = f((i) / nn) * 0.997 * (i + 1 + 0.003) / nn2
        FFT_F = FFT_F * complex(0, 1)

    elif kernelType == "bone":
        x = np.array([0, 0.25, 0.5, 0.75, 1])
        # y = np.array([1, 0.815, 0.4564, 0.1636, 0])
        y = np.array([1, 1.0485, 1.17, 1.2202, 0.9201])
        # y = np.array([1, 0.9338, 0.7441, 0.4425, 0.0531])
        f = interp1d(x, y, kind='quadratic')
        FFT_F = np.zeros(nn2)
        for i in range(nn):
            FFT_F[i] = f((i) / nn) * 0.997 * (i + 0.003) / nn2
            FFT_F[nn2 - i - 1] = f((i) / nn) * 0.997 * (i + 1 + 0.003) / nn2
        FFT_F = FFT_F * complex(0, 1)

    elif kernelType == "none":
        FFT_F = np.ones(nn2) * complex(0, 1)

    # elif kernelType == "edgeplus":
    #     FFT_F = np.zeros(nn2)
        
    #     beta = 2.0   # high-frequency tilt strength
    
    #     for i in range(nn):
    #         u = i / nn
    #         weight = 1.0 + beta * u
    
    #         FFT_F[i] = weight * 0.997 * (i + 0.003) / nn2
    #         FFT_F[nn2 - i - 1] = weight * 0.997 * (i + 1 + 0.003) / nn2
    
    #     FFT_F = FFT_F * complex(0, 1)
    elif kernelType == "edgeplus":
        x = np.array([0, 0.25, 0.5, 0.75, 1])
        y = np.array([1, 1.0485, 1.17, 1.3, 1.5])
        f = interp1d(x, y, kind='quadratic')
        FFT_F = np.zeros(nn2)
        for i in range(nn):
            FFT_F[i] = f((i) / nn) * 0.997 * (i + 0.003) / nn2
            FFT_F[nn2 - i - 1] = f((i) / nn) * 0.997 * (i + 1 + 0.003) / nn2
        FFT_F = FFT_F * complex(0, 1)

    else: 
        raise Exception("******** Error! An unsupported kernel was specified: {:s}. ********".format(kernelType))

    return FFT_F

YL = 544
detector_spacing = 1.08   # mm
nn = int(math.pow(2, (math.ceil(math.log2(abs(YL))) + 1)))

H_rl = np.imag(createHSP(nn, "R-L"))
H_sl = np.imag(createHSP(nn, "S-L"))
H_soft = np.imag(createHSP(nn, "soft"))
H_standard = np.imag(createHSP(nn, "standard"))
H_bone = np.imag(createHSP(nn, "bone"))
H_edge = np.imag(createHSP(nn, "edgeplus"))

n = len(H_rl) // 2
H_rl = H_rl[:n]
H_sl = H_sl[:n]
H_soft = H_soft[:n]
H_standard = H_standard[:n]
H_bone = H_bone[:n]
H_edge = H_edge[:n]

x = np.linspace(0, 1, n, endpoint=False)   # normalized frequency

plt.figure(figsize=(8,5))
plt.plot(x, H_rl, label="R-L")
plt.plot(x, H_sl, label="S-L")
plt.plot(x, H_soft, label="soft")
plt.plot(x, H_standard, label="standard")
plt.plot(x, H_bone, label="bone")
plt.plot(x, H_edge, label="edgeplus")
plt.xlabel("Normalized frequency")
plt.ylabel("Filter value")
#plt.title("R-L vs edgeplus")
plt.grid(True)
plt.legend()
plt.show()