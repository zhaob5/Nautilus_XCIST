# XCIST Real-Data Reconstruction
---

## Disclaimer

This work is based on the open-source [XCIST](https://github.com/xcist) (X-ray-based Cancer Imaging Simulation Toolkit) framework.   
This repository contains modifications and extensions for real-data reconstruction and CUDA acceleration.

---

## 1. Overview

This project extends the XCIST pipeline to support:

- Reconstruction from **real CT sinogram data**
- Use of **measured source/detector trajectories**
- Optional **CUDA acceleration** for backprojection

### Key additions:
- Short-scan (<360°) reconstruction
- Real trajectory integration (per-view geometry)
- GPU-accelerated FBP

---

## 2. Repository Structure

```
gecatsim/
+-- reconstruction/
¦   +-- pyfiles/
¦   ¦   +-- axial_short_scan_realData.py		# main Python recon pipeline
¦   ¦
¦   +-- src/
¦   ¦   +-- Axial_Short_Scan_realData.cu		# CUDA backprojection
¦   ¦
¦   +-- lib/
¦   ¦   +-- Axial_Short_Scan_CUDA_realData.dll		# Compiled .dll file from .cu
¦
main scripts:
+-- reconFromRealData.py          # entry point
+-- protocol.cfg                  # configuration file
+-- physics.cfg                   # configuration file
+-- scanner.cfg                   # configuration file
+-- recon.cfg                   	# configuration file
```

---

## 3. Dependencies

### Python
- numpy
- matplotlib
- ctypes
- scipy

### GPU
- CUDA Toolkit
- Visual Studio C++ build tools

---

## 4. Input Data

### 4.1 Sinogram

```python
prep = np.load("...sinogram.npy")
```

Expected shape:

```
(ViewN, DetectorRows, DetectorCols)
```

Example:

```
(737, 32, 864)
```

---

### 4.2 Trajectory Data

```python
data = np.load("trajectory.npz")

src = data['array2']   # (ViewN, 2)
det = data['array3']   # (ViewN, 2)
geo = data['array1']   # optional
```

---

## 5. Configuration Files (`cfg`) Setup

The reconstruction pipeline uses configuration (`cfg`) files from XCIST to define system geometry, detector properties, and reconstruction settings.

These are typically loaded via:

```python
ct = xc.CatSim("phantom", "protocol", "scanner", "recon", "physics")
cfg = ct.get_current_cfg()
```

---

### 5.1 Key Configuration Categories

The `cfg` object contains several groups of parameters:

#### Scanner
- Detector size (`ColSize`, `rowSize`)
- Source-to-detector distance (SDD)
- Source-to-isocenter distance (SID)

#### Protocol
- Number of views (`ViewN`)
- Rotation direction
- Scan angle / short-scan setup

#### Recon
- Image size (`imageSize`)
- Number of slices (`sliceCount`)
- Slice thickness
- Reconstruction kernel

#### Physics
- Simulation-specific parameters (used mainly for synthetic data)

---

### 5.2 Behavior for Real Data Reconstruction

For **real-data reconstruction (this workflow)**:

- The following parameters are **overridden by measured trajectory data**:
  - Source position (SID per view)
  - Detector position (SDD per view)
  - View angles

- Therefore, these cfg parameters are **less critical**:
  - `ScanR`, `DistD`
  - `BetaS`, `BetaE`
  - Uniform angular spacing assumptions

However, the following parameters are still used and must be set correctly:

- Detector geometry:
  - `ColSize`, `rowSize`
  - Number of detector rows/columns
- Image grid:
  - `imageSize`
  - `sliceCount`
  - `sliceThickness`
- Offsets:
  - `centerOffset`
  - detector offsets

---

### 5.3 Behavior for Simulation

For **simulation using XCIST (synthetic data)**:

- All geometry parameters in `cfg` are **actively used**
- The system assumes **ideal circular geometry**
- The following must be carefully tuned:

  - Source-to-isocenter distance (`ScanR`)
  - Source-to-detector distance (`DistD`)
  - Angular sampling (`ViewN`, `BetaS`, `BetaE`)
  - Detector spacing and offsets
  - Reconstruction parameters

Incorrect values may lead to:
- Distorted reconstructions
- Incorrect scaling
- Artifacts

---

### 5.4 Practical Recommendation

- **Real data workflow**
  - Focus on:
    - trajectory input (`src`, `det`)
    - sinogram alignment
    - detector geometry
  - Treat cfg geometry as **secondary**

- **Simulation workflow**
  - Carefully configure all geometry parameters in `cfg`
  - Ensure consistency between:
    - scanner
    - protocol
    - reconstruction settings

---

### 5.5 Summary

| Parameter Type        | Real Data | Simulation |
|----------------------|----------|-----------|
| Trajectory (SID/SDD) |  Used   |  Ignored |
| View angles          |  Used   |  Derived |
| cfg geometry         |  Partial |  Critical |
| Detector settings    |  Required |  Required |
| Image grid           |  Required |  Required |

---

This distinction is important:  
**Real-data reconstruction relies primarily on measured geometry, while simulation relies entirely on cfg-defined geometry.**


### 5.6 Geometry Processing

### Convert trajectory to parameters

```python
def compute_view_geometry(
    src,
    det,
    iso_center=(0.0, 0.0),
    angle_offset_deg=0.0,
    view_shift=0
):
    import numpy as np

    src = np.asarray(src, dtype=np.float64)
    det = np.asarray(det, dtype=np.float64)
    iso = np.asarray(iso_center, dtype=np.float64)

    src_rel = src - iso

    SID = np.linalg.norm(src_rel, axis=1)
    SDD = np.linalg.norm(det - src, axis=1)

    view_angle_rad = np.arctan2(src_rel[:, 1], src_rel[:, 0])
    view_angle_rad = np.unwrap(view_angle_rad)

    view_angle_rad += np.deg2rad(angle_offset_deg)

    if view_shift != 0:
        view_angle_rad = np.roll(view_angle_rad, view_shift)
        SID = np.roll(SID, view_shift)
        SDD = np.roll(SDD, view_shift)

    return view_angle_rad, SID, SDD
```

---

## 6. Reconstruction Pipeline

### Step 1 — Load sinogram

```python
prep = np.load(file_path).astype(np.float32)
```

Optional:

```python
prep = prep[:, :, start:end]
```

---

### Step 2 — Reconstruction

```python
rec = axial_short_scan_realData(cfg, prep, SID, SDD, view_angle_rad)
```

---

### Step 3 — Noise reduction

```python
img = np.mean(rec[:, :, 1:-1], axis=2)
```

---

### Step 4 — Visualization

```python
plt.imshow(img, cmap='gray')
plt.title("Reconstruction from Real Data")
plt.show()
```

---

## 7. Real Trajectory Integration

### Ideal geometry (old)

```
View = BetaS + i * DeltaFai
ScanR = constant
DistD = constant
```

### Real geometry (new)

```cpp
float View  = ViewAngleArray[i];
float ScanR = SIDArray[i];
float DistD = SDDArray[i];
```

---

## 8. Python to CUDA Data Flow

```python
SID = np.ascontiguousarray(SID, dtype=np.float32)
SDD = np.ascontiguousarray(SDD, dtype=np.float32)
view_angle = np.ascontiguousarray(view_angle_rad, dtype=np.float32)

t.SIDArray = SID.ctypes.data_as(PtrFLOAT)
t.SDDArray = SDD.ctypes.data_as(PtrFLOAT)
t.ViewAngleArray = view_angle.ctypes.data_as(PtrFLOAT)
```

---

## 9. CUDA Reconstruction

### File

```
Axial_Short_Scan_realData.cu
```

### Key struct addition

```cpp
float* SIDArray;
float* SDDArray;
float* ViewAngleArray;
```

---

### Backprojection

```cpp
float View  = ViewAngleArray[ProjInd];
float ScanR = SIDArray[ProjInd];
float DistD = SDDArray[ProjInd];
```

---

## 10. Compile CUDA to DLL

### Step 1 — Open build environment

Open **"x64 Native Tools Command Prompt for Visual Studio"**.

---

### Step 2 — Navigate to source folder

```bash
cd /d "F:\Shared drives\Nautilus Engineering\XCIST\Nautilus_XCIST\gecatsim\reconstruction\src"
```

---

### Step 3 — Compile `.cu` to `.dll`

```bash
nvcc -shared -o Axial_Short_Scan_CUDA_realData.dll \
    Axial_Short_Scan_realData.cu \
    -Xcompiler "/MD" -arch=sm_61
```

---

### Step 4 — Move DLL to library folder

After compilation, move the generated `.dll` file to:

```
gecatsim/reconstruction/lib/
```

Final expected location:

```
gecatsim/reconstruction/lib/Axial_Short_Scan_CUDA_realData.dll
```

---

### Notes

- Ensure CUDA Toolkit and Visual Studio C++ build tools are installed
- If you modify the `.cu` file, you must recompile the `.dll`
- Architecture flag (`-arch=sm_61`) should match your GPU
- Ignore warnings about older architectures unless compilation fails

---

### Important: DLL Name Consistency

If you change the output DLL name during compilation, you must also update it in:

```
axial_short_scan_realData.py
```

Specifically inside the function:

```python
def load_C_recon_lib():
    recon_lib = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../lib")

    ll = ct.cdll.LoadLibrary
    if os.name == "nt":
        lib_file = "Axial_Short_Scan_CUDA_realData.dll"

    clib = ll(os.path.join(recon_lib, lib_file))
    return clib
```

Make sure `lib_file` matches the compiled `.dll` filename exactly, otherwise the reconstruction code will fail to load the library.

## 11. Run Workflow

```python
prep = np.load(...)
src, det = load_trajectory(...)

view_angle, SID, SDD = compute_view_geometry(src, det)

rec = axial_short_scan_realData(cfg, prep, SID, SDD, view_angle)

img = np.mean(rec[:, :, 1:-1], axis=2)

plt.imshow(img, cmap='gray')
```
