import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy.ndimage import map_coordinates, center_of_mass, binary_erosion
from skimage import img_as_float, restoration, filters
from scipy.fft import fft, fftfreq
from scipy.signal import savgol_filter

image_path = 'F:\\Shared drives\\Nautilus Engineering\\XCIST\\Simulation_Results_Mach_3_5\\Sinovision_MTF_phantom\\MTF.png'
#'G:\\Shared drives\\Nautilus Engineering\\Recon\\Mach-3.5\\code\\Images\\'

def detect_circle_hough(img_norm, threshold=0.5, dp=1.2, min_dist=50,
                        param1=100, param2=30, min_radius=10, max_radius=0,
                        morph_kernel_size=5, morph_iters=2):
    """
    Detect circle in the image using Hough Transform after preprocessing with morphological operations.

    Args:
        img_norm: Normalized input image (2D numpy array).
        threshold: Threshold for binary mask creation.
        dp: Inverse ratio of the accumulator resolution to the image resolution.
        min_dist: Minimum distance between detected centers.
        param1: Upper threshold for the Canny edge detector.
        param2: Accumulator threshold for circle detection.
        min_radius: Minimum circle radius.
        max_radius: Maximum circle radius.
        morph_kernel_size: Size of the morphological operation kernel.
        morph_iters: Number of iterations for morphological operations.
   """     
   # --- Step 1: Create mask ---
    mask = (img_norm > threshold).astype(np.uint8) * 255

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                       (morph_kernel_size, morph_kernel_size))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=morph_iters)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=morph_iters)
    mask = cv2.GaussianBlur(mask, (5, 5), 0)

    plt.figure(figsize=(6, 6))
    plt.imshow(mask, cmap='gray')

    # --- Step 2: Hough circle detection ---
    circles = cv2.HoughCircles(mask, cv2.HOUGH_GRADIENT, dp=dp, minDist=min_dist,
                               param1=param1, param2=param2,
                               minRadius=min_radius, maxRadius=max_radius)
    if circles is None:
        # raise RuntimeError("No circles detected in mask.")
        return None, None
    circles = np.round(circles[0, :]).astype(int)
    circles = sorted(circles, key=lambda c: c[2], reverse=True)
    x0, y0, r = circles[0]
    print(f"Detected circle: center=({x0}, {y0}), radius={r}")

    return x0, y0, r

def detect_circle_simple(img_norm, threshold=0.5):
    """
    Detect circle in the image using simple centroid and radius estimation after thresholding.

    Args:
        img_norm: Normalized input image (2D numpy array).
        threshold: Threshold for binary mask creation.
    """     
    # --- Step 1: Create mask ---
    mask = (img_norm > threshold).astype(np.uint8)

    plt.figure(figsize=(6, 6))
    plt.imshow(mask, cmap='gray')

    # # --- Step 2: Find contours and estimate circle ---
    # contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # if len(contours) == 0:
    #     return None, None, None

    # largest_contour = max(contours, key=cv2.contourArea)
    # (x0, y0), r = cv2.minEnclosingCircle(largest_contour)
    # x0, y0, r = int(x0), int(y0), int(r)
    # print(f"Detected circle (simple): center=({x0}, {y0}), radius={r}")

    y0, x0 = center_of_mass(mask)
    eroded = binary_erosion(mask)
    boundary = mask ^ eroded
    ys, xs = np.nonzero(boundary)
    distances = np.sqrt((xs - x0)**2 + (ys - y0)**2) 
    r = int(np.mean(distances))

    return x0, y0, r

def normalize_for_mtf(img):
    img = img.astype(np.float32)

    # Remove low-frequency pedestal (background)
    p0, p99 = np.percentile(img, (0.5, 99.5))  # you can tweak to 1/99 if needed
    img = np.clip(img, p0, p99)
    img = (img - p0) / (p99 - p0)

    return img

def sharpen_for_mtf(image, psf_sigma=1.0, iterations=20):

    # Convert and normalize
    image = normalize_for_mtf(image)

    # Approximate PSF from Gaussian (isotropic optics blur)
    size = int(2 * np.ceil(3 * psf_sigma) + 1)
    x = np.linspace(-(size // 2), size // 2, size)
    psf = np.exp(-(x**2) / (2 * psf_sigma**2))
    psf /= psf.sum()

    # 2D separable PSF
    psf_2d = np.outer(psf, psf)

    # Run Richardson-Lucy deconvolution
    deconv = restoration.richardson_lucy(image, psf_2d, num_iter=iterations)

    # Light denoise to suppress high-frequency noise
    deconv = filters.median(deconv)

    return deconv

def compute_mtf_aapm(image, pixel_size=1.0, threshold=0.5, dp=1.2, min_dist=50,
                     param1=100, param2=30, min_radius=10, max_radius=0,
                     morph_kernel_size=5, morph_iters=2, dr=0.1, smooth=True, smooth_window=15, smooth_poly=3):
    """
    Compute MTF using an AAPM-like edge method with mask preprocessing.
    """

    # Normalize image
    img_norm = (image - image.min()) / (image.max() - image.min())

    plt.figure(figsize=(6, 6))
    plt.imshow(img_norm, cmap='gray')
    plt.title('Normalized Image')

    # x0, y0, r = detect_circle_hough(img_norm, threshold=threshold, dp=dp, min_dist=min_dist,
    #                                param1=param1, param2=param2,
    #                                min_radius=min_radius, max_radius=max_radius,
    #                                morph_kernel_size=morph_kernel_size, morph_iters=morph_iters)

    x0, y0, r = detect_circle_simple(img_norm, threshold=threshold)

     # Visualization of detected circle

    # --- Step 3: Collect multiple ESFs around the circle ---
    n_samples = 360
    profiles = []

    for theta in np.linspace(0, np.pi, n_samples, endpoint=False):
        dx, dy = np.cos(theta), np.sin(theta)
        N_samples = int(r / dr)
        line = np.linspace(-r//2, r//2, N_samples)
        xs = x0 + (r + line) * dx
        ys = y0 + (r + line) * dy
        profile = map_coordinates(img_norm, [ys, xs], order=1, mode='reflect')
        profiles.append(profile)

    plt.figure(figsize=(6, 6))
    plt.imshow(img_norm, cmap='gray')
    plt.scatter([x0], [y0], color='red', s=100, label='Detected Center')
    circ = plt.Circle((x0, y0), r, color='yellow', fill=False, linewidth=2)
    plt.gca().add_artist(circ)
    plt.plot(xs, ys, 'cyan', linewidth=1, label='Sampling Line')
    plt.title('Detected Circle on Mask')


    esf_raw = np.mean(profiles, axis=0)

    # --- Step 4: Oversampling ESF (binning method) ---
    x = np.arange(len(esf_raw))
    esf = np.interp(x, x, esf_raw)

    plt.figure(figsize=(10, 4))
    plt.plot(esf_raw, 'kx', label='Raw ESF')
    plt.plot(x, esf, label='Oversampled ESF', alpha=0.7)
    plt.title('Edge Spread Function (ESF)')
    plt.xlabel('Pixel Index (subsampled)')
    plt.ylabel('Intensity')
    plt.legend()

    # --- Step 5: LSF ---
    lsf = np.gradient(esf)
    peak_idx = np.argmax(np.abs(lsf))  # index of largest peak (edge)
    half_width = int(0.1 * len(lsf))  # 20% of LSF length, tune as needed
    start = max(peak_idx - half_width, 0)
    end   = min(peak_idx + half_width, len(lsf))
    lsf_cropped = lsf[start:end]

    plt.figure(figsize=(10, 4))
    plt.plot(lsf, label='Line Spread Function (LSF)', color='orange')
    plt.title('Line Spread Function (LSF)')
    plt.xlabel('Subpixel Index')
    plt.ylabel('LSF Amplitude')

    # --- Step 6: MTF ---
    mtf = np.abs(fft(lsf_cropped))
    mtf /= np.max(mtf)  # normalize

    if smooth:
        if smooth_window >= len(mtf):
            smooth_window = len(mtf) - 1 if len(mtf) % 2 == 0 else len(mtf)
        if smooth_window % 2 == 0:
            smooth_window += 1
        mtf = savgol_filter(mtf, smooth_window, smooth_poly)

    # Frequency axis in line pairs / cm
    n_points = len(lsf_cropped)
    freqs = fftfreq(n_points, d=dr)  # cycles/mm
    freqs = freqs[:n_points//2] * 10 / pixel_size   # cycles/cm (line pairs/cm)

    plt.figure(figsize=(10, 6))
    plt.plot(freqs, mtf[:n_points//2], 'r', linewidth=3,label='MTF')
    plt.title('Modulation Transfer Function (MTF)')
    plt.xlabel('Spatial Frequency (line pairs/cm)')
    plt.ylabel('MTF')
    plt.grid(True)
    plt.xlim(0, 20)
    plt.legend()

    plt.show()

    return freqs, mtf[:n_points//2]

if __name__ == '__main__':
    data = np.load(image_path + 'mtf-10-29-2025-with-corrections-10slices-10voxel-0sweep-300iterations-xtalk.npy')
    # Select the middle slice for MTF calculation   
    mtfs = []
    freqs = []
    # slice_index = data.shape[0] // 2
    start_slice = 0
    end_slice = 9
    skips = [13]

    idx = np.arange(start_slice, end_slice)
    idx = np.setdiff1d(idx, skips)
    image_ave = np.mean(data[idx], axis=0)

    print(np.min(image_ave), np.max(image_ave), np.mean(image_ave))
    image_ave[image_ave > 300] = 300 # clip high values for better segmeting
    image_ave[image_ave < 0] = 0 # clip high values for better segmeting
    
    # for slice_index in range(start_slice, end_slice):
    #     image_slice = data[slice_index, :, :]

    print(image_ave.dtype, image_ave.min(), image_ave.max(), image_ave.mean())

    plt.figure(figsize=(6, 6))
    plt.imshow(image_ave, cmap='gray')  
    plt.title('Averaged Image for MTF Calculation')

    image_ave = sharpen_for_mtf(image_ave, psf_sigma=0.6, iterations=15)    

    plt.figure(figsize=(6, 6))
    plt.imshow(image_ave, cmap='gray') 
    plt.title('Averaged and Sharpened Image for MTF Calculation')   


    print("Calculating MTF...")
    freq, mtf = compute_mtf_aapm(image_ave, pixel_size=1.0, param2=30, threshold=0.2, morph_kernel_size=5, dr=0.1, smooth=False, smooth_window=5)
    if mtf is None or freq is None:
            print(f"Skipping slice due to no detected circle.")
            # continue
    # mtfs.append(mtf)
    # freqs.append(freq)
    print("Done.")

    # plt.figure(figsize=(10, 6))
    # for i, mtf in enumerate(mtfs):
    #     plt.plot(freqs[i], mtf, label=f'Slice {i}')
    # plt.title('MTF for All Slices')
    # plt.xlabel('Spatial Frequency (line pairs/cm)')
    # plt.ylabel('MTF')
    # plt.grid(True)
    # plt.xlim(0, 20)
    # plt.legend()

    np.savez(image_path + 'mtf_results_10-29-2025-with-corrections-10slices-10voxel-0sweep-300iterations-xtalk.npz', freq=freq, mtf=mtf)

    plt.show()