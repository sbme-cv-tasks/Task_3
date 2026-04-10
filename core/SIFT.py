import numpy as np
import cv2
import time
import sys
from pathlib import Path

# =========================
# 1. Gaussian Blur
# =========================
def gaussian_blur(img, ksize=5, sigma=1.0):
    if ksize == 0:
        # Auto-calculate kernel size to ensure enough coverage for the sigma
        ksize = int(2 * np.ceil(3 * sigma)) + 1
    return cv2.GaussianBlur(img, (ksize, ksize), sigma)


# =========================
# 2. Build Gaussian Pyramid
# =========================
def build_gaussian_pyramid(img, num_octaves=4, num_scales=5):
    pyramid = []
    k = 2 ** (1.0 / (num_scales - 1))  
    sigma0 = 1.6
    
    for o in range(num_octaves):
        octave = []
        sigma = sigma0
        for s in range(num_scales):
            blurred = gaussian_blur(img, ksize=0, sigma=sigma)
            octave.append(blurred)
            sigma *= k
        pyramid.append(octave)
        #  Use INTER_LINEAR for downsampling to maintain better feature quality
        h, w = img.shape
        img = cv2.resize(octave[-1], (w // 2, h // 2), interpolation=cv2.INTER_LINEAR)
    return pyramid


# =========================
# 3. Difference of Gaussian (DoG)
# =========================
def build_dog_pyramid(gaussian_pyramid):
    dog_pyramid = []
    for octave in gaussian_pyramid:
        dog_octave = []
        for i in range(1, len(octave)):
            dog = octave[i] - octave[i - 1]
            dog_octave.append(dog)
        dog_pyramid.append(dog_octave)
    return dog_pyramid


# =========================
# 4. Keypoint Detection
# =========================
def is_extrema(patch):
    """Check if center point is a local extremum (max or min) in 3x3x3 neighborhood"""
    center = patch[1, 1, 1]
    neighbors = patch.flatten()
    # Remove the center element to compare only with the 26 neighbors
    neighbors = np.delete(neighbors, 13) 
    
    is_max = center > np.max(neighbors)
    is_min = center < np.min(neighbors)
    return is_max or is_min


def is_edge_response(curr_img, i, j, r_threshold=10):
    """Eliminate edge responses using Hessian matrix trace/det ratio"""
    Dxx = curr_img[i, j+1] + curr_img[i, j-1] - 2*curr_img[i, j]
    Dyy = curr_img[i+1, j] + curr_img[i-1, j] - 2*curr_img[i, j]
    Dxy = (curr_img[i+1, j+1] - curr_img[i+1, j-1] - 
           curr_img[i-1, j+1] + curr_img[i-1, j-1]) / 4
    
    trace = Dxx + Dyy
    det = Dxx * Dyy - Dxy ** 2
    
    if det <= 0:
        return True  
    
    r = (trace ** 2) / det
    return r > ((r_threshold + 1) ** 2 / r_threshold)


def detect_keypoints(dog_pyramid, threshold=0.03):
    keypoints = []
    for o, octave in enumerate(dog_pyramid):
        for s in range(1, len(octave) - 1):
            prev_img = octave[s - 1]
            curr_img = octave[s]
            next_img = octave[s + 1]
            h, w = curr_img.shape

            #  Optimized neighborhood search to reduce redundant calculations
            for i in range(1, h - 1):
                for j in range(1, w - 1):
                    if abs(curr_img[i, j]) <= threshold:
                        continue

                    patch = np.stack([
                        prev_img[i-1:i+2, j-1:j+2],
                        curr_img[i-1:i+2, j-1:j+2],
                        next_img[i-1:i+2, j-1:j+2]
                    ])

                    if is_extrema(patch):
                        if not is_edge_response(curr_img, i, j):
                            keypoints.append((o, s, i, j))
    return keypoints


# =========================
# 5. SIFT Part 2: Feature Descriptors
# =========================

def compute_gradients(image):
    dx = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)
    dy = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = np.sqrt(dx**2 + dy**2)
    orientation = np.degrees(np.arctan2(dy, dx)) % 360
    return magnitude, orientation


def assign_orientation(magnitude, orientation, kp, scale, num_bins=36, window_factor=1.5):
    o, s, i, j = kp
    h, w = magnitude.shape
    #  Window size adjusted to 3 * sigma for standard SIFT orientation stability
    radius = int(3 * window_factor * scale)
    sigma_w = window_factor * scale
    hist = np.zeros(num_bins)

    for di in range(-radius, radius + 1):
        for dj in range(-radius, radius + 1):
            ni, nj = i + di, j + dj
            if 0 <= ni < h and 0 <= nj < w:
                weight = np.exp(-(di**2 + dj**2) / (2 * sigma_w**2))
                bin_idx = int(orientation[ni, nj] / (360 / num_bins)) % num_bins
                hist[bin_idx] += weight * magnitude[ni, nj]

    peak = np.max(hist)
    dominant_angles = [b * (360 / num_bins) for b in range(num_bins) if hist[b] >= 0.8 * peak]
    return dominant_angles if dominant_angles else [np.argmax(hist) * (360 / num_bins)]


def compute_descriptor(magnitude, orientation, kp, dominant_angle, scale, num_spatial_bins=4, num_orient_bins=8):
    o, s, i, j = kp
    h, w = magnitude.shape
    
    #  Standardized window size to 16x16 for a consistent 128-D vector
    window_size = 16 
    half_w = window_size // 2
    descriptor = np.zeros((num_spatial_bins, num_spatial_bins, num_orient_bins))

    cos_a = np.cos(np.radians(-dominant_angle))
    sin_a = np.sin(np.radians(-dominant_angle))

    for di in range(-half_w, half_w):
        for dj in range(-half_w, half_w):
            ni, nj = i + di, j + dj
            if 0 <= ni < h and 0 <= nj < w:
                # Rotate coordinates for rotation invariance
                rot_i = cos_a * di + sin_a * dj
                rot_j = -sin_a * di + cos_a * dj
                
                #  Map relative pixel position to one of the 4x4 spatial bins
                bin_i = int((rot_i + half_w) / (window_size / num_spatial_bins))
                bin_j = int((rot_j + half_w) / (window_size / num_spatial_bins))

                if 0 <= bin_i < num_spatial_bins and 0 <= bin_j < num_spatial_bins:
                    rel_angle = (orientation[ni, nj] - dominant_angle) % 360
                    orient_bin = int(rel_angle / (360 / num_orient_bins)) % num_orient_bins
                    
                    #  Apply Gaussian weight to prioritize center samples
                    weight = np.exp(-(di**2 + dj**2) / (2 * (half_w)**2))
                    descriptor[bin_i, bin_j, orient_bin] += magnitude[ni, nj] * weight

    desc = descriptor.flatten()
    #  Normalize, clip at 0.2, and re-normalize to ensure illumination robustness
    norm = np.linalg.norm(desc)
    if norm > 1e-7:
        desc /= norm
    desc = np.clip(desc, 0, 0.2)
    norm = np.linalg.norm(desc)
    if norm > 1e-7:
        desc /= norm
    return desc.astype(np.float32)


def extract_descriptors(keypoints, gaussian_pyramid, sigma0=1.6, num_scales=5, return_orientations=False):
    start = time.time()
    k = 2 ** (1.0 / (num_scales - 1))
    descriptors = []
    valid_keypoints = []
    valid_orientations = []

    for kp in keypoints:
        o, s, i, j = kp
        if o >= len(gaussian_pyramid) or s >= len(gaussian_pyramid[o]):
            continue

        scale_image = gaussian_pyramid[o][s]
        scale = sigma0 * (k ** s)
        mag, ori = compute_gradients(scale_image)
        angles = assign_orientation(mag, ori, kp, scale)

        for angle in angles:
            desc = compute_descriptor(mag, ori, kp, angle, scale)
            descriptors.append(desc)
            valid_keypoints.append(kp)
            valid_orientations.append(angle)

    elapsed = time.time() - start
    descriptors_array = np.array(descriptors) if descriptors else np.empty((0, 128))
    if return_orientations:
        return descriptors_array, valid_keypoints, elapsed, valid_orientations
    return descriptors_array, valid_keypoints, elapsed


# =========================
# 6. Pipeline & Visualization
# =========================
def draw_keypoints(img, keypoints):
    output = img.copy()
    for kp in keypoints:
        o, s, i, j = kp
        scale = 2 ** o  
        x, y = int(j * scale), int(i * scale)
        if not (0 <= x < output.shape[1] and 0 <= y < output.shape[0]):
            continue
        radius = max(3, int(scale * 2))
        cv2.circle(output, (x, y), radius, (0, 255, 0), 2)
        cv2.circle(output, (x, y), 1, (0, 0, 255), -1)
    return output


def draw_keypoints_with_orientation(img, keypoints, orientations, scale_factor=8.0):
    output = img.copy()

    for kp, angle in zip(keypoints, orientations):
        o, _, i, j = kp

        scale = 2 ** o
        x = int(j * scale)
        y = int(i * scale)

        if not (0 <= x < output.shape[1] and 0 <= y < output.shape[0]):
            continue

        theta = np.radians(angle)
        arrow_len = max(4, int(scale_factor * scale))
        x2 = int(x + arrow_len * np.cos(theta))
        y2 = int(y - arrow_len * np.sin(theta))

        cv2.circle(output, (x, y), 3, (0, 255, 0), -1)
        cv2.arrowedLine(output, (x, y), (x2, y2), (0, 0, 255), 2, tipLength=0.35)

    return output


def run_sift_pipeline(image, num_octaves=4, show_orientation=False):
    """Full execution logic with time reporting"""
    start_total = time.time()
    
    # Pre-processing
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    
    # Steps
    gauss_pyr = build_gaussian_pyramid(gray, num_octaves=num_octaves)
    dog_pyr = build_dog_pyramid(gauss_pyr)
    
    keypoints = detect_keypoints(dog_pyr)

    _, valid_kps, _, valid_orientations = extract_descriptors(
        keypoints,
        gauss_pyr,
        return_orientations=True,
    )
    output_img = (
        draw_keypoints_with_orientation(image, valid_kps, valid_orientations)
        if show_orientation
        else draw_keypoints(image, keypoints)
    )
    
    total_time = time.time() - start_total
    
    # print(f"\n--- SIFT Execution Report ---")
    # print(f"Description Time: {desc_time:.4f}s")
    # print(f"Total Time: {total_time:.4f}s")
    # print(f"Keypoints Detected: {len(keypoints)}")
    # print(f"Keypoints Used In Descriptors: {len(valid_kps)}")
    
    return keypoints, output_img, total_time
