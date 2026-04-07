import numpy as np
import cv2
import time
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.converters import cv_to_pixmap

# =========================
# 1. Gaussian Blur
# =========================
def gaussian_blur(img, ksize=5, sigma=1.0):
    if ksize == 0:
        # Auto calculate kernel size based on sigma
        ksize = int(2 * np.ceil(3 * sigma)) + 1
    return cv2.GaussianBlur(img, (ksize, ksize), sigma)


# =========================
# 2. Build Gaussian Pyramid
# =========================
def build_gaussian_pyramid(img, num_octaves=4, num_scales=5):
    pyramid = []
    k = 2 ** (1.0 / (num_scales - 1))  # SIFT factor
    sigma0 = 1.6
    
    for o in range(num_octaves):
        octave = []
        sigma = sigma0
        for s in range(num_scales):
            blurred = gaussian_blur(img, ksize=0, sigma=sigma)
            octave.append(blurred)
            sigma *= k
        pyramid.append(octave)
        img = cv2.pyrDown(img)  # downsample for next octave
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
    
    # Get all 26 neighbors (flattened patch minus center)
    neighbors = patch.flatten()
    neighbors = neighbors[neighbors != center]
    
    if len(neighbors) == 0:
        return False
    
    is_max = center > np.max(neighbors)
    is_min = center < np.min(neighbors)
    
    return is_max or is_min


def is_edge_response(curr_img, i, j, r_threshold=10):
    """Eliminate edge responses - حذف keypoints على الحواف"""
    Dxx = curr_img[i, j+1] + curr_img[i, j-1] - 2*curr_img[i, j]
    Dyy = curr_img[i+1, j] + curr_img[i-1, j] - 2*curr_img[i, j]
    Dxy = (curr_img[i+1, j+1] - curr_img[i+1, j-1] - 
           curr_img[i-1, j+1] + curr_img[i-1, j-1]) / 4
    
    trace = Dxx + Dyy
    det = Dxx * Dyy - Dxy ** 2
    
    if det <= 0:
        return True  # discard
    
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

            for i in range(1, h - 1):
                for j in range(1, w - 1):

                    # 3x3x3 cube
                    patch = np.stack([
                        prev_img[i-1:i+2, j-1:j+2],
                        curr_img[i-1:i+2, j-1:j+2],
                        next_img[i-1:i+2, j-1:j+2]
                    ])

                    if is_extrema(patch):
                        if abs(curr_img[i, j]) > threshold:
                            if not is_edge_response(curr_img, i, j):
                                keypoints.append((o, s, i, j))

    return keypoints


# =========================
# 5. Full Pipeline
# =========================
def sift_part1(image_path):
    start_time = time.time()

    # Read image
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = gray.astype(np.float32) / 255.0

    # Gaussian Pyramid
    gaussian_pyramid = build_gaussian_pyramid(gray)

    # DoG Pyramid
    dog_pyramid = build_dog_pyramid(gaussian_pyramid)

    # Detect keypoints
    keypoints = detect_keypoints(dog_pyramid)

    end_time = time.time()

    print("Number of keypoints:", len(keypoints))
    print("Detection Time:", end_time - start_time)

    return keypoints, img


def run_sift_pipeline(image, num_octaves=4):
    start_time = time.time()

    if image is None:
        raise ValueError("Input image is None")

    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    gray = gray.astype(np.float32) / 255.0

    gaussian_pyramid = build_gaussian_pyramid(gray, num_octaves=num_octaves)
    dog_pyramid = build_dog_pyramid(gaussian_pyramid)
    keypoints = detect_keypoints(dog_pyramid)
    output = draw_keypoints(image, keypoints)

    elapsed_time = time.time() - start_time
    return keypoints, output, elapsed_time


# =========================
# 6. Draw Keypoints
# =========================
def draw_keypoints(img, keypoints):
    output = img.copy()

    for kp in keypoints:
        o, s, i, j = kp
        scale = 2 ** o  
        x = int(j * scale)
        y = int(i * scale)
        radius = max(2, scale * 2)
        cv2.circle(output, (x, y), radius, (0, 255, 0), 1)

    return output

 
# =========================
# SIFT Part 2: Feature Descriptors
# =========================

# =========================
# 1. Gradients
# =========================
def compute_gradients(image):
    dx = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)
    dy = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3)

    magnitude = np.sqrt(dx**2 + dy**2)
    orientation = np.degrees(np.arctan2(dy, dx)) % 360

    return magnitude, orientation


# =========================
# 2. Orientation Assignment
# =========================
def assign_orientation(magnitude, orientation, kp, scale,
                       num_bins=36, window_factor=1.5):

    o, s, i, j = kp
    h, w = magnitude.shape

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

    dominant_angles = []
    for b in range(num_bins):
        if hist[b] >= 0.8 * peak:
            angle = b * (360 / num_bins)
            dominant_angles.append(angle)

    return dominant_angles if dominant_angles else [np.argmax(hist) * (360 / num_bins)]


# =========================
# 3. Descriptor 
# =========================
def compute_descriptor(magnitude, orientation, kp, dominant_angle,
                       scale, num_spatial_bins=4, num_orient_bins=8):

    o, s, i, j = kp
    h, w = magnitude.shape

    window_size = num_spatial_bins * scale * 2
    half_w = int(window_size / 2)

    descriptor = np.zeros((num_spatial_bins,
                           num_spatial_bins,
                           num_orient_bins), dtype=np.float32)

    cos_a = np.cos(np.radians(-dominant_angle))
    sin_a = np.sin(np.radians(-dominant_angle))

    cell_size = window_size / num_spatial_bins

    for di in range(-half_w, half_w + 1):
        for dj in range(-half_w, half_w + 1):

            ni, nj = i + di, j + dj

            if not (0 <= ni < h and 0 <= nj < w):
                continue

            # rotate coordinates
            rot_i = cos_a * di + sin_a * dj
            rot_j = -sin_a * di + cos_a * dj

            # spatial bin 
            bin_i = int((rot_i + half_w) / cell_size)
            bin_j = int((rot_j + half_w) / cell_size)

            if not (0 <= bin_i < num_spatial_bins and 0 <= bin_j < num_spatial_bins):
                continue

            # rotation invariant angle
            rel_angle = (orientation[ni, nj] - dominant_angle) % 360

            # HARD orientation bin
            orient_bin = int(rel_angle / (360 / num_orient_bins)) % num_orient_bins

            weight = np.exp(-(di**2 + dj**2) / (2 * (half_w / 2)**2))

            descriptor[bin_i, bin_j, orient_bin] += magnitude[ni, nj] * weight

    # flatten to 128-D
    desc = descriptor.flatten()

    # normalize
    norm = np.linalg.norm(desc)
    if norm > 1e-8:
        desc = desc / norm

    return desc.astype(np.float32)


# =========================
# 4. Full Pipeline
# =========================
def extract_descriptors(keypoints, gaussian_pyramid,
                        sigma0=1.6, num_scales=5):

    start = time.time()
    k = 2 ** (1.0 / (num_scales - 1))

    descriptors = []
    valid_keypoints = []

    for kp in keypoints:
        o, s, i, j = kp

        # check bounds
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

    elapsed = time.time() - start

    descriptors_array = (
        np.array(descriptors, dtype=np.float32)
        if descriptors else np.empty((0, 128), dtype=np.float32)
    )

    return descriptors_array, valid_keypoints, elapsed










# def compute_descriptor(magnitude, orientation, kp,
#                               dominant_angle,
#                               num_spatial_bins=4,
#                               num_orient_bins=8):

#     i, j = kp[2], kp[3]  

#     descriptor = np.zeros((num_spatial_bins,
#                            num_spatial_bins,
#                            num_orient_bins))

#     h, w = magnitude.shape

#     patch_size = 8  
#     half = patch_size // 2

#     for di in range(-half, half):
#         for dj in range(-half, half):

#             ni, nj = i + di, j + dj

#             if ni < 0 or ni >= h or nj < 0 or nj >= w:
#                 continue

#             bin_i = (di + half) * num_spatial_bins // patch_size
#             bin_j = (dj + half) * num_spatial_bins // patch_size

#             rel_angle = orientation[ni, nj] - dominant_angle
#             rel_angle = rel_angle % 360

#             orient_bin = int(rel_angle // (360 / num_orient_bins))

#             descriptor[bin_i, bin_j, orient_bin] += magnitude[ni, nj]

#     desc = descriptor.flatten()

#     # normalization
#     norm = np.linalg.norm(desc)
#     if norm > 0:
#         desc = desc / norm

#     return desc