import numpy as np
import time

# -------------------------------------------
# 1) SSD Distance (Sum of Squared Differences)
# -------------------------------------------
def ssd_distance(desc1, desc2):
    """
    Compute SSD distance between two descriptors.
    """
    return np.sum((desc1 - desc2) ** 2)

# -------------------------------------------
# 2) NCC (Normalized Cross Correlation)
# -------------------------------------------
def ncc_distance(desc1, desc2):
    """
    Compute distance based on 1 - NCC.
    Higher NCC means better match → Smaller distance.
    """
    d1 = desc1 - np.mean(desc1)
    d2 = desc2 - np.mean(desc2)

    denom = (np.linalg.norm(d1) * np.linalg.norm(d2)) + 1e-8
    ncc = np.sum(d1 * d2) / denom

    return 1 - ncc

# ------------------------------------------------
# 3) Match descriptors using SSD (Brute Force)
# ------------------------------------------------
def match_descriptors_ssd(descs1, descs2, threshold=float('inf'), max_matches=None):
    """
    For each descriptor in image1, find best match in image2 using SSD.
    Optional:
        threshold = ignore matches with score > threshold
        max_matches = limit number of matches returned
    """
    matches = []
    start = time.time()

    for i, d1 in enumerate(descs1):
        best_j = None
        best_score = float('inf')

        for j, d2 in enumerate(descs2):
            score = ssd_distance(d1, d2)
            if score < best_score:
                best_score = score
                best_j = j

        if best_score <= threshold:
            matches.append((i, best_j, best_score))

    matches = sorted(matches, key=lambda x: x[2])
    if max_matches is not None:
        matches = matches[:max_matches]

    elapsed = time.time() - start
    return matches, elapsed

# ------------------------------------------------
# 4) Match descriptors using NCC (Brute Force)
# ------------------------------------------------
def match_descriptors_ncc(descs1, descs2, threshold=float('inf'), max_matches=None):
    """
    For each descriptor in image1, find best match in image2 using NCC.
    Optional:
        threshold = ignore matches with score > threshold
        max_matches = limit number of matches returned
    """
    matches = []
    start = time.time()

    for i, d1 in enumerate(descs1):
        best_j = None
        best_score = float('inf')

        for j, d2 in enumerate(descs2):
            score = ncc_distance(d1, d2)
            if score < best_score:
                best_score = score
                best_j = j

        if best_score <= threshold:
            matches.append((i, best_j, best_score))

    matches = sorted(matches, key=lambda x: x[2])
    if max_matches is not None:
        matches = matches[:max_matches]

    elapsed = time.time() - start
    return matches, elapsed

# ------------------------------------------------
# 5) Full Matching Pipeline with Threshold & Max Matches
# ------------------------------------------------
def match_feature_sets(descs1, descs2, threshold=float('inf'), max_matches=None):
    """
    Perform SSD and NCC matching with optional threshold and max matches.
    Returns:
        ssd_matches, ncc_matches, ssd_time, ncc_time
    """
    ssd_matches, ssd_time = match_descriptors_ssd(descs1, descs2, threshold, max_matches)
    ncc_matches, ncc_time = match_descriptors_ncc(descs1, descs2, threshold, max_matches)

    return ssd_matches, ncc_matches, ssd_time, ncc_time