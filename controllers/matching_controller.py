import cv2
import numpy as np
from time import time
from PySide6.QtCore import Qt

from core.SIFT import (
    build_gaussian_pyramid,
    build_dog_pyramid,
    detect_keypoints,
    extract_descriptors,
)
from core.matcher import (
    match_descriptors_ncc,
    match_descriptors_ssd,
)
from utils.converters import cv_to_pixmap


class MatchingController:
    def __init__(self, ui, model):
        self.ui = ui
        self.model = model
        self._connect_signals()

    def _connect_signals(self):
        self.ui.btnMatchSSD.clicked.connect(self.run_match_ssd)
        self.ui.btnMatchNCC.clicked.connect(self.run_match_ncc)

    def _get_image_from_label(self, label):
        image_path = label.property("imagePath")
        if not image_path:
            return None
        image = cv2.imread(image_path)
        return image

    def _load_match_images(self):
        img1 = self._get_image_from_label(self.ui.lblMatchImage1)
        img2 = self._get_image_from_label(self.ui.lblMatchImage2)
        if img1 is None or img2 is None:
            self.ui.statusbar.showMessage("Please load both match images.", 3000)
            return None, None
        return img1, img2

    def _compute_descriptors(self, image):
        if image is None:
            return np.empty((0, 128), dtype=np.float32), []
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        gaussian_pyramid = build_gaussian_pyramid(gray)
        dog_pyramid = build_dog_pyramid(gaussian_pyramid)
        keypoints = detect_keypoints(dog_pyramid)
        descriptors, valid_keypoints, _ = extract_descriptors(keypoints, gaussian_pyramid)
        return descriptors, valid_keypoints

    def _to_image_point(self, kp):
        octave, _, i, j = kp
        scale = 2 ** octave
        return int(j * scale), int(i * scale)

    def _draw_matches(self, img1, img2, keypoints1, keypoints2, matches):
        threshold_value = self.ui.matchThreshold.value()
        matches = [m for m in matches if m[2] <= threshold_value]
        max_matches = max(1, min(int(self.ui.matchCount.value()), len(matches)))
        draw_matches = matches[:max_matches]

        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        output = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
        output[:h1, :w1] = img1
        output[:h2, w1:w1 + w2] = img2

        for match in draw_matches:
            i, j, _ = match
            if i is None or j is None:
                continue
            pt1 = self._to_image_point(keypoints1[i])
            pt2 = self._to_image_point(keypoints2[j])
            cv2.circle(output, (pt1[0], pt1[1]), 3, (0, 255, 0), -1)
            cv2.circle(output, (pt2[0] + w1, pt2[1]), 3, (0, 255, 0), -1)
            cv2.line(output, (pt1[0], pt1[1]), (pt2[0] + w1, pt2[1]), (255, 0, 0), 1)
        return output

    def _run_matching(self, method):
        img1, img2 = self._load_match_images()
        if img1 is None or img2 is None:
            return

        descs1, keypoints1 = self._compute_descriptors(img1)
        descs2, keypoints2 = self._compute_descriptors(img2)

        if descs1.size == 0 or descs2.size == 0:
            self.ui.statusbar.showMessage("Could not extract descriptors from one or both images.", 3000)
            return

        start_time = time()

        if method == "ssd":
            matches, _ = match_descriptors_ssd(descs1, descs2)
        else:
            matches, _ = match_descriptors_ncc(descs1, descs2)

        matches = sorted(matches, key=lambda x: x[2])

        result_image = self._draw_matches(img1, img2, keypoints1, keypoints2, matches)
        pixmap = cv_to_pixmap(result_image)
        if pixmap is not None:
            self.ui.lblMatchResult.setPixmap(
                pixmap.scaled(
                    self.ui.lblMatchResult.size(),
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation,
                )
            )

        elapsed = time() - start_time
        self.ui.lblTimeValueMatch.setText(f"{elapsed:.3f} S")
        self.ui.statusbar.showMessage(f"Matching completed using {method.upper()}.", 3000)

        if not matches:
            self.ui.statusbar.showMessage("No matches found.", 3000)

    def run_match_ssd(self):
        self._run_matching("ssd")

    def run_match_ncc(self):
        self._run_matching("ncc")