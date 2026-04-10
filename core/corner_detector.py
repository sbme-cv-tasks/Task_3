import time

import cv2
import numpy as np


def corner_detector(input_img, k = 0.04, window_size = 5, threshold = 0.01,lambda_minus_flag = False):
    start_total = time.time()
    corner_list = []

    if len(input_img.shape) == 3:
        gray_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
    else:
        gray_img = input_img.copy()
    output_img = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2RGB)

    offset = int(window_size / 2)
    y_range = gray_img.shape[0] - offset
    x_range = gray_img.shape[1] - offset

    dy, dx = np.gradient(gray_img.astype(float))

    Ixx = dx ** 2
    Ixy = dy * dx
    Iyy = dy ** 2

    response_map = np.zeros_like(gray_img, dtype=float)

    for y in range(offset, y_range):
        for x in range(offset, x_range):

            # Values of sliding window
            start_y = y - offset
            end_y = y + offset + 1
            start_x = x - offset
            end_x = x + offset + 1

            # The variable names are representative to
            # the variable of the Harris corner equation
            windowIxx = Ixx[start_y: end_y, start_x: end_x]
            windowIxy = Ixy[start_y: end_y, start_x: end_x]
            windowIyy = Iyy[start_y: end_y, start_x: end_x]

            # Sum of squares of intensities of partial derevatives
            Sxx = windowIxx.sum()
            Sxy = windowIxy.sum()
            Syy = windowIyy.sum()

            # Calculate determinant and trace of the matrix
            det = (Sxx * Syy) - (Sxy ** 2)
            trace = Sxx + Syy
            
            if  lambda_minus_flag:
                discriminant = (trace**2) - (4 * det)
                # Prevent negative square roots due to floating point inaccuracies
                if discriminant < 0:
                    discriminant = 0
                # Calculate minimum eigenvalue (lambda minus)
                r = (trace - np.sqrt(discriminant)) / 2.0
            
            else:#Harris
                
                r = det - k * (trace ** 2)


            response_map[y, x] = r

    max_response = response_map.max()
    dynamic_threshold = threshold* max_response

    # Non-Maximum Suppression step
    for y in range(offset, y_range):
        for x in range(offset, x_range):
            r = response_map[y, x]

            if r > dynamic_threshold:

                neighborhood = response_map[y - 1: y + 2, x - 1: x + 2]

                if r == neighborhood.max():
                    corner_list.append([x, y, r])
                    cv2.circle(output_img, (x, y), radius=3, color=(0, 0, 255), thickness=-1)

    total_time = time.time() - start_total
    return corner_list, output_img,total_time