import cv2
import numpy as np

# Color ranges in HSV format (tuned for vivid plastic/wood blocks to prevent false positive desk matches)
COLOR_HSV_RANGES = {
    "red": [
        (np.array([0, 130, 90]), np.array([10, 255, 255])),
        (np.array([168, 130, 90]), np.array([180, 255, 255]))
    ],
    "green": [
        (np.array([35, 110, 80]), np.array([85, 255, 255]))
    ],
    "blue": [
        (np.array([95, 120, 80]), np.array([130, 255, 255]))
    ],
    "yellow": [
        (np.array([18, 120, 110]), np.array([35, 255, 255]))
    ],
    "orange": [
        (np.array([10, 130, 110]), np.array([22, 255, 255]))
    ],
    "purple": [
        (np.array([130, 90, 70]), np.array([160, 255, 255]))
    ]
}

def detect_colored_objects(crop_img, min_area=60, max_area=5000):
    """
    Detects colored cubes/objects in the calibrated Octo POV image.
    Rejects giant contours (> max_area) to prevent background wood/table false positives.
    Returns dict: {color_name: {"pixel": (u, v), "area": int, "phys_mm": (x_mm, y_mm)}}
    """
    if crop_img is None or crop_img.size == 0:
        return {}

    hsv = cv2.cvtColor(crop_img, cv2.COLOR_BGR2HSV)
    h, w = crop_img.shape[:2]
    detected = {}

    for color_name, ranges in COLOR_HSV_RANGES.items():
        mask = np.zeros((h, w), dtype=np.uint8)
        for lower, upper in ranges:
            m = cv2.inRange(hsv, lower, upper)
            mask = cv2.bitwise_or(mask, m)

        # Morphological opening to clean noise
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue

        # Filter contours by valid object area (blocks are between min_area and max_area)
        valid_contours = [c for c in contours if min_area <= cv2.contourArea(c) <= max_area]
        if not valid_contours:
            continue

        largest = max(valid_contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)
        M = cv2.moments(largest)
        if M["m00"] > 0:
            u = float(M["m10"] / M["m00"])
            v = float(M["m01"] / M["m00"])
            # Inverse homography pixel (u, v) to physical Dobot (X, Y) mm:
            # - Screen Left (u=0) is Robot Right (Y = -140 mm)
            # - Screen Right (u=w-1) is Robot Left (Y = +140 mm)
            # - Screen Top (v=0) is near Dobot base (X = 80 mm)
            # - Screen Bottom (v=h-1) is far table edge (X = 360 mm)
            ws_span_x = 280.0
            ws_span_y = 280.0
            x_min_mm = 80.0
            y_min_mm = -140.0

            y_mm = y_min_mm + (u / (w - 1.0)) * ws_span_y
            x_mm = x_min_mm + (v / (h - 1.0)) * ws_span_x

            detected[color_name] = {
                "pixel": (int(u), int(v)),
                "area": int(area),
                "phys_mm": (float(x_mm), float(y_mm))
            }

    return detected
