"""
Calibration helper: measure the actual cm_per_pixel for your camera setup.

Usage:
    python scripts/calibrate_cm_per_pixel.py

Steps:
    1. Run this script - it opens your camera.
    2. Place a known-width object (cup=8.5cm, phone=7.2cm, A4 paper=21cm) in view.
    3. Type the number shown on screen to use that object for calibration.
    4. Measure the REAL distance (with a ruler) from hand to target.
    5. Read the screen to see the recommended cm_per_pixel value.
"""

import cv2
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from config import AppConfig
from detector import create_detector


KNOWN_OBJECTS = {
    "1": ("cup", 10),
    "2": ("cell phone", 7.2),
    "3": ("paper (A4)", 21.0),
}

print("=" * 60)
print(" cm_per_pixel 校准工具")
print("=" * 60)
print("\n可选的已知宽度物体：")
for key, (name, width) in KNOWN_OBJECTS.items():
    print(f"  按 {key} = {name} (宽度={width}cm)")
print("\n将物体放在摄像头前，按对应数字键开始校准")
print("按 q 退出\n")

config = AppConfig()
detector = create_detector(config.detector_backend, config.target_labels, config.detection_confidence)

cap = cv2.VideoCapture(config.camera_index)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.frame_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.frame_height)

if not cap.isOpened():
    print("错误：无法打开摄像头")
    sys.exit(1)

selected_object = None

while True:
    ok, frame = cap.read()
    if not ok:
        break

    display = frame.copy()
    detections = detector.detect_all(frame, restrict_to_allowed=True)

    # Show instructions
    cv2.putText(display, "Press 1/2/3 to select calibration object, q to quit",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    if selected_object:
        label, real_width = selected_object
        matching = [d for d in detections if d["label"] == label]
        if matching:
            det = matching[0]
            x1, y1, x2, y2 = det["bbox"]
            bbox_w = x2 - x1
            bbox_h = y2 - y1
            ref_px = max(bbox_w, bbox_h)
            measured_cmpp = real_width / ref_px if ref_px > 10 else 0

            cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(display, f"{label}: {ref_px:.0f}px -> {measured_cmpp:.4f} cm/px",
                        (x1, max(30, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            cv2.putText(display, f"RECOMMENDED: approx_cm_per_pixel = {measured_cmpp:.4f}",
                        (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(display, f"(Update config.py line 16: approx_cm_per_pixel={measured_cmpp:.4f})",
                        (10, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 100), 1)
        else:
            cv2.putText(display, f"Looking for: {label} ...",
                        (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    else:
        for i, (key, (name, width)) in enumerate(KNOWN_OBJECTS.items()):
            cv2.putText(display, f"Press {key}: {name} ({width}cm)",
                        (10, 70 + i * 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

    cv2.imshow("Calibration", display)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break
    elif chr(key) in KNOWN_OBJECTS:
        selected_object = KNOWN_OBJECTS[chr(key)]
        print(f"\n已选择: {selected_object[0]} (宽度={selected_object[1]}cm)")
        print("确保物体在画面中显示完整...\n")

cap.release()
cv2.destroyAllWindows()

if selected_object:
    print(f"\n{'='*60}")
    print("校准结果")
    print(f"{'='*60}")
    print(f"选择的物体: {selected_object[0]}")
    print(f"物体真实宽度: {selected_object[1]}cm")
    print(f"\n将 config.py 第16行的:")
    print(f"  approx_cm_per_pixel: float = 0.18")
    print(f"改为:")
    print(f"  approx_cm_per_pixel: float = {selected_object[2]:.4f}")
    print(f"\n注意：这是近似值，建议用尺子实测验证")
