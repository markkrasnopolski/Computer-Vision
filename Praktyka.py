import cv2
import subprocess
import json
import numpy as np
import torch
from collections import Counter
from ultralytics import YOLO

YOUTUBE_URL = "https://www.youtube.com/live/Lxqcg1qt0XU?si=C3FyLUnHHySL10Ou"


def get_stream_url(url):
    cmd = ["yt-dlp", "-j", "-f", "best[ext=mp4]", url]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return json.loads(result.stdout)["url"]


cap = cv2.VideoCapture(get_stream_url(YOUTUBE_URL))
if not cap.isOpened():
    raise RuntimeError("Не вдалося відкрити відео")

fps = cap.get(cv2.CAP_PROP_FPS) or 30
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(f"Роздільна здатність: {frame_width}x{frame_height}, FPS: {fps}")

device = 0 if torch.cuda.is_available() else "cpu"
print("DEVICE:", device)


model = YOLO("yolov8x.pt").to(device)

CONF_THRESH = 0.3
IOU_THRESH = 0.5

VEHICLE_CLASSES = {
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck"
}

meter_per_pixel = 3 / 120


COUNT_Y = int(frame_height * 0.6)
counted_ids = set()
total_counter = Counter()
prev_centers = {}
frame_count = 0

tracking_buffer = {}


def match_detections_to_tracks(detections, prev_tracks, max_distance=100):
    """Прив'язує нові детекції до старих треків на основі відстані"""
    matches = {}

    for det_idx, det in enumerate(detections):
        best_match = None
        min_dist = max_distance

        for track_id, track_data in prev_tracks.items():
            prev_cx, prev_cy = track_data[0], track_data[1]
            dist = np.hypot(det[0] - prev_cx, det[1] - prev_cy)

            if dist < min_dist:
                min_dist = dist
                best_match = track_id

        if best_match is not None:
            matches[det_idx] = best_match

    return matches


while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    if frame_width > 1280:
        scale = 1280 / frame_width
        new_width = 1280
        new_height = int(frame_height * scale)
        frame = cv2.resize(frame, (new_width, new_height))
        frame_width, frame_height = new_width, new_height
        COUNT_Y = int(frame_height * 0.6)

    results = model(
        frame,
        conf=CONF_THRESH,
        iou=IOU_THRESH,
        verbose=False,
        classes=list(VEHICLE_CLASSES.keys()),
        device=device
    )

    r = results[0]

    current_centers = {}
    current_detections = []

    if r.boxes is not None:
        boxes = r.boxes
        xyxy = boxes.xyxy.cpu().numpy()
        class_ids = boxes.cls.cpu().numpy()

        has_tracking = hasattr(boxes, 'id') and boxes.id is not None

        if has_tracking:
            track_ids = boxes.id.cpu().numpy()
        else:
            track_ids = np.arange(len(xyxy)) * -1 - 1

        for i in range(len(xyxy)):
            class_id = int(class_ids[i])
            if class_id not in VEHICLE_CLASSES:
                continue

            x1, y1, x2, y2 = xyxy[i].astype(int)

            width = x2 - x1
            height = y2 - y1
            if width < 30 or height < 30:
                continue

            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            class_name = VEHICLE_CLASSES[class_id]

            current_detections.append((cx, cy, x1, y1, x2, y2, class_id, class_name, i))

        if prev_centers and not has_tracking:
            detections_centers = [(d[0], d[1]) for d in current_detections]
            matches = match_detections_to_tracks(detections_centers, prev_centers)

            for det_idx, track_id in matches.items():
                det = current_detections[det_idx]
                cx, cy, x1, y1, x2, y2, class_id, class_name, orig_idx = det
                tid = track_id
                current_centers[tid] = (cx, cy, class_name)

                if tid in prev_centers:
                    prev_cy = prev_centers[tid][1]
                    if prev_cy <= COUNT_Y < cy and tid not in counted_ids:
                        counted_ids.add(tid)
                        total_counter[class_name] += 1
                        print(f"✅ Пораховано: {class_name} (ID: {tid}) - Всього: {total_counter}")
                    elif prev_cy >= COUNT_Y > cy and tid not in counted_ids:
                        counted_ids.add(tid)
                        total_counter[class_name] += 1
                        print(f"✅ Пораховано: {class_name} (ID: {tid}) - Всього: {total_counter}")

                speed = 0.0
                if tid in prev_centers:
                    dx = cx - prev_centers[tid][0]
                    dy = cy - prev_centers[tid][1]
                    distance = np.hypot(dx, dy)
                    speed = distance * meter_per_pixel * fps * 3.6

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                speed_color = (0, 255, 0) if speed < 60 else (0, 255, 255) if speed < 90 else (0, 0, 255)

                cv2.putText(
                    frame,
                    f"{class_name} | {speed:.0f} km/h",
                    (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    speed_color,
                    2
                )

            for det_idx, det in enumerate(current_detections):
                if det_idx not in matches:
                    cx, cy, x1, y1, x2, y2, class_id, class_name, orig_idx = det
                    new_id = f"new_{frame_count}_{det_idx}_{cx}_{cy}"
                    current_centers[new_id] = (cx, cy, class_name)

                    # ===== DRAW =====
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(
                        frame,
                        f"{class_name} | NEW",
                        (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (255, 255, 0),
                        2
                    )

        else:
            for i in range(len(xyxy)):
                if i >= len(current_detections):
                    continue

                det = current_detections[i]
                cx, cy, x1, y1, x2, y2, class_id, class_name, orig_idx = det

                if has_tracking:
                    tid = int(track_ids[orig_idx])
                else:
                    tid = f"det_{frame_count}_{i}_{cx}_{cy}"

                current_centers[tid] = (cx, cy, class_name)

                # ===== COUNT =====
                if tid in prev_centers:
                    prev_cy = prev_centers[tid][1]
                    if prev_cy <= COUNT_Y < cy and tid not in counted_ids:
                        counted_ids.add(tid)
                        total_counter[class_name] += 1
                        print(f"✅ Пораховано: {class_name} (ID: {tid}) - Всього: {total_counter}")
                    elif prev_cy >= COUNT_Y > cy and tid not in counted_ids:
                        counted_ids.add(tid)
                        total_counter[class_name] += 1
                        print(f"✅ Пораховано: {class_name} (ID: {tid}) - Всього: {total_counter}")

                speed = 0.0
                if tid in prev_centers:
                    dx = cx - prev_centers[tid][0]
                    dy = cy - prev_centers[tid][1]
                    distance = np.hypot(dx, dy)
                    speed = distance * meter_per_pixel * fps * 3.6

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                speed_color = (0, 255, 0) if speed < 60 else (0, 255, 255) if speed < 90 else (0, 0, 255)

                cv2.putText(
                    frame,
                    f"{class_name} | {speed:.0f} km/h",
                    (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    speed_color,
                    2
                )

    prev_centers = {k: (v[0], v[1]) for k, v in current_centers.items()}

    if len(tracking_buffer) > 100:
        tracking_buffer.clear()

    for tid in list(counted_ids):
        if tid not in prev_centers and tid in tracking_buffer:
            tracking_buffer[tid] = tracking_buffer.get(tid, 0) + 1
            if tracking_buffer[tid] > 30:
                counted_ids.discard(tid)
                del tracking_buffer[tid]

    panel_height = 80 + 26 * len(VEHICLE_CLASSES)
    cv2.rectangle(frame, (5, 5), (320, panel_height), (0, 0, 0), -1)

    y = 30
    cv2.putText(frame, "TRAFFIC COUNT", (10, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)
    y += 28

    for class_name in VEHICLE_CLASSES.values():
        count = total_counter[class_name]
        cv2.putText(frame, f"{class_name}: {count}", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        y += 24

    total = sum(total_counter.values())
    cv2.putText(frame, f"TOTAL: {total}", (10, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    y += 24

    cv2.putText(frame, f"Active: {len(prev_centers)}", (10, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    cv2.imshow("Traffic AI (YOLOv8m)", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    elif key == ord("r"):
        counted_ids.clear()
        total_counter.clear()
        prev_centers.clear()
        tracking_buffer.clear()
        print("🔄 Лічильник скинуто")

cap.release()
cv2.destroyAllWindows()

print("\n" + "=" * 40)
print("ФІНАЛЬНИЙ ЗВІТ")
print("=" * 40)
for class_name in VEHICLE_CLASSES.values():
    count = total_counter[class_name]
    print(f"{class_name:12}: {count:3}")
print("=" * 40)
print(f"{'TOTAL':12}: {sum(total_counter.values()):3}")
print("=" * 40)