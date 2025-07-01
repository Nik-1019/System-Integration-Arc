from ultralytics import YOLO
import cv2
import math
import uuid
import numpy as np
from datetime import datetime
from collections import deque
from itertools import combinations
import threading
import queue
import os

# === CONFIGURATION ===
MODEL_PATH = r"yolov8s.pt"  # Use a lighter model for speed
RTSP_URL = "rtsp://seekurity:191001@192.168.1.26/stream1"
CONFIDENCE_THRESHOLD = 0.25
MAX_HISTORY = 10
CLOSE_PROXIMITY = 75
AGGRESSIVE_MOVEMENT = 10
ALLOWED_CLASSES = {"person", "cell phone", "knife"}
FRAME_SKIP = 3
DOWNSCALE_SIZE = (320, 240)
ENABLE_DISPLAY = True

# === SETUP ===
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|timeout;5000000"
model = YOLO(MODEL_PATH)
class_names = model.names
person_trackers = {}

frame_queue = queue.Queue(maxsize=5)
stop_event = threading.Event()

# === UTILS ===
def euclidean_distance(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

def log_event(message):
    timestamp = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
    entry = f"{timestamp} {message}"
    print(entry)
    with open("logs.txt", "a") as f:
        f.write(entry + "\n")

# === THREAD: FRAME CAPTURE ===
def frame_reader():
    cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        print("Failed to open RTSP stream.")
        stop_event.set()
        return

    print(f"Connected to RTSP URL: {RTSP_URL}")
    count = 0

    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame.")
            stop_event.set()
            break

        count += 1
        if count % FRAME_SKIP != 0:
            continue

        if frame_queue.full():
            try:
                frame_queue.get_nowait()  # Drop oldest
            except queue.Empty:
                pass

        frame_queue.put(frame)

    cap.release()

# === MAIN THREAD: INFERENCE AND VISUALIZATION ===
def main_loop():
    # Warm-up
    _ = model.predict(np.zeros((DOWNSCALE_SIZE[1], DOWNSCALE_SIZE[0], 3), dtype=np.uint8), imgsz=DOWNSCALE_SIZE[0])

    while not stop_event.is_set():
        try:
            frame = frame_queue.get(timeout=0.05)
        except queue.Empty:
            continue

        original_frame = frame.copy()
        inference_frame = cv2.resize(frame, DOWNSCALE_SIZE)

        try:
            result = model.predict(inference_frame, imgsz=DOWNSCALE_SIZE[0], conf=CONFIDENCE_THRESHOLD, verbose=False)[0]
        except Exception as e:
            print(f"Inference error: {e}")
            continue

        current_ids, knife_boxes = [], []
        scale_x = original_frame.shape[1] / DOWNSCALE_SIZE[0]
        scale_y = original_frame.shape[0] / DOWNSCALE_SIZE[1]

        for box in result.boxes:
            class_id = int(box.cls[0])
            label = class_names[class_id].lower()
            if label not in ALLOWED_CLASSES:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            x1 = int(x1 * scale_x)
            x2 = int(x2 * scale_x)
            y1 = int(y1 * scale_y)
            y2 = int(y2 * scale_y)
            conf = float(box.conf[0])
            center = ((x1 + x2) // 2, (y1 + y2) // 2)

            if label == "person":
                matched_id = None
                for pid, pdata in person_trackers.items():
                    if euclidean_distance(center, pdata['centers'][-1]) < CLOSE_PROXIMITY:
                        matched_id = pid
                        break

                if not matched_id:
                    matched_id = str(uuid.uuid4())[:8]
                    person_trackers[matched_id] = {
                        'centers': deque(maxlen=MAX_HISTORY),
                        'box': (),
                        'status': 'normal',
                        'conf': conf
                    }

                tracker = person_trackers[matched_id]
                tracker['centers'].append(center)
                tracker['box'] = (x1, y1, x2, y2)
                tracker['conf'] = conf
                current_ids.append(matched_id)

            elif label == "knife":
                knife_boxes.append((x1, y1, x2, y2))

        # --- ARMING DETECTION ---
        for pid in current_ids:
            person = person_trackers[pid]
            for kx1, ky1, kx2, ky2 in knife_boxes:
                px1, py1, px2, py2 = person['box']
                if kx1 >= px1 and ky1 >= py1 and kx2 <= px2 and ky2 <= py2:
                    person['status'] = 'armed'

        # --- FIGHTING DETECTION ---
        for id1, id2 in combinations(current_ids, 2):
            p1, p2 = person_trackers[id1], person_trackers[id2]
            c1, c2 = p1['centers'], p2['centers']
            if len(c1) < 2 or len(c2) < 2:
                continue

            speed1 = euclidean_distance(c1[-1], c1[-2])
            speed2 = euclidean_distance(c2[-1], c2[-2])
            dist = euclidean_distance(c1[-1], c2[-1])

            if speed1 > AGGRESSIVE_MOVEMENT and speed2 > AGGRESSIVE_MOVEMENT and dist < CLOSE_PROXIMITY:
                if p1['status'] == 'normal' and p2['status'] == 'normal':
                    p1['status'] = p2['status'] = 'fighting'
                    log_event(f"Fighting detected between {id1} and {id2}")

        # --- ASSAULT DETECTION ---
        for id1, id2 in combinations(current_ids, 2):
            p1, p2 = person_trackers[id1], person_trackers[id2]
            if p1['status'] == 'armed' and p2['status'] == 'normal':
                armed, target = p1, p2
            elif p2['status'] == 'armed' and p1['status'] == 'normal':
                armed, target = p2, p1
            else:
                continue

            if len(armed['centers']) >= 2:
                speed = euclidean_distance(armed['centers'][-1], armed['centers'][-2])
                dist = euclidean_distance(armed['centers'][-1], target['centers'][-1])
                if speed > AGGRESSIVE_MOVEMENT and dist < CLOSE_PROXIMITY:
                    armed['status'] = 'assaulting'
                    log_event("Assault detected: Armed approaching unarmed")

        # --- DISPLAY ---
        if ENABLE_DISPLAY:
            violence_detected = any(person_trackers[pid]['status'] in ['fighting', 'assaulting'] for pid in current_ids)
            if violence_detected:
                overlay = np.full_like(original_frame, (0, 0, 255))
                original_frame = cv2.addWeighted(overlay, 0.3, original_frame, 0.7, 0)
                cv2.rectangle(original_frame, (10, 10), (270, 50), (0, 0, 255), -1)
                cv2.putText(original_frame, "VIOLENCE DETECTED", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            for pid in current_ids:
                person = person_trackers[pid]
                x1, y1, x2, y2 = person['box']
                status, conf = person['status'], person['conf']
                color = {
                    'normal': (0, 255, 0),
                    'armed': (0, 165, 255),
                    'fighting': (0, 255, 255),
                    'assaulting': (0, 0, 255)
                }.get(status, (255, 255, 255))
                label = f"{status} {conf:.2f}"
                cv2.rectangle(original_frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(original_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            for x1, y1, x2, y2 in knife_boxes:
                cv2.rectangle(original_frame, (x1, y1), (x2, y2), (128, 0, 128), 2)
                cv2.putText(original_frame, "knife", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (128, 0, 128), 2)

            cv2.imshow("Violence Detection", original_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                stop_event.set()
                break

    cv2.destroyAllWindows()

# === RUN ===
reader_thread = threading.Thread(target=frame_reader, daemon=True)
reader_thread.start()
main_loop()
