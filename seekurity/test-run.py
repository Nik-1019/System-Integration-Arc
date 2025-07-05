import cv2
import numpy as np
import time
import queue
import os
from threading import Thread, Lock
from datetime import datetime
from ultralytics import YOLO

# ---------------- Create Logs Directory ---------------- #
os.makedirs("logs", exist_ok=True)

# ---------------- RTSP Stream Class ---------------- #
class RTSPStream:
    def __init__(self, rtsp_url):
        self.rtsp_url = rtsp_url
        self.cap = None
        self.frame_queue = queue.Queue(maxsize=2)
        self.lock = Lock()
        self.running = False
        self.connect()

    def connect(self):
        self.cap = cv2.VideoCapture(self.rtsp_url)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.running = True
        Thread(target=self.update_frame, daemon=True).start()

    def update_frame(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                print("Connection lost, reconnecting...")
                self.reconnect()
                continue
            if self.frame_queue.full():
                try:
                    self.frame_queue.get_nowait()
                except queue.Empty:
                    pass
            self.frame_queue.put(frame)

    def reconnect(self):
        with self.lock:
            if self.cap:
                self.cap.release()
            time.sleep(1)
            self.connect()

    def read(self):
        try:
            return self.frame_queue.get_nowait()
        except queue.Empty:
            return None

    def release(self):
        self.running = False
        if self.cap:
            self.cap.release()

# ---------------- Utility Functions ---------------- #
def get_center(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2, (y1 + y2) / 2)

def euclidean_distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def calculate_ioa(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    areaA = max(0, boxA[2] - boxA[0]) * max(0, boxA[3] - boxA[1])
    return interArea / areaA if areaA > 0 else 0

# ---------------- Log Events ---------------- #
def trigger_alert(label, box):
    print(f">>> ALERT triggered for {label.upper()} at box {box}")

def log_event(label, box, last_logged_time, cooldown=30):
    allowed_labels = {"armed", "knife", "gun", "fighting", "assault"}
    if label not in allowed_labels:
        return

    current_time = time.time()
    last_time = last_logged_time.get(label, 0)
    if current_time - last_time >= cooldown:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        x1, y1, x2, y2 = map(int, box)
        log_msg = f"[{timestamp}] [LOG] {label.upper()} detected at position: ({x1},{y1}) to ({x2},{y2})"

        print(log_msg)

        with open("detection_logs.txt", "a") as f:
            f.write(log_msg + "\n")

        with open(f"logs/{label}_logs.txt", "a") as f:
            f.write(log_msg + "\n")

        trigger_alert(label, box)
        last_logged_time[label] = current_time

# ---------------- Scene Analysis ---------------- #
def analyze_scene(results, model, prev_people):
    detections = results[0].boxes
    class_ids = detections.cls.cpu().numpy()
    boxes = detections.xyxy.cpu().numpy()
    names = model.names

    people, knives, guns = [], [], []
    labels = []

    for cls_id, box in zip(class_ids, boxes):
        label = names[int(cls_id)]
        if label == "person":
            people.append(box)
        elif label in ["knife", "scissors", "tie", "bottle", "hair dryer"]:
            knives.append(box)
        elif label in ["cell phone", "baseball bat", "remote"]:
            guns.append(box)

    armed_people, normal_people = [], []

    for person_box in people:
        armed = False
        for weapon_box in knives + guns:
            ioa = calculate_ioa(weapon_box, person_box)
            if ioa >= 0.5:
                labels.append(("armed", person_box))
                armed_people.append(person_box)
                armed = True
                break
        if not armed:
            labels.append(("normal", person_box))
            normal_people.append(person_box)

    for i in range(len(people)):
        for j in range(i + 1, len(people)):
            p1 = get_center(people[i])
            p2 = get_center(people[j])
            if euclidean_distance(p1, p2) < 50:
                labels.append(("fighting", people[i]))
                labels.append(("fighting", people[j]))

    for armed_box in armed_people:
        for normal_box in normal_people:
            ioa = calculate_ioa(armed_box, normal_box)
            if ioa > 0.3:
                labels.append(("assault", armed_box))
                break

    for knife_box in knives:
        labels.append(("knife", knife_box))
    for gun_box in guns:
        labels.append(("gun", gun_box))

    return labels, people

# ---------------- Main Runtime ---------------- #
def run_detection():
    model = YOLO('yolov8m.pt')
    model.fuse()

    rtsp_url = "rtsp://seekurity:191001@192.168.1.26/stream1"
    stream = RTSPStream(rtsp_url)
    _ = model(np.zeros((640, 640, 3), dtype=np.uint8))

    prev_people = []
    last_logged_time = {}

    color_map = {
        "normal": (0, 255, 0),
        "armed": (0, 0, 255),
        "fighting": (0, 255, 255),
        "assault": (128, 0, 128),
        "knife": (0, 165, 255),
        "gun": (0, 165, 255),
    }

    try:
        while True:
            start_time = time.time()
            frame = stream.read()
            if frame is None:
                time.sleep(0.01)
                continue

            results = model(frame, imgsz=640, half=True, verbose=False)
            labels, prev_people = analyze_scene(results, model, prev_people)

            for label, box in labels:
                log_event(label, box, last_logged_time)

                x1, y1, x2, y2 = map(int, box)
                color = color_map.get(label, (255, 255, 255))
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

            fps = 1 / (time.time() - start_time)
            cv2.putText(frame, f"FPS: {int(fps)}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

            cv2.imshow("Violence Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        stream.release()
        cv2.destroyAllWindows()

# ---------------- Entry Point ---------------- #
if __name__ == "__main__":
    run_detection()
