import cv2
from ultralytics import YOLO
import time
from threading import Thread, Lock
import queue

class RTSPStream:
    def __init__(self, rtsp_url):
        self.rtsp_url = rtsp_url
        self.cap = None
        self.frame_queue = queue.Queue(maxsize=2)  # Small buffer
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
            
            # Keep only the latest frame
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
            time.sleep(1)  # Wait before reconnecting
            self.connect()

    def read(self):
        return self.frame_queue.get() if not self.frame_queue.empty() else None

    def release(self):
        self.running = False
        if self.cap:
            self.cap.release()

def run_low_latency_detection():
    # Initialize model (half precision for faster inference)
    model = YOLO('yolov8m.pt')
    model.fuse()  # Optimize model
    
    # RTSP stream setup
    rtsp_url = "rtsp://seekurity:191001@192.168.1.26/stream1"
    stream = RTSPStream(rtsp_url)
    
    # Warmup model
    _ = model(np.zeros((640, 640, 3), dtype=np.uint8))
    
    try:
        while True:
            start_time = time.time()
            
            # Get the latest frame
            frame = stream.read()
            if frame is None:
                time.sleep(0.01)
                continue
            
            # Run inference (reduced size for speed)
            results = model(frame, imgsz=640, half=True, verbose=False)
            
            # Display results
            annotated_frame = results[0].plot()
            cv2.imshow("Low-Latency Detection", annotated_frame)
            
            # Calculate and display FPS
            fps = 1 / (time.time() - start_time)
            cv2.putText(annotated_frame, f"FPS: {int(fps)}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    finally:
        stream.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    import numpy as np
    run_low_latency_detection()