import cv2
import math
import uuid
import numpy as np
from datetime import datetime
from collections import deque, defaultdict
from itertools import combinations
import threading
import queue
import os
import json
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
from enum import Enum
import logging
from ultralytics import YOLO

# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class Config:
    # Model settings
    MODEL_PATH: str = "yolov8s.pt"
    CONFIDENCE_THRESHOLD: float = 0.25
    WEAPON_CONFIDENCE_THRESHOLD: float = 0.15
    
    # Stream settings
    RTSP_URL: str = "rtsp://seekurity:191001@192.168.1.26/stream1"
    FRAME_SKIP: int = 3
    DOWNSCALE_SIZE: Tuple[int, int] = (640, 480)
    
    # Detection parameters
    CLOSE_PROXIMITY: float = 100.0
    AGGRESSIVE_MOVEMENT: float = 15.0
    MAX_HISTORY: int = 10
    MIN_VIOLENCE_FRAMES: int = 5
    TRACKER_TIMEOUT: int = 30  # frames
    
    # Weapon detection
    WEAPON_CLASSES: Set[str] = None
    PERSON_CLASSES: Set[str] = None
    
    # System settings
    ENABLE_DISPLAY: bool = True
    ENABLE_LOGGING: bool = True
    LOG_FILE: str = "violence_detection.log"
    DEBUG_MODE: bool = False
    
    def __post_init__(self):
        if self.WEAPON_CLASSES is None:
            self.WEAPON_CLASSES = {
                'knife', 'scissors', 'sword', 'blade', 'machete',
                'cleaver', 'dagger', 'scalpel', 'razor', 'box cutter'
            }
        if self.PERSON_CLASSES is None:
            self.PERSON_CLASSES = {'person', 'people'}

class PersonStatus(Enum):
    NORMAL = "normal"
    ARMED = "armed"
    FIGHTING = "fighting"
    ASSAULTING = "assaulting"
    SUSPICIOUS = "suspicious"

# =============================================================================
# LOGGING SETUP
# =============================================================================

def setup_logging(config: Config) -> logging.Logger:
    """Setup logging configuration"""
    logger = logging.getLogger('ViolenceDetection')
    logger.setLevel(logging.DEBUG if config.DEBUG_MODE else logging.INFO)
    
    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)
    
    # File handler
    if config.ENABLE_LOGGING:
        file_handler = logging.FileHandler(config.LOG_FILE)
        file_handler.setLevel(logging.DEBUG)
        file_format = logging.Formatter('%(asctime)s - %(levelname)s - %(funcName)s - %(message)s')
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)
    
    return logger

# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class Detection:
    """Represents a single detection"""
    box: Tuple[int, int, int, int]  # x1, y1, x2, y2
    center: Tuple[int, int]
    confidence: float
    class_name: str
    class_id: int

@dataclass
class PersonTracker:
    """Tracks a person across frames"""
    id: str
    centers: deque
    boxes: deque
    status: PersonStatus
    confidence_history: deque
    last_seen: int
    violence_frame_count: int
    
    def __post_init__(self):
        if not isinstance(self.centers, deque):
            self.centers = deque(maxlen=10)
        if not isinstance(self.boxes, deque):
            self.boxes = deque(maxlen=10)
        if not isinstance(self.confidence_history, deque):
            self.confidence_history = deque(maxlen=10)
    
    @property
    def current_box(self) -> Tuple[int, int, int, int]:
        return self.boxes[-1] if self.boxes else (0, 0, 0, 0)
    
    @property
    def current_center(self) -> Tuple[int, int]:
        return self.centers[-1] if self.centers else (0, 0)
    
    @property
    def current_confidence(self) -> float:
        return self.confidence_history[-1] if self.confidence_history else 0.0
    
    def update(self, detection: Detection, frame_number: int):
        """Update tracker with new detection"""
        self.centers.append(detection.center)
        self.boxes.append(detection.box)
        self.confidence_history.append(detection.confidence)
        self.last_seen = frame_number
    
    def get_movement_speed(self) -> float:
        """Calculate movement speed between last two positions"""
        if len(self.centers) < 2:
            return 0.0
        return euclidean_distance(self.centers[-1], self.centers[-2])
    
    def is_moving_aggressively(self) -> bool:
        """Check if person is moving aggressively"""
        return self.get_movement_speed() > 15.0

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def euclidean_distance(p1: Tuple[int, int], p2: Tuple[int, int]) -> float:
    """Calculate euclidean distance between two points"""
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def calculate_overlap(box1: Tuple[int, int, int, int], 
                     box2: Tuple[int, int, int, int]) -> float:
    """Calculate overlap percentage between two boxes"""
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    # Calculate intersection
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)
    
    if x2_i <= x1_i or y2_i <= y1_i:
        return 0.0
    
    intersection_area = (x2_i - x1_i) * (y2_i - y1_i)
    box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
    
    return intersection_area / box1_area if box1_area > 0 else 0.0

def is_weapon_near_person(weapon_box: Tuple[int, int, int, int], 
                         person_box: Tuple[int, int, int, int]) -> bool:
    """Check if weapon is near or overlapping with person"""
    overlap = calculate_overlap(weapon_box, person_box)
    return overlap > 0.1  # 10% overlap threshold

# =============================================================================
# DETECTION SYSTEM
# =============================================================================

class WeaponDetector:
    """Handles weapon detection and classification"""
    
    def __init__(self, config: Config, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.weapon_classes = config.WEAPON_CLASSES
    
    def detect_weapons(self, detections: List[Detection]) -> List[Detection]:
        """Filter detections to find weapons"""
        weapons = []
        for detection in detections:
            if self._is_weapon_class(detection.class_name):
                weapons.append(detection)
                self.logger.debug(f"Weapon detected: {detection.class_name} (conf: {detection.confidence:.3f})")
        return weapons
    
    def _is_weapon_class(self, class_name: str) -> bool:
        """Check if class name represents a weapon"""
        class_lower = class_name.lower()
        return any(weapon in class_lower for weapon in self.weapon_classes)

class PersonTrackingSystem:
    """Handles person tracking and management"""
    
    def __init__(self, config: Config, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.trackers: Dict[str, PersonTracker] = {}
        self.frame_number = 0
    
    def update_trackers(self, person_detections: List[Detection]) -> List[str]:
        """Update all person trackers with new detections"""
        self.frame_number += 1
        current_ids = []
        
        # Match detections to existing trackers
        for detection in person_detections:
            tracker_id = self._find_matching_tracker(detection)
            
            if tracker_id is None:
                # Create new tracker
                tracker_id = self._create_new_tracker(detection)
            
            # Update tracker
            self.trackers[tracker_id].update(detection, self.frame_number)
            current_ids.append(tracker_id)
        
        # Clean up old trackers
        self._cleanup_old_trackers(current_ids)
        
        return current_ids
    
    def _find_matching_tracker(self, detection: Detection) -> Optional[str]:
        """Find existing tracker that matches the detection"""
        best_match = None
        min_distance = float('inf')
        
        for tracker_id, tracker in self.trackers.items():
            distance = euclidean_distance(detection.center, tracker.current_center)
            if distance < self.config.CLOSE_PROXIMITY and distance < min_distance:
                min_distance = distance
                best_match = tracker_id
        
        return best_match
    
    def _create_new_tracker(self, detection: Detection) -> str:
        """Create a new person tracker"""
        tracker_id = str(uuid.uuid4())[:8]
        self.trackers[tracker_id] = PersonTracker(
            id=tracker_id,
            centers=deque(maxlen=self.config.MAX_HISTORY),
            boxes=deque(maxlen=self.config.MAX_HISTORY),
            status=PersonStatus.NORMAL,
            confidence_history=deque(maxlen=self.config.MAX_HISTORY),
            last_seen=self.frame_number,
            violence_frame_count=0
        )
        self.logger.debug(f"Created new tracker: {tracker_id}")
        return tracker_id
    
    def _cleanup_old_trackers(self, current_ids: List[str]):
        """Remove trackers that haven't been seen recently"""
        to_remove = []
        for tracker_id, tracker in self.trackers.items():
            if (tracker_id not in current_ids and 
                self.frame_number - tracker.last_seen > self.config.TRACKER_TIMEOUT):
                to_remove.append(tracker_id)
        
        for tracker_id in to_remove:
            del self.trackers[tracker_id]
            self.logger.debug(f"Removed old tracker: {tracker_id}")

class ViolenceAnalyzer:
    """Analyzes behavior for violence detection"""
    
    def __init__(self, config: Config, logger: logging.Logger):
        self.config = config
        self.logger = logger
    
    def analyze_violence(self, tracking_system: PersonTrackingSystem, 
                        weapon_detections: List[Detection]) -> bool:
        """Analyze current frame for violence"""
        violence_detected = False
        
        # Check for armed persons
        self._detect_armed_persons(tracking_system, weapon_detections)
        
        # Check for fighting behavior
        violence_detected |= self._detect_fighting(tracking_system)
        
        # Check for assault behavior
        violence_detected |= self._detect_assault(tracking_system)
        
        return violence_detected
    
    def _detect_armed_persons(self, tracking_system: PersonTrackingSystem, 
                             weapon_detections: List[Detection]):
        """Detect if any person is armed"""
        for tracker_id, tracker in tracking_system.trackers.items():
            is_armed = False
            
            for weapon in weapon_detections:
                if is_weapon_near_person(weapon.box, tracker.current_box):
                    is_armed = True
                    break
            
            if is_armed and tracker.status == PersonStatus.NORMAL:
                tracker.status = PersonStatus.ARMED
                self.logger.warning(f"Person {tracker_id} detected as ARMED")
    
    def _detect_fighting(self, tracking_system: PersonTrackingSystem) -> bool:
        """Detect fighting between persons"""
        violence_detected = False
        
        for id1, id2 in combinations(tracking_system.trackers.keys(), 2):
            tracker1 = tracking_system.trackers[id1]
            tracker2 = tracking_system.trackers[id2]
            
            if len(tracker1.centers) < 2 or len(tracker2.centers) < 2:
                continue
            
            # Check proximity and movement
            distance = euclidean_distance(tracker1.current_center, tracker2.current_center)
            speed1 = tracker1.get_movement_speed()
            speed2 = tracker2.get_movement_speed()
            
            if (distance < self.config.CLOSE_PROXIMITY and 
                speed1 > self.config.AGGRESSIVE_MOVEMENT and 
                speed2 > self.config.AGGRESSIVE_MOVEMENT):
                
                # Increment violence frame count
                tracker1.violence_frame_count += 1
                tracker2.violence_frame_count += 1
                
                # Only trigger if violence persists
                if (tracker1.violence_frame_count >= self.config.MIN_VIOLENCE_FRAMES and
                    tracker2.violence_frame_count >= self.config.MIN_VIOLENCE_FRAMES):
                    
                    tracker1.status = PersonStatus.FIGHTING
                    tracker2.status = PersonStatus.FIGHTING
                    violence_detected = True
                    self.logger.warning(f"FIGHTING detected between {id1} and {id2}")
            else:
                # Reset violence frame count if not fighting
                tracker1.violence_frame_count = max(0, tracker1.violence_frame_count - 1)
                tracker2.violence_frame_count = max(0, tracker2.violence_frame_count - 1)
        
        return violence_detected
    
    def _detect_assault(self, tracking_system: PersonTrackingSystem) -> bool:
        """Detect assault (armed person approaching unarmed)"""
        violence_detected = False
        
        for id1, id2 in combinations(tracking_system.trackers.keys(), 2):
            tracker1 = tracking_system.trackers[id1]
            tracker2 = tracking_system.trackers[id2]
            
            armed_tracker = None
            target_tracker = None
            
            if tracker1.status == PersonStatus.ARMED and tracker2.status == PersonStatus.NORMAL:
                armed_tracker, target_tracker = tracker1, tracker2
            elif tracker2.status == PersonStatus.ARMED and tracker1.status == PersonStatus.NORMAL:
                armed_tracker, target_tracker = tracker2, tracker1
            
            if armed_tracker and target_tracker and len(armed_tracker.centers) >= 2:
                distance = euclidean_distance(armed_tracker.current_center, target_tracker.current_center)
                speed = armed_tracker.get_movement_speed()
                
                if distance < self.config.CLOSE_PROXIMITY and speed > self.config.AGGRESSIVE_MOVEMENT:
                    armed_tracker.violence_frame_count += 1
                    
                    if armed_tracker.violence_frame_count >= self.config.MIN_VIOLENCE_FRAMES:
                        armed_tracker.status = PersonStatus.ASSAULTING
                        violence_detected = True
                        self.logger.warning(f"ASSAULT detected: {armed_tracker.id} approaching {target_tracker.id}")
                else:
                    armed_tracker.violence_frame_count = max(0, armed_tracker.violence_frame_count - 1)
        
        return violence_detected

# =============================================================================
# MAIN DETECTION SYSTEM
# =============================================================================

class ViolenceDetectionSystem:
    """Main violence detection system"""
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = setup_logging(config)
        self.model = None
        self.frame_queue = queue.Queue(maxsize=5)
        self.stop_event = threading.Event()
        
        # Initialize components
        self.weapon_detector = WeaponDetector(config, self.logger)
        self.tracking_system = PersonTrackingSystem(config, self.logger)
        self.violence_analyzer = ViolenceAnalyzer(config, self.logger)
        
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize YOLO model"""
        try:
            self.model = YOLO(self.config.MODEL_PATH)
            self.logger.info(f"Model loaded: {self.config.MODEL_PATH}")
            
            # Print available classes for debugging
            if self.config.DEBUG_MODE:
                self.logger.info("Available model classes:")
                for i, name in enumerate(self.model.names.values()):
                    self.logger.info(f"  {i}: {name}")
                    
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise
    
    def _setup_stream(self):
        """Setup RTSP stream capture"""
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|timeout;5000000"
        cap = cv2.VideoCapture(self.config.RTSP_URL, cv2.CAP_FFMPEG)
        
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open RTSP stream: {self.config.RTSP_URL}")
        
        self.logger.info(f"Connected to RTSP stream: {self.config.RTSP_URL}")
        return cap
    
    def _frame_reader_thread(self):
        """Thread for reading frames from RTSP stream"""
        try:
            cap = self._setup_stream()
            frame_count = 0
            
            while not self.stop_event.is_set():
                ret, frame = cap.read()
                if not ret:
                    self.logger.error("Failed to read frame from stream")
                    break
                
                frame_count += 1
                if frame_count % self.config.FRAME_SKIP != 0:
                    continue
                
                # Manage frame queue
                if self.frame_queue.full():
                    try:
                        self.frame_queue.get_nowait()
                    except queue.Empty:
                        pass
                
                self.frame_queue.put(frame)
                
        except Exception as e:
            self.logger.error(f"Frame reader error: {e}")
        finally:
            cap.release()
    
    def _process_detections(self, result) -> Tuple[List[Detection], List[Detection]]:
        """Process YOLO detection results"""
        person_detections = []
        all_detections = []
        
        if not result.boxes:
            return person_detections, all_detections
        
        for box in result.boxes:
            class_id = int(box.cls[0])
            class_name = self.model.names[class_id]
            confidence = float(box.conf[0])
            
            # Get bounding box coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            center = ((x1 + x2) // 2, (y1 + y2) // 2)
            
            detection = Detection(
                box=(x1, y1, x2, y2),
                center=center,
                confidence=confidence,
                class_name=class_name,
                class_id=class_id
            )
            
            all_detections.append(detection)
            
            # Filter person detections
            if class_name.lower() in self.config.PERSON_CLASSES:
                if confidence >= self.config.CONFIDENCE_THRESHOLD:
                    person_detections.append(detection)
            
            # Debug output
            if self.config.DEBUG_MODE:
                self.logger.debug(f"Detection: {class_name} (conf: {confidence:.3f})")
        
        return person_detections, all_detections
    
    def _render_visualization(self, frame: np.ndarray, 
                            weapon_detections: List[Detection], 
                            violence_detected: bool) -> np.ndarray:
        """Render visualization on frame"""
        if not self.config.ENABLE_DISPLAY:
            return frame
        
        # Violence overlay
        if violence_detected:
            overlay = np.full_like(frame, (0, 0, 255), dtype=np.uint8)
            frame = cv2.addWeighted(overlay, 0.3, frame, 0.7, 0)
            
            # Violence alert banner
            cv2.rectangle(frame, (10, 10), (400, 60), (0, 0, 255), -1)
            cv2.putText(frame, "VIOLENCE DETECTED", (20, 45), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
        
        # Draw person trackers
        for tracker_id, tracker in self.tracking_system.trackers.items():
            x1, y1, x2, y2 = tracker.current_box
            status = tracker.status.value
            conf = tracker.current_confidence
            
            # Color based on status
            color_map = {
                PersonStatus.NORMAL.value: (0, 255, 0),
                PersonStatus.ARMED.value: (0, 165, 255),
                PersonStatus.FIGHTING.value: (0, 255, 255),
                PersonStatus.ASSAULTING.value: (0, 0, 255),
                PersonStatus.SUSPICIOUS.value: (255, 0, 255)
            }
            color = color_map.get(status, (255, 255, 255))
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f"{tracker_id[:4]}: {status} ({conf:.2f})"
            cv2.putText(frame, label, (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Draw weapon detections
        for weapon in weapon_detections:
            x1, y1, x2, y2 = weapon.box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (128, 0, 128), 2)
            cv2.putText(frame, f"{weapon.class_name} ({weapon.confidence:.2f})", 
                       (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (128, 0, 128), 2)
        
        return frame
    
    def run(self):
        """Main detection loop"""
        try:
            # Start frame reader thread
            reader_thread = threading.Thread(target=self._frame_reader_thread, daemon=True)
            reader_thread.start()
            
            # Warm up model
            dummy_frame = np.zeros((self.config.DOWNSCALE_SIZE[1], self.config.DOWNSCALE_SIZE[0], 3), dtype=np.uint8)
            _ = self.model.predict(dummy_frame, conf=self.config.CONFIDENCE_THRESHOLD, verbose=False)
            
            self.logger.info("Starting violence detection system...")
            
            while not self.stop_event.is_set():
                try:
                    # Get frame from queue
                    frame = self.frame_queue.get(timeout=0.1)
                    original_frame = frame.copy()
                    
                    # Resize for inference
                    inference_frame = cv2.resize(frame, self.config.DOWNSCALE_SIZE)
                    
                    # Run inference
                    results = self.model.predict(inference_frame, 
                                               conf=self.config.WEAPON_CONFIDENCE_THRESHOLD,
                                               verbose=False)
                    
                    if not results:
                        continue
                    
                    # Process detections
                    person_detections, all_detections = self._process_detections(results[0])
                    
                    # Scale detections back to original frame size
                    scale_x = original_frame.shape[1] / self.config.DOWNSCALE_SIZE[0]
                    scale_y = original_frame.shape[0] / self.config.DOWNSCALE_SIZE[1]
                    
                    for detection in person_detections + all_detections:
                        x1, y1, x2, y2 = detection.box
                        detection.box = (int(x1 * scale_x), int(y1 * scale_y),
                                       int(x2 * scale_x), int(y2 * scale_y))
                        detection.center = ((detection.box[0] + detection.box[2]) // 2,
                                          (detection.box[1] + detection.box[3]) // 2)
                    
                    # Detect weapons
                    weapon_detections = self.weapon_detector.detect_weapons(all_detections)
                    
                    # Update person tracking
                    current_person_ids = self.tracking_system.update_trackers(person_detections)
                    
                    # Analyze for violence
                    violence_detected = self.violence_analyzer.analyze_violence(
                        self.tracking_system, weapon_detections)
                    
                    # Render visualization
                    if self.config.ENABLE_DISPLAY:
                        display_frame = self._render_visualization(
                            original_frame, weapon_detections, violence_detected)
                        
                        cv2.imshow("Violence Detection System", display_frame)
                        
                        key = cv2.waitKey(1) & 0xFF
                        if key == ord('q'):
                            break
                        elif key == ord('d'):
                            self.config.DEBUG_MODE = not self.config.DEBUG_MODE
                            self.logger.info(f"Debug mode: {self.config.DEBUG_MODE}")
                
                except queue.Empty:
                    continue
                except Exception as e:
                    self.logger.error(f"Processing error: {e}")
                    continue
        
        except KeyboardInterrupt:
            self.logger.info("Shutting down...")
        except Exception as e:
            self.logger.error(f"System error: {e}")
        finally:
            self.stop_event.set()
            if self.config.ENABLE_DISPLAY:
                cv2.destroyAllWindows()

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main function"""
    config = Config()
    
    # Override config for debugging
    config.DEBUG_MODE = True
    config.WEAPON_CONFIDENCE_THRESHOLD = 0.1  # Lower threshold for weapon detection
    
    system = ViolenceDetectionSystem(config)
    system.run()

if __name__ == "__main__":
    main()