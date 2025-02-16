#!/usr/bin/env python3

import os
from dotenv import load_dotenv
import csv
import cv2
import json
import time
import queue
import logging
import yt_dlp
import threading
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Any
from functools import wraps
from dataclasses import dataclass, field
from datetime import datetime
from flask import Flask, Response, request, session, redirect, url_for, render_template_string
from logging.handlers import RotatingFileHandler
from concurrent.futures import ThreadPoolExecutor
from threading import Thread, Lock

# ---- 3rd Party Detection Libraries ----
# (Make sure you have ultralytics==8.x, supervision, etc. installed)
from ultralytics import YOLO
import supervision as sv

# ---- License Plate Detection Class (Separate File) ----
# For demonstration, assume LicensePlateDetector_class.py in the same directory.
# The file must define a `LicensePlateDetector` class with a `.detect_plates()` method.
from LicensePlateDetector_class import LicensePlateDetector


# =============================================================================
#                           1. Configuration & Logging
# =============================================================================
load_dotenv()
@dataclass
class Config:
    """Central configuration object for the application."""
    # Flask Server
    host: str = field(default="0.0.0.0")
    port: int = field(default=8001)

    # YOLO Model
    model_name: str = field(default="best.onnx")
    resolution: str = field(default="1080")
    confidence_threshold: float = field(default=0.5)

    # Logging
    debug_lvl: str = field(default="INFO")

    # Visualization
    display: bool = field(default=False)
    heat_map: bool = field(default=False)
    dot: bool = field(default=False)
    halo: bool = field(default=False)
    percentage_bar: bool = field(default=False)
    stream: bool = field(default=True)

    # Classes to detect (by name)
    desired_classes: List[str] = field(default_factory=lambda: [
        'person', 'bicycle', 'car', 'motorcycle', 'bus', 'train', 'truck', 'License_Plate'
    ])

    # CSV output
    csv_file_name: str = field(default='./output/detections.csv')

    # -------------------------------------------------------------------------
    # This loads config from a JSON file if present, otherwise uses defaults.
    # -------------------------------------------------------------------------
    @classmethod
    def load_or_create(cls, json_path: str) -> 'Config':
        path_obj = Path(json_path)
        if path_obj.exists():
            try:
                with open(path_obj, 'r', encoding='utf-8') as fp:
                    data = json.load(fp)
                return cls(**data)
            except (json.JSONDecodeError, Exception) as e:
                print(f"[WARNING] Could not parse {json_path}, using default config. Error: {e}")
        # Fallback: create default config
        default_cfg = cls()
        default_cfg.save_to_json(path_obj)
        return default_cfg

    def save_to_json(self, path_obj: Path) -> None:
        """Save config to a JSON file."""
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        with open(path_obj, 'w', encoding='utf-8') as fp:
            json.dump(self.__dict__, fp, indent=4)


def get_config() -> Config:
    """
    Retrieve Config instance.
    Loads from config.json if it exists; otherwise uses defaults,
    then applies environment variable overrides for production usage.
    """
    config = Config.load_or_create("config.json")
    # Environment variable overrides
    config.host = os.getenv("APP_HOST", config.host)
    config.port = int(os.getenv("APP_PORT", str(config.port)))
    config.debug_lvl = os.getenv("APP_LOG_LEVEL", config.debug_lvl)
    return config


def setup_logging(debug_level: str = "INFO", log_dir: str = "logs") -> None:
    """
    Sets up rotating file and console logging handlers.
    This ensures logs persist on disk and appear on console.
    """
    os.makedirs(log_dir, exist_ok=True)
    log_format = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    log_file = Path(log_dir) / "app.log"

    # Root logger config
    logging.basicConfig(
        level=getattr(logging, debug_level.upper(), logging.INFO),
        format=log_format,
        handlers=[
            logging.StreamHandler(),
            RotatingFileHandler(log_file, maxBytes=10 * 1024 * 1024, backupCount=5, encoding='utf-8')
        ]
    )


# =============================================================================
#                         2. Video Stream Handling
# =============================================================================

class VideoStreamHandler:
    """
    Handles video stream initialization, frame capture, and
    synchronization with a processing thread.
    """
    def __init__(self, config: Config) -> None:
        self.config = config
        # The environment variable for the video source (file path or URL):
        self.source: str = os.getenv("AI_VIDEO_SOURCE_URL", "")
        self.cap: cv2.VideoCapture = None
        self.stream_url: str = None

        # Thread-safety
        self.frame_lock: Lock = Lock()
        self.current_frame: np.ndarray = None
        self.running: bool = True

        self.logger = logging.getLogger('video_stream')
        self.logger.setLevel(getattr(logging, self.config.debug_lvl.upper(), logging.INFO))

    def get_stream_url(self) -> str:
        """
        If the source is a YouTube or HTTP link, tries to retrieve the actual stream URL.
        Otherwise, returns the local file path.
        """
        if not self.source:
            raise RuntimeError("AI_VIDEO_SOURCE_URL is not set.")

        # Example: If "http" in the source, assume it might be a YouTube link and attempt
        # best format selection. If it's a direct RTSP/HTTP stream, we just return the URL.
        if "http" in self.source:
            try:
                with yt_dlp.YoutubeDL({}) as ydl:
                    self.logger.info(f"Fetching info for: {self.source}")
                    info_dict = ydl.extract_info(self.source, download=False)
                    formats = [fmt for fmt in info_dict.get('formats', [info_dict])
                               if fmt.get('height') is not None]
                    best_format = max(formats, key=lambda f: f['height'])
                    return best_format['url']
            except Exception as e:
                self.logger.error(f"Failed to retrieve YouTube stream URL: {e}", exc_info=True)
                raise
        return self.source

    def initialize_capture(self) -> None:
        """
        Attempts to open the video source multiple times before failing.
        """
        for attempt in range(5):
            try:
                self.stream_url = self.get_stream_url()
                self.cap = cv2.VideoCapture(self.stream_url,cv2.CAP_FFMPEG )
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)
                if not self.cap.isOpened():
                    self.logger.warning(f"Attempt {attempt+1}/5: Unable to open capture.")
                    time.sleep(1.0)
                else:
                    self.logger.info("Video capture initialized successfully.")
                    return
            except Exception as e:
                self.logger.error(f"Error initializing capture: {e}")
                time.sleep(1.0)
        raise RuntimeError("Failed to open video source after 5 attempts.")

    def get_frame_dimensions(self) -> Tuple[int, int]:
        """
        Return the integer tuple for desired resolution.
        """
        resolutions = {
            "1080": (1920, 1080),
            "720": (1280, 720),
            "480": (640, 480)
        }
        return resolutions.get(self.config.resolution, (1280, 720))


# =============================================================================
#                     3. License Plate Detection Helpers
# =============================================================================

@dataclass
class PlateDetection:
    track_id: int
    frame: np.ndarray
    bbox: Tuple[int, int, int, int]
    conf: float
    cropped_img: np.ndarray
    detection_base: dict
    timestamp: float


class OptimizedLicensePlateDetector:
    """
    Batch-processes license plate detections in a background thread,
    preventing the main detection loop from blocking on external calls.
    """

    def __init__(self,
                 LP_detector: LicensePlateDetector,
                 output_plates_dir: str,
                 batch_size: int = 5,
                 max_queue_size: int = 100):
        self.LP_detector = LP_detector
        self.output_plates_dir = output_plates_dir
        self.batch_size = batch_size
        self.detection_queue = queue.Queue(maxsize=max_queue_size)
        self.detected_plates: Dict[int, Dict[str, Any]] = {}
        self.results_cache: Dict[int, Any] = {}

        # Lock to protect shared data
        self.lock = threading.Lock()

        # Thread pool to parallelize detection work
        self.thread_pool = ThreadPoolExecutor(max_workers=8)

        # Start the background consumer thread
        self.processing_thread = threading.Thread(
            target=self._process_detection_queue, daemon=True
        )
        self.processing_thread.start()

    def _should_update_detection(self, track_id: int, confidence: float) -> bool:
        """
        Decide if the new detection is better than a prior one for the same track_id.
        If so, remove the old image file from disk.
        """
        with self.lock:
            current_record = self.detected_plates.get(track_id)
            if not current_record:
                return True
            if current_record['confidence'] >= confidence:
                # Existing detection has higher or equal confidence
                return False
            else:
                # Remove old plate file
                old_path = current_record.get('file_path', '')
                if old_path and os.path.exists(old_path):
                    try:
                        os.remove(old_path)
                    except OSError:
                        pass
                self.detected_plates.pop(track_id, None)
                return True

    def _process_detection_queue(self) -> None:
        """
        Continuously process queued detection tasks in batches.
        """
        while True:
            batch = []
            try:
                # Block waiting for the first item
                item = self.detection_queue.get(timeout=0.1)
                batch.append(item)

                # Grab up to self.batch_size items without blocking
                while len(batch) < self.batch_size:
                    try:
                        item = self.detection_queue.get_nowait()
                        batch.append(item)
                    except queue.Empty:
                        break

                # Process the full batch
                self._process_batch(batch)

            except queue.Empty:
                time.sleep(0.01)  # idle briefly

    def _process_batch(self, batch: List[PlateDetection]) -> None:
        """
        Process each item in the batch in the thread pool.
        """
        for detection in batch:
            if detection.conf > 0.01:
                future = self.thread_pool.submit(
                    self._process_single_detection, detection
                )
                with self.lock:
                    self.results_cache[detection.track_id] = future

    def _process_single_detection(self, detection: PlateDetection) -> dict:
        """
        Run license plate detection on a single cropped image, then
        record the best results. Also updates the main detection structure.
        """
        processed_frame, plate_detections = self.LP_detector.detect_plates(
            detection.cropped_img
        )
        detection_data = []

        for plate_detection in plate_detections:
            plate_number = plate_detection.text
            plate_confidence = float(f"{plate_detection.confidence:.2f}")

            # Compose filename
            file_n = f"{plate_number}_{plate_confidence:.2f}.jpg"
            filepath = os.path.join(self.output_plates_dir, file_n)

            if self._should_update_detection(detection.track_id, plate_confidence):
                # Save new detection
                with self.lock:
                    self.detected_plates[detection.track_id] = {
                        'confidence': plate_confidence,
                        'file_path': filepath,
                        'track_id': detection.track_id,
                        'plate_number': plate_number,
                        'bbox': detection.bbox,
                        'timestamp': detection.timestamp
                    }
                # Save the entire frame for context
                cv2.imwrite(filepath, detection.frame)

            # Generate metadata to integrate with the main detection pipeline
            detection_data = detection.detection_base.copy()
            detection_data['name'] = plate_number

        return detection_data

    def handle_license_plate_detection(self,
                                       frame: np.ndarray,
                                       cropped_img: np.ndarray,
                                       track_id: int,
                                       conf: float,
                                       detection_base: dict,
                                       detections_data: List[dict],
                                       bbox: Tuple[int, int, int, int]) -> None:
        """
        Enqueue license plate detection and check for cached results.
        """
        detection_obj = PlateDetection(
            track_id=track_id,
            frame=frame.copy(),
            bbox=bbox,
            conf=conf,
            cropped_img=cropped_img,
            detection_base=detection_base,
            timestamp=time.time()
        )
        try:
            self.detection_queue.put_nowait(detection_obj)
        except queue.Full:
            logging.warning("License Plate detection queue is full; skipping detection.")

        # Check if there's a completed future from a previous detection
        with self.lock:
            future = self.results_cache.get(track_id)
            if future and future.done():
                results = future.result()
                detections_data.extend(results if isinstance(results, list) else [results])
                del self.results_cache[track_id]

    def cleanup(self) -> None:
        """
        Shut down thread pool cleanly.
        """
        self.thread_pool.shutdown(wait=True)


# =============================================================================
#                     4. Main Object Detection & Tracking
# =============================================================================

def _initialize_annotators():
    return {
        'heat_map': sv.HeatMapAnnotator(),
        'dot': sv.DotAnnotator(),
        'halo': sv.HaloAnnotator(),
        'percentage_bar': sv.PercentageBarAnnotator()
    }


class ObjectDetector:
    """
    Manages the YOLO model, processes frames, draws bounding boxes,
    and coordinates specialized detections (license plates, etc.).
    """

    def __init__(self, config: Config) -> None:
        self.config = config
        self.logger = logging.getLogger('object_detector')
        self.logger.setLevel(getattr(logging, self.config.debug_lvl.upper(), logging.INFO))

        # Load YOLO model
        self.model_coco = YOLO(config.model_name)
        # For CPU usage explicitly:
        self.loaded_device = 'cpu'

        # For each detection, track the time it first appeared
        self.start_time_dict: Dict[int, float] = {}

        # Visualization / annotation
        self.annotators = _initialize_annotators()

        # YOLO class mapping
        # e.g. {0: 'person', 1: 'bicycle', 2: 'car', ... }
        self.class_mapping: Dict[str, int] = {name: idx for idx, name in self.model_coco.names.items()}

        # For ID generation if YOLO tracker not used
        self.next_track_id = 1

        # Instantiate LicensePlateDetector (synchronous)
        self.LP_detector = LicensePlateDetector()

        # Create specialized plate detector with concurrency
        self.plate_detector = OptimizedLicensePlateDetector(
            LP_detector=self.LP_detector,
            output_plates_dir=os.path.join('output', 'plates'),
            batch_size=30,
            max_queue_size=100
        )

        # Directories to store snapshots
        self.output_person_dir = os.path.join('output', 'person')
        os.makedirs(self.output_person_dir, exist_ok=True)

        self.output_car_dir = os.path.join('output', 'cars')
        os.makedirs(self.output_car_dir, exist_ok=True)

        # Data structures to store the best detection snapshots
        self.detected_people: Dict[int, Dict[str, Any]] = {}
        self.detected_cars: Dict[int, Dict[str, Any]] = {}

        # CSV setup
        self.csv_path = self.config.csv_file_name
        self.csv_headers = [
            'timestamp', 'name', 'track_id', 'confidence', 'x1', 'y1', 'x2', 'y2', 'elapsed_time'
        ]
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(self.csv_headers)

    def _should_update_detection(self, track_id: int, confidence: float, class_name: str) -> bool:
        """
        Returns True if the new detection is better (higher confidence)
        than any existing detection for that track_id.
        """
        if class_name == "person":
            record = self.detected_people.get(track_id)
        elif class_name == "car":
            record = self.detected_cars.get(track_id)
        else:
            return True

        if not record:
            return True
        if record['confidence'] >= confidence:
            return False

        # Remove old file
        old_path = record.get('file_path', '')
        if old_path and os.path.exists(old_path):
            try:
                os.remove(old_path)
            except OSError:
                pass

        # Remove the old detection
        if class_name == "person":
            self.detected_people.pop(track_id, None)
        elif class_name == "car":
            self.detected_cars.pop(track_id, None)

        return True

    def format_time(self, elapsed_time: float) -> str:
        """Convert seconds to HH:MM:SS format."""
        hours, remainder = divmod(int(elapsed_time), 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    def _handle_person_detection(self,
                                 frame: np.ndarray,
                                 cropped_img: np.ndarray,
                                 track_id: int,
                                 conf: float,
                                 detection_base: Dict[str, Any],
                                 detections_data: List[Dict[str, Any]],
                                 bbox: Tuple[int, int, int, int],
                                 elapsed_time: float) -> None:
        """
        Specialized logic for processing a 'person' class detection.
        """
        x1, y1, x2, y2 = bbox
        detection_base['name'] = "person"
        detections_data.append(detection_base)

        if self._should_update_detection(track_id, conf, "person"):
            filename = f"person_ID_{track_id}.jpg"
            filepath = os.path.join(self.output_person_dir, filename)
            cv2.imwrite(filepath, cropped_img)
            self.detected_people[track_id] = {
                'confidence': conf,
                'file_path': filepath,
                'track_id': track_id
            }

        # Drawing bounding boxes
        color = (0, 255, 0) if elapsed_time < 60.0 else (0, 0, 255)
        label = f"person[{track_id}] {self.format_time(elapsed_time)}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)

    def _handle_car_detection(self,
                              frame: np.ndarray,
                              cropped_img: np.ndarray,
                              track_id: int,
                              conf: float,
                              detection_base: Dict[str, Any],
                              detections_data: List[Dict[str, Any]],
                              bbox: Tuple[int, int, int, int],
                              elapsed_time: float) -> None:
        """
        Specialized logic for processing a 'car' class detection.
        """
        x1, y1, x2, y2 = bbox
        detection_base['name'] = "car"
        detections_data.append(detection_base)

        if self._should_update_detection(track_id, conf, "car"):
            filename = f"car_ID_{track_id}.jpg"
            filepath = os.path.join(self.output_car_dir, filename)
            cv2.imwrite(filepath, cropped_img)
            self.detected_cars[track_id] = {
                'confidence': conf,
                'file_path': filepath,
                'track_id': track_id
            }

        # Drawing bounding boxes
        color = (0, 255, 0) if elapsed_time < 60.0 else (0, 0, 255)
        label = f"car[{track_id}] {self.format_time(elapsed_time)}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    def _process_detections(self, frame: np.ndarray, result: Any) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """
        Extract bounding boxes, class IDs, confidences, and track IDs from YOLO results.
        Updates internal data structures, writes images, returns detection info.
        """
        detections_data: List[Dict[str, Any]] = []

        if not hasattr(result, 'boxes') or len(result.boxes) == 0:
            return frame, detections_data  # No detections

        boxes_xyxy = result.boxes.xyxy.numpy()  # shape (N, 4)
        class_ids = result.boxes.cls.int().tolist()
        confidences = result.boxes.conf.numpy()
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        current_time = time.time()

        # YOLO might produce track IDs if using track(). If not, we generate them manually.
        track_ids = (result.boxes.id.int().tolist()
                     if hasattr(result.boxes, 'id') and result.boxes.id is not None
                     else list(range(self.next_track_id, self.next_track_id + len(boxes_xyxy))))

        if not hasattr(result.boxes, 'id') or result.boxes.id is None:
            self.next_track_id = track_ids[-1] + 1 if track_ids else self.next_track_id

        for box_xyxy, class_id, track_id, conf in zip(boxes_xyxy, class_ids, track_ids, confidences):
            class_name = self.model_coco.names[class_id]
            if class_name not in self.config.desired_classes:
                continue

            x1, y1, x2, y2 = map(int, box_xyxy)
            cropped_img = frame[y1:y2, x1:x2]

            if track_id not in self.start_time_dict:
                self.start_time_dict[track_id] = current_time

            elapsed_time = current_time - self.start_time_dict[track_id]
            detection_base = {
                'timestamp': timestamp,
                'track_id': track_id,
                'confidence': f"{conf:.2f}",
                'bbox': [x1, y1, x2, y2],
                'elapsed_time': elapsed_time
            }

            if class_name == "person":
                self._handle_person_detection(frame, cropped_img, track_id,
                                              conf, detection_base,
                                              detections_data, (x1, y1, x2, y2), elapsed_time)
            elif class_name == "car":
                self._handle_car_detection(frame, cropped_img, track_id,
                                           conf, detection_base,
                                           detections_data, (x1, y1, x2, y2), elapsed_time)
            elif class_name == "License_Plate":
                # Enqueue license plate detection
                self.plate_detector.handle_license_plate_detection(
                    frame, cropped_img, track_id, conf,
                    detection_base, detections_data, (x1, y1, x2, y2)
                )

        return frame, detections_data

    def _annotate_frame(self, frame: np.ndarray, result: Any) -> np.ndarray:
        """
        Apply additional advanced supervision-based annotations, if configured.
        """
        try:
            detections = sv.Detections.from_ultralytics(result)
            if self.config.heat_map:
                frame = self.annotators['heat_map'].annotate(frame.copy(), detections)
            if self.config.dot:
                frame = self.annotators['dot'].annotate(frame.copy(), detections)
            if self.config.halo:
                frame = self.annotators['halo'].annotate(frame.copy(), detections)
            if self.config.percentage_bar:
                custom_values = self._calculate_elapsed_times(result)
                if len(custom_values) > 0:
                    frame = self.annotators['percentage_bar'].annotate(
                        frame.copy(), detections, custom_values=custom_values
                    )
        except Exception as e:
            logging.error(f"Error annotating frame: {e}", exc_info=True)
        return frame

    def _calculate_elapsed_times(self, result: Any) -> np.ndarray:
        """
        Map each track ID to an elapsed time ratio between 0 and 1,
        for display in a percentage bar annotation.
        """
        if not (hasattr(result.boxes, 'id') and result.boxes.id is not None):
            return np.array([])

        track_ids = result.boxes.id.int().tolist()
        current_time = time.time()
        for t_id in track_ids:
            if t_id not in self.start_time_dict:
                self.start_time_dict[t_id] = current_time

        # Cap at 1.0 after 10s (example)
        elapsed_ratios = [
            min((current_time - self.start_time_dict[t_id]) / 10.0, 1.0)
            for t_id in track_ids
        ]
        return np.array(elapsed_ratios)

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """
        Main public method to process a single frame with YOLO, track objects,
        annotate the frame, and gather detections.
        """
        detections_data: List[Dict[str, Any]] = []
        try:
            # Limit classes to the ones in config.desired_classes
            class_indices = [
                self.class_mapping[c]
                for c in self.config.desired_classes
                if c in self.class_mapping
            ]

            results_stream = self.model_coco.track(
                frame,
                device=self.loaded_device,
                persist=True,
                conf=self.config.confidence_threshold,
                classes=class_indices if class_indices else None,
                stream=True,
                iou=0.5
            )

            # YOLO 'track(..., stream=True)' yields multiple results for the same frame
            for result in results_stream:
                if not result:
                    continue
                # Annotate
                frame = self._annotate_frame(frame, result)
                # Extract detection data
                frame, partial_data = self._process_detections(frame, result)
                if partial_data:
                    detections_data.extend(partial_data)
        except Exception as e:
            self.logger.error(f"Error processing frame: {e}", exc_info=True)
        return frame, detections_data


# =============================================================================
#                          5. Flask Application
# =============================================================================

def login_required(f):
    """Decorator to restrict access to authenticated users."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'logged_in' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function


class ObjectTrackingApp:
    """
    Main application that wires together:
    - Configuration
    - Logging
    - Video Streaming
    - Object Detection
    - Flask Routes & Session Management
    """

    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger('app')
        self.logger.info("Initializing Object Tracking Application")

        # Flask app
        self.flask_app = Flask(__name__)
        # Load secret key from environment variable for security
        self.flask_app.secret_key = os.getenv("FLASK_SECRET_KEY", "CHANGE_ME_IN_PRODUCTION")

        # Core components
        self.stream_handler = VideoStreamHandler(self.config)
        self.detector = ObjectDetector(self.config)

        # Streaming control
        self.is_streaming = True

        # CSV for detections
        self.csv_path = self.config.csv_file_name
        self.csv_headers = [
            'timestamp', 'name', 'track_id', 'confidence', 'x1', 'y1', 'x2', 'y2', 'elapsed_time'
        ]
        self._ensure_csv_headers()

        # Register HTTP routes
        self._register_routes()

    def _register_routes(self):
        """
        Define all Flask routes here: login, index, video_feed, logout, etc.
        """

        @self.flask_app.route('/login', methods=['GET', 'POST'])
        def login():
            error = None
            if request.method == 'POST':
                # Replace with real authentication logic
                if (request.form.get('username') == 'admin' and
                        request.form.get('password') == 'password'):
                    session['logged_in'] = True
                    return redirect(url_for('index'))
                else:
                    error = 'Invalid credentials.'

            return render_template_string('''
                <!DOCTYPE html>
                <html>
                <head>
                    <title>Login</title>
                </head>
                <body>
                    <h2>Login</h2>
                    {% if error %}
                        <p style="color:red;">{{ error }}</p>
                    {% endif %}
                    <form method="POST">
                        <label>Username: <input type="text" name="username"></label><br><br>
                        <label>Password: <input type="password" name="password"></label><br><br>
                        <button type="submit">Login</button>
                    </form>
                </body>
                </html>
            ''', error=error)

        @self.flask_app.route('/')
        @login_required
        def index():
            return render_template_string('''
                <!DOCTYPE html>
                <html>
                <head>
                    <title>Object Detection Stream</title>
                    <style>
                        .logout-btn {
                            position: absolute;
                            top: 20px;
                            right: 20px;
                            background-color: #f44336;
                            color: white;
                            padding: 10px 15px;
                            text-decoration: none;
                            border-radius: 4px;
                        }
                    </style>
                </head>
                <body>
                    <a href="{{ url_for('logout') }}" class="logout-btn">Logout</a>
                    <h1>Object Detection Stream</h1>
                    <img src="{{ url_for('video_feed') }}" style="max-width:100%;" />
                </body>
                </html>
            ''')

        @self.flask_app.route('/video_feed')
        @login_required
        def video_feed():
            if not self.config.stream:
                return Response('Streaming is disabled', mimetype='text/plain')
            return Response(self._generate_frames(),
                            mimetype='multipart/x-mixed-replace; boundary=frame')

        @self.flask_app.route('/logout')
        def logout():
            session.pop('logged_in', None)
            return redirect(url_for('login'))

    def _ensure_csv_headers(self):
        """Create CSV file with headers if it doesn't exist."""
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(self.csv_headers)

    def _generate_frames(self):
        """
        Continuously yield frames (JPEG) for the Flask streaming endpoint.
        Capped at ~30 FPS to prevent overloading.
        """
        frame_interval = 1.0 / 30.0
        last_frame_time = time.time()

        while self.is_streaming:
            current_time = time.time()
            elapsed = current_time - last_frame_time
            if elapsed < frame_interval:
                time.sleep(0.01)
                continue

            with self.stream_handler.frame_lock:
                frame = self.stream_handler.current_frame
            if frame is None:
                time.sleep(0.1)
                continue

            ret, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            if not ret:
                self.logger.warning("Failed to encode frame.")
                #time.sleep(0.1)
                continue

            last_frame_time = current_time

            # Yield frame in multipart format
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' +
                   buffer.tobytes() +
                   b'\r\n')

    def run(self):
        """
        Start the Flask server and the object detection thread.
        In production, you would typically run this app behind Gunicorn or uWSGI.
        """
        try:
            # Open the capture
            self.stream_handler.initialize_capture()

            # Launch detection in a background thread
            detection_thread = Thread(target=self._detection_loop, daemon=True)
            detection_thread.start()

            # Start Flask's development server (not for production use).
            self.flask_app.run(
                host=self.config.host,
                port=self.config.port,
                threaded=True,
                use_reloader=False
            )

        except KeyboardInterrupt:
            self.logger.info("Received keyboard interrupt, shutting down.")
            self.cleanup()
        except Exception as e:
            self.logger.error(f"Application error: {e}", exc_info=True)
            self.cleanup()

    def _detection_loop(self):
        """
        Continuously read frames from the capture, run object detection,
        log detection events, and store frames for streaming.
        """
        while self.stream_handler.running:
            ret, frame = self.stream_handler.cap.read()
            if not ret:
                self.logger.warning("Failed to read frame. Reinitializing capture...")
                #time.sleep(0.5)
                #self.stream_handler.initialize_capture()
                continue
            if frame is not None and frame.shape[0] > 1 and frame.shape[1] > 1:
                # Resize frame for uniform processing
                frame = cv2.resize(frame, self.stream_handler.get_frame_dimensions())

                processed_frame, detections_data = self.detector.process_frame(frame)
                if detections_data:
                    # Log to CSV
                    with open(self.csv_path, 'a', newline='', encoding='utf-8') as f:
                        writer = csv.writer(f)
                        for d in detections_data:
                            writer.writerow([
                                d['timestamp'],
                                d['name'],
                                d['track_id'],
                                d['confidence'],
                                d['bbox'][0],
                                d['bbox'][1],
                                d['bbox'][2],
                                d['bbox'][3],
                                d['elapsed_time']
                            ])
                    self.logger.info(f"Logged {len(detections_data)} detections to CSV.")

                # Update current_frame for streaming
                with self.stream_handler.frame_lock:
                    self.stream_handler.current_frame = processed_frame

                if self.config.display:
                    cv2.imshow('Detection Feed', processed_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        self.cleanup()
                        break

    def cleanup(self):
        """
        Gracefully stop streaming and detection, release resources.
        """
        self.is_streaming = False
        self.stream_handler.running = False
        if self.stream_handler.cap:
            self.stream_handler.cap.release()
        cv2.destroyAllWindows()
        # Cleanup license plate thread pool
        if hasattr(self.detector, 'plate_detector'):
            self.detector.plate_detector.cleanup()
        self.logger.info("Application cleanup complete.")


# =============================================================================
#                            6. Main Entry Point
# =============================================================================
if __name__ == "__main__":
    # 1. Load configuration
    cfg = get_config()

    # 2. Setup logging
    setup_logging(cfg.debug_lvl)

    # 3. Initialize the app
    app = ObjectTrackingApp(cfg)

    # 4. Run
    app.run()
