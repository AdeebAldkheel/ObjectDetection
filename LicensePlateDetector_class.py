import re
import time
import cv2
import easyocr
import logging
import numpy as np
from datetime import datetime
from typing import List, Tuple
from dataclasses import dataclass


@dataclass
class PlateDetection:
    """
    Represents a single license plate detection.

    Attributes:
        text (str): Recognized plate text (e.g., "ABC-123-XYZ").
        confidence (float): Confidence score from OCR.
        timestamp (str): Timestamp of detection in '%Y%m%d_%H%M%S' format.
        bbox (List[int]): Coordinates of the bounding box [(x1, y1), (x2, y2), ...].
    """
    text: str
    confidence: float
    timestamp: str
    bbox: List[int]


class LicensePlateDetector:
    """
    Detects and recognizes license plates in a given video frame
    using EasyOCR. It is configured to look for alphanumeric patterns,
    and filters out results that do not match the specified regex pattern.

    Typical usage:
    -------------
        detector = LicensePlateDetector(debug_level="DEBUG")
        frame = cv2.imread('plate_image.jpg')
        annotated_frame, detections = detector.detect_plates(frame)
    """

    def __init__(self, debug_level: str = 'INFO'):
        """
        Initializes the license plate detector and its OCR reader.

        Args:
            debug_level (str): Logging level (e.g., "DEBUG", "INFO", etc.). Default is "INFO".
        """
        self.logger = logging.getLogger('license_plate_detector')
        self.logger.setLevel(getattr(logging, debug_level.upper(), logging.INFO))

        # Instantiate EasyOCR with English language support; adjust as needed for other languages.
        self.reader = easyocr.Reader(['en'], gpu=False)
        self.detected_plates = {}

        self.logger.debug("LicensePlateDetector initialized with EasyOCR Reader (English).")

    def detect_plates(self, frame: np.ndarray) -> Tuple[np.ndarray, List[PlateDetection]]:
        """
        Detects license plates within the provided frame using OCR.

        Args:
            frame (np.ndarray): The input frame in which to detect plates.

        Returns:
            (annotated_frame, detections):
                annotated_frame (np.ndarray) - A copy of the original frame (currently unannotated).
                detections (List[PlateDetection]) - A list of plate detection results.
        """
        try:
            # Create a copy for potential annotation or further processing.
            annotated_frame = frame.copy()
            detections: List[PlateDetection] = []

            # Perform OCR to find text blocks in the image.
            self.logger.debug("Performing OCR on the current frame.")
            ocr_results = self.reader.readtext(annotated_frame)

            # Compile the regex pattern for a valid plate.
            # Example pattern: 'ABC-123-XYZ'
            valid_plate_pattern = re.compile(r"^([A-Z]{1,3})-(\d{1,3})-([A-Z]{1,3})")
            filtered_results = []

            # Filter results based on the pattern and confidence threshold.
            for (bbox, raw_text, prob) in ocr_results:
                # Convert text to uppercase to ensure consistent matching.
                text_upper = raw_text.upper()

                # Check if it matches the plate pattern (e.g., "ABC-123-XYZ").
                if valid_plate_pattern.match(text_upper):
                    filtered_results.append((bbox, text_upper, prob))

            # If no results match the pattern, return early with no detections.
            if not filtered_results:
                self.logger.debug("No valid plate patterns detected.")
                return annotated_frame, []

            # Select the plate candidate with the highest confidence.
            best_candidate = max(filtered_results, key=lambda x: x[2])
            bbox, text, prob = best_candidate

            # Apply a minimum confidence threshold of 0.8 (adjust as needed).
            if prob < 0.8:
                self.logger.debug(f"Plate detected but below confidence threshold (prob={prob:.2f}).")
                return annotated_frame, []

            # Construct the final detection object.
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            detection = PlateDetection(
                text=text,
                confidence=prob,
                timestamp=timestamp_str,
                bbox=bbox
            )
            detections.append(detection)

            # Return the unannotated frame along with the detection(s).
            return annotated_frame, detections

        except Exception as e:
            self.logger.error(f"Error in detect_plates: {e}", exc_info=True)
            return frame, []
