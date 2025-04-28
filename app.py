import streamlit as st
import cv2
import numpy as np
import tempfile
from ultralytics import YOLO
import tensorflow as tf
from collections import defaultdict, deque
from scipy.spatial import distance as dist

# Load models
yolo_model = YOLO("vehicle_detection_model.pt")
classification_model = tf.keras.models.load_model("vehicle_classification_model_augmented3.keras")

# Class labels
class_labels = ['bus', 'car', 'motorcycle', 'truck', 'van']

# Tracking parameters
MAX_DISAPPEARED = 5  # Frames to keep tracking without detection
MIN_CONFIDENCE = 0.5  # Minimum detection confidence
TRACKING_THRESHOLD = 50  # Pixel distance for tracking matches

class VehicleTracker:
    def __init__(self):
        self.next_id = 0
        self.vehicles = {}
        self.disappeared = {}
        self.class_history = defaultdict(lambda: deque(maxlen=5))

    def update(self, detections):
        if len(detections) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > MAX_DISAPPEARED:
                    del self.vehicles[object_id]
                    del self.disappeared[object_id]
            return

        centroids = np.array([d['centroid'] for d in detections]).reshape(-1, 2)
        object_ids = list(self.vehicles.keys())
        object_centroids = np.array([v['centroid'] for v in self.vehicles.values()]).reshape(-1, 2)

        if len(object_centroids) == 0:
            for detection in detections:
                self.register_vehicle(detection)
            return

        D = dist.cdist(object_centroids, centroids)
        rows = D.min(axis=1).argsort()
        cols = D.argmin(axis=1)[rows]

        used_rows = set()
        used_cols = set()

        for (row, col) in zip(rows, cols):
            if row in used_rows or col in used_cols:
                continue

            if D[row, col] > TRACKING_THRESHOLD:
                continue

            object_id = object_ids[row]
            self.vehicles[object_id]['centroid'] = centroids[col]
            self.class_history[object_id].append(detections[col]['class'])
            self.disappeared[object_id] = 0

            used_rows.add(row)
            used_cols.add(col)

        unused_rows = set(range(D.shape[0])).difference(used_rows)
        unused_cols = set(range(D.shape[1])).difference(used_cols)

        for row in unused_rows:
            object_id = object_ids[row]
            self.disappeared[object_id] += 1
            if self.disappeared[object_id] > MAX_DISAPPEARED:
                del self.vehicles[object_id]
                del self.disappeared[object_id]

        for col in unused_cols:
            self.register_vehicle(detections[col])

    def register_vehicle(self, detection):
        self.vehicles[self.next_id] = {
            'centroid': detection['centroid']
        }
        self.class_history[self.next_id].append(detection['class'])
        self.disappeared[self.next_id] = 0
        self.next_id += 1


tracker = VehicleTracker()

def classify_image(image):
    img_resized = cv2.resize(image, (224, 224))
    img_normalized = tf.keras.applications.mobilenet_v2.preprocess_input(img_resized)
    prediction = classification_model.predict(np.expand_dims(img_normalized, axis=0))
    return class_labels[np.argmax(prediction)]

# Streamlit interface
st.title("Vehicle Tracking System")
uploaded_file = st.file_uploader("📤 Upload traffic video", type=["mp4", "avi"])
frame_display = st.empty()

if uploaded_file:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())

    cap = cv2.VideoCapture(tfile.name)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (1280, 720))
        detections = []

        # Perform detection with YOLO
        results = yolo_model.predict(frame)

        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Extract bounding box coordinates and confidence score
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
                conf_score = float(box.conf[0])  # Confidence score
                class_id = int(box.cls[0])  # Class ID
                
                # Only process boxes with confidence above a threshold
                if conf_score < MIN_CONFIDENCE:
                    continue
                
                # Get class label from class ID
                vehicle_class = class_labels[class_id]

                # Draw bounding box and label on the frame
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label_text = f"{vehicle_class} ({conf_score:.2f})"
                cv2.putText(frame, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Add detection data for tracking
                centroid = (int((x1 + x2) / 2), int((y1 + y2) / 2))
                detections.append({
                    'centroid': centroid,
                    'class': vehicle_class
                })

        tracker.update(detections)

        # Display the processed frame in Streamlit
        frame_display.image(frame, channels="BGR")

    cap.release()

