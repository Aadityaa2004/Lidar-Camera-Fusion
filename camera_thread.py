from PySide6 import QtCore
import cv2 as cv
import numpy as np
from ultralytics import YOLO
import cvzone
import time
import math
from config import YOLO_MODEL_PATH, CLASS_NAMES, CAMERA_FOV_H, CAMERA_RESOLUTION_WIDTH, CAMERA_RESOLUTION_HEIGHT, NUM_SEGMENTS

class CameraThread(QtCore.QThread):
    new_frame = QtCore.Signal(np.ndarray)
    
    def __init__(self):
        super().__init__()
        self.stop_flag = False
        self.model = YOLO(YOLO_MODEL_PATH)
        self.lidar_data = None
        self.segment_width = CAMERA_RESOLUTION_WIDTH // NUM_SEGMENTS
        self.distance_color_map = [
            (0, (0, 0, 255)),          # Red
            (30, (77, 77, 255)),       # Light Red
            (60, (153, 153, 255)),     # Lighter Red
            (90, (255, 153, 204)),     # Light Purple
            (120, (255, 102, 178)),    # Medium Purple
            (150, (255, 51, 153)),     # Darker Purple
            (180, (255, 0, 128)),      # Dark Blue
            (210, (255, 51, 153)),     # Medium Blue
            (240, (255, 102, 178)),    # Lighter Blue
            (270, (255, 153, 204)),    # Lightest Blue
            (300, (0, 255, 0))         # Green
        ]

    def run(self):
        cap = cv.VideoCapture(0)
        if not cap.isOpened():
            print("Couldn't open Camera")
            return
        
        pTime = 0
        while not self.stop_flag:
            ret, frame = cap.read()
            if ret:
                frame = self.process_frame(frame)
                cTime = time.time()
                fps = 1 / (cTime - pTime)
                pTime = cTime
                cv.putText(frame, f"FPS: {int(fps)}", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (10, 10, 10), 2)
                self.new_frame.emit(frame)
            else:
                print("Failed to read from camera")
                break

        cap.release()

    def process_frame(self, frame):
        results = self.model(frame, stream=True)
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = math.ceil((box.conf[0] * 100)) / 100
                cls = int(box.cls[0])
                if conf > 0.5:
                    self.draw_box(frame, x1, y1, x2, y2, cls, conf)
        
        if self.lidar_data is not None:
            self.draw_colored_segments(frame)

        return frame

    def draw_box(self, frame, x1, y1, x2, y2, cls, conf):
        color = (0, 255, 0)
        cvzone.putTextRect(frame, f'{CLASS_NAMES[cls]} {conf}', 
                           (max(0, x1), max(35, y1)), scale=2, thickness=2,
                           colorB=color, colorT=(0, 0, 0), colorR=color, offset=5)
        cv.rectangle(frame, (x1, y1), (x2, y2), color, 3)

    def update_lidar_data(self, x, y, distances, angles):
        self.lidar_data = (x, y, distances, angles)

    def get_color(self, distance):
        if distance > 300:
            return (0, 255, 0)  # Green for distances > 300
        for i, (dist, color) in enumerate(self.distance_color_map):
            if distance <= dist:
                if i == 0:
                    return color
                prev_dist, prev_color = self.distance_color_map[i-1]
                ratio = (distance - prev_dist) / (dist - prev_dist)
                return self.interpolate_color(prev_color, color, ratio)
        return self.distance_color_map[-1][1]

    def interpolate_color(self, color1, color2, ratio):
        return tuple(int((1 - ratio) * c1 + ratio * c2) for c1, c2 in zip(color1, color2))

    def draw_colored_segments(self, frame):
        x, y, distances, angles = self.lidar_data
        
        # Adjust angles to align with camera
        adjusted_angles = (angles + 150) % 360
        
        # Filter angles within camera FOV
        half_fov = CAMERA_FOV_H / 2
        center_angle = 180
        filtered_mask = ((adjusted_angles >= center_angle - half_fov) & 
                         (adjusted_angles <= center_angle + half_fov))
        filtered_angles = adjusted_angles[filtered_mask]
        filtered_distances = distances[filtered_mask]
        
        # Create histogram
        hist, bin_edges = np.histogram(filtered_angles, bins=NUM_SEGMENTS, 
                                       weights=filtered_distances, 
                                       range=(center_angle - half_fov, center_angle + half_fov))
        
        # Create colored overlay
        overlay = np.zeros_like(frame)
        
        for i, distance in enumerate(hist):
            if distance > 0:
                start_x = int((i / NUM_SEGMENTS) * CAMERA_RESOLUTION_WIDTH)
                end_x = int(((i + 1) / NUM_SEGMENTS) * CAMERA_RESOLUTION_WIDTH)
                
                color = self.get_color(distance)
                
                cv.rectangle(overlay, (start_x, 0), (end_x, frame.shape[0]), color, -1)
                
                # Add distance text
                text_x = (start_x + end_x) // 2 - 40
                cv.putText(frame, f'{distance:.2f} cm', 
                           (text_x, 50), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # Apply colored overlay with transparency
        cv.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
        
        # Draw vertical lines for each segment
        for i in range(NUM_SEGMENTS):
            x = int(i * self.segment_width)
            cv.line(frame, (x, 0), (x, frame.shape[0]), (255, 255, 255), 2)

    def stop(self):
        self.stop_flag = True
        self.wait()