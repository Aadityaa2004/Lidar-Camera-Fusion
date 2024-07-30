from PySide6 import QtCore
import cv2 as cv
import numpy as np
from ultralytics import YOLO
import cvzone
import time
import math
from config import LIDAR_BAUDRATE, LIDAR_PORT, YOLO_MODEL_PATH, CLASS_NAMES, CAMERA_FOV_H, CAMERA_RESOLUTION_WIDTH, CAMERA_RESOLUTION_HEIGHT, NUM_SEGMENTS
from lidar_thread import LidarThread

class CameraThread(QtCore.QThread):
    new_frame = QtCore.Signal(np.ndarray)
    new_lidar_data = QtCore.Signal(np.ndarray, np.ndarray, np.ndarray, np.ndarray)
    
    def __init__(self):
        super().__init__()
        self.stop_flag = False
        self.model = YOLO(YOLO_MODEL_PATH)
        self.lidar_thread = LidarThread(port=LIDAR_PORT, baudrate=LIDAR_BAUDRATE)
        self.lidar_thread.new_data.connect(self.handle_lidar_data)
        self.lidar_data = None
        self.distance_values = [None] * 12
        self.distance_color_map = [
            (0, (0, 0, 0)),          # Red
            (30, (0, 0, 255)),         # Red
            (70, (0, 69, 255)),        # Dark Orange
            (110, (0, 140, 255)),      # Medium Orange
            (150, (0, 180, 255)),      # Light Orange
            (190, (0, 225, 255)),      # Yellow-Orange
            (230, (0, 255, 255)),      # Yellow
            (270, (0, 255, 200)),      # Yellow-Green
            (310, (0, 255, 150)),      # Light Green
            (350, (0, 255, 100)),      # Medium Green
            (390, (0, 255, 0))         # Green
        ]

    def run(self):
        self.lidar_thread.start()
        cap = cv.VideoCapture(0)
        if not cap.isOpened():
            print("Couldn't open Camera")
            return

        pTime = 0
        while not self.stop_flag:
            ret, frame = cap.read()
            if ret:
                frame = self.process_frame(frame)
                self.draw_distance_boxes(frame)

                # Draw center circle
                center_x = frame.shape[1] // 2
                center_y = frame.shape[0] // 2
                cv.circle(frame, (center_x, center_y), 10, (255, 0, 0), -1)

                # Calculate and display FPS
                cTime = time.time()
                fps = 1 / (cTime - pTime)
                pTime = cTime
                cv.putText(frame, f"FPS: {int(fps)}", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (10, 10, 10), 4)
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
        return frame

    def draw_box(self, frame, x1, y1, x2, y2, cls, conf):
        color = (0, 255, 0)
        cvzone.putTextRect(frame, f'{CLASS_NAMES[cls]} {conf}', 
                           (max(0, x1), max(35, y1)), scale=2, thickness=2,
                           colorB=color, colorT=(0, 0, 0), colorR=color, offset=5)
        cv.rectangle(frame, (x1, y1), (x2, y2), color, 3)
        center_x_box, center_y_box = (x1 + x2) // 2, (y1 + y2) // 2
        cv.circle(frame, (center_x_box, center_y_box), 5, color, -1)

    def process_lidar_data(self):
        if self.lidar_data is None:
            return

        x, y, distances, angles = self.lidar_data
        transformed_angles = np.where(angles > 309, angles - 360, angles)
        filtered_mask = (angles > 309) | (angles < 51)
        filtered_angles = transformed_angles[filtered_mask]

        target_angles = [310, 320, 330, 340, 350, 360, 0, 10, 20, 30, 40, 50]
        for i, angle_target in enumerate(target_angles):
            valid_x, valid_y, valid_distances = [], [], []
            for j, angle in enumerate(angles):
                if angle_target - 5 <= angle < angle_target + 5 or (angle_target == 360 and 355 <= angle < 360) or (angle_target == 0 and 0 <= angle < 5):
                    valid_x.append(x[j])
                    valid_y.append(y[j])
                    valid_distances.append(distances[j])

            if valid_distances:
                min_distance_index = valid_distances.index(min(valid_distances))
                closest_distance = valid_distances[min_distance_index]
                self.distance_values[i] = closest_distance
            else:
                self.distance_values[i] = None

        print("Distance Values: ", self.distance_values)

    def draw_distance_boxes(self, frame):
        box_height = 50
        y_start = frame.shape[0] - box_height
        segments = frame.shape[1] / 11

        for i in range(11):
            x_start = int(segments * i)
            x_end = int(segments * (i + 1))
            
            if self.distance_values[i] is not None:
                color = self.get_color(self.distance_values[i])
                cv.rectangle(frame, (x_start, y_start), (x_end, frame.shape[0]), color, -1)  # Filled rectangle
                cv.rectangle(frame, (x_start, y_start), (x_end, frame.shape[0]), (255, 255, 255), 2)  # White outline
                cv.putText(frame, f"{self.distance_values[i]:.2f}", (x_start + 5, y_start + 30), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    def get_color(self, distance):
        for threshold, color in self.distance_color_map:
            if distance <= threshold:
                return color
        return self.distance_color_map[-1][1]  # Return the last color if distance exceeds all thresholds

    def handle_lidar_data(self, x, y, distances, angles):
        self.lidar_data = (x, y, distances, angles)
        self.process_lidar_data()
        self.new_lidar_data.emit(x, y, distances, angles)

    def stop(self):
        self.stop_flag = True
        self.lidar_thread.stop()
        self.wait()