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
                
                segments = frame.shape[1] / 11
                for i in range(1, 12):
                    x = int(segments * i)
                    cv.line(frame, ( x, 0), ( x, frame.shape[0]), (0, 0, 0), 2)

                center_x = frame.shape[1] // 2
                center_y = frame.shape[0] // 2
                cv.circle(frame, (center_x, center_y), 10, (255, 0, 0), -1)
                
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

        if self.lidar_data is not None:
            frame = self.draw_lidar_info(frame)
        
        return frame

    def draw_box(self, frame, x1, y1, x2, y2, cls, conf):
        color = (0, 255, 0)
        cvzone.putTextRect(frame, f'{CLASS_NAMES[cls]} {conf}', 
                           (max(0, x1), max(35, y1)), scale=2, thickness=2,
                           colorB=color, colorT=(0, 0, 0), colorR=color, offset=5)
        cv.rectangle(frame, (x1, y1), (x2, y2), color, 3)
        center_x_box, center_y_box = (x1 + x2) // 2, (y1 + y2) // 2
        cv.circle(frame, (center_x_box, center_y_box), 5, color, -1)

    def draw_lidar_info(self, frame):
        # Add your LiDAR visualization logic here
        # This is where you can draw LiDAR information on the frame
        return frame

    def handle_lidar_data(self, x, y, distances, angles):
        self.lidar_data = (x, y, distances, angles)
        self.new_lidar_data.emit(x, y, distances, angles)
        self.print_lidar_data(distances, angles)

    def print_lidar_data(self, distances, angles):
        print("LiDAR Data:")
        # for i in range(min(len(distances), len(angles))):
        #     print(f"Angle: {angles[i]:.2f}°, Distance: {distances[i]:.2f} cm")
        # print("--------------------")

    def stop(self):
        self.stop_flag = True
        self.lidar_thread.stop()
        self.wait()