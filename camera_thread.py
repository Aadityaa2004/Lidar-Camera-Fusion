from PySide6 import QtCore
import cv2 as cv
import numpy as np
from ultralytics import YOLO
import cvzone
import time
import math
from config import YOLO_MODEL_PATH, CLASS_NAMES

class CameraThread(QtCore.QThread):
    new_frame = QtCore.Signal(np.ndarray)
    
    def __init__(self):
        super().__init__()
        self.stop_flag = False
        self.model = YOLO(YOLO_MODEL_PATH)

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
        return frame

    def draw_box(self, frame, x1, y1, x2, y2, cls, conf):
        color = (0, 255, 0)
        cvzone.putTextRect(frame, f'{CLASS_NAMES[cls]} {conf}', 
                           (max(0, x1), max(35, y1)), scale=1, thickness=1,
                           colorB=color, colorT=(255,255,255), colorR=color, offset=5)
        cv.rectangle(frame, (x1, y1), (x2, y2), color, 3)

    def stop(self):
        self.stop_flag = True
        self.wait()