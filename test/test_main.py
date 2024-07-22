import sys
import cv2 as cv
import numpy as np
from math import cos, sin, radians
from PySide6 import QtCore, QtWidgets, QtGui
import pyqtgraph as pg
from pyrplidar import PyRPlidar
from ultralytics import YOLO
import cvzone
import time
import math
from sort import *

class LidarThread(QtCore.QThread):
    new_data = QtCore.Signal(np.ndarray, np.ndarray, np.ndarray, np.ndarray)

    def __init__(self, port, baudrate):
        super().__init__()
        self.lidar = PyRPlidar()
        self.port = port
        self.baudrate = baudrate
        self.stop_flag = False

    def run(self):
        self.lidar.connect(port=self.port, baudrate=self.baudrate, timeout=3)
        self.lidar.set_motor_pwm(660)
        scan_generator = self.lidar.force_scan()

        while not self.stop_flag:
            x_coords = []
            y_coords = []
            angles = []
            distances = []

            for count, scan in enumerate(scan_generator()):
                line = str(scan)
                line = line.replace('{', ' ').replace('}', ' ').split(',')
                angle = float(line[2].split(':')[1])
                distance = float(line[3].split(':')[1]) / 10
                if distance > 0:
                    x = distance * sin(radians(angle))
                    y = distance * cos(radians(angle))
                    x_coords.append(x)
                    y_coords.append(y)
                    angles.append(angle)
                    distances.append(distance)

                if count >= 360:  # Emit data every 360 points
                    self.new_data.emit(np.array(x_coords), np.array(y_coords), np.array(distances), np.array(angles))
                    x_coords = []
                    y_coords = []
                    break

        self.lidar.stop()
        self.lidar.set_motor_pwm(0)
        self.lidar.disconnect()

    def stop(self):
        self.stop_flag = True
        self.wait()

class CameraThread(QtCore.QThread):
    new_frame = QtCore.Signal(np.ndarray)

    def __init__(self):
        super().__init__()
        self.stop_flag = False

    def run(self):
        cap = cv.VideoCapture(0)  # Use video file or 0 for webcam
        cap.set(3, 1280)
        cap.set(4, 720)

        model = YOLO('/Users/aaditya/dev/YOLO/YOLO-weights/yolov8n.pt')  # Path to YOLO model weights

        classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
                      "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
                      "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
                      "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
                      "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
                      "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
                      "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
                      "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
                      "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
                      "teddy bear", "hair drier", "toothbrush"]

        tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)
        limits = [400, 297, 673, 297]
        totalCount = []
        pTime = 0

        if not cap.isOpened():
            print("Couldn't open Camera")
            return

        while not self.stop_flag:
            ret, frame = cap.read()
            if not ret:
                print("Frame not read")
                break

            # Directly process the captured frame
            results = model(frame, stream=True)
            detections = np.empty((0, 5))

            for r in results:
                boxes = r.boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    conf = math.ceil((box.conf[0]*100))/100
                    classId = int(box.cls[0])
                    currentClass = classNames[classId]

                    # Detecting specific items
                    if conf > 0.3:
                        if currentClass in ["car", "truck", "bus", "motorbike"]:
                            # Handle vehicles
                            print(f"Detected {currentClass} with confidence {conf:.2f}")
                        else:
                            print(f"Detected {currentClass} with confidence {conf:.2f}")

                        currentArray = np.array([x1, y1, x2, y2, conf])
                        detections = np.vstack((detections, currentArray))

            cTime = time.time()
            fps = 1 / (cTime - pTime)
            pTime = cTime

            resultsTracker = tracker.update(detections)
            cv.line(frame, (limits[0], limits[1]), (limits[2], limits[3]), (0, 0, 255), 5)

            for result in resultsTracker:
                x1, y1, x2, y2, Id = result
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                cx, cy = x1 + (x2-x1) // 2, y1 + (y2-y1) // 2
                cv.circle(frame, (cx, cy), 5, (255, 0, 255), cv.FILLED)

                if limits[0] < cx < limits[2] and limits[1] - 15 < cy < limits[1] + 15:
                    if totalCount.count(Id) == 0:
                        totalCount.append(Id)
                        cv.line(frame, (limits[0], limits[1]), (limits[2], limits[3]), (0, 255, 0), 5)
                    
                cvzone.putTextRect(frame, f'Count: {len(totalCount)}', (50, 50))

            # Emit frame
            self.new_frame.emit(frame)

            cv.imshow('Camera Feed', frame)

            if cv.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv.destroyAllWindows()

    def stop(self):
        self.stop_flag = True
        self.wait()


class MainWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('LiDAR, Camera Feed, and Object Detection')
        self.resize(1200, 900)

        # LiDAR plot setup
        self.lidar_plot_widget = pg.GraphicsLayoutWidget()
        self.lidar_plot_widget.setFixedSize(600, 450)
        self.lidar_plot = self.lidar_plot_widget.addPlot(title="LiDAR Points")
        self.lidar_plot.setAspectLocked(True)
        self.lidar_plot.setXRange(-500, 500)
        self.lidar_plot.setYRange(-500, 500)
        self.lidar_plot_data = self.lidar_plot.plot([], [], pen=None, symbolBrush=(255, 0, 0), symbolSize=3, symbolPen=None)

        self.lines = []
        self.text_items = []
        for _ in range(36):
            line = pg.PlotDataItem([], [], pen=pg.mkPen(color=(0, 255, 0), width=2))
            self.lidar_plot.addItem(line)
            self.lines.append(line)
            
            text_item = pg.TextItem('', anchor=(0.5, 1))
            self.lidar_plot.addItem(text_item)
            self.text_items.append(text_item)

        # Add field of view outline
        self.fov_lines = self.add_fov_lines()

        # Camera label setup
        self.camera_label = QtWidgets.QLabel()
        self.camera_label.setFixedSize(600, 450)
        self.camera_label.setScaledContents(True)

        # Histogram plot setup
        self.histogram_plot_widget = pg.GraphicsLayoutWidget()
        self.histogram_plot_widget.setFixedSize(600, 450)
        self.histogram_plot = self.histogram_plot_widget.addPlot(title="Distance Histogram")
        self.histogram_plot.setLabel('bottom', 'Angle (degrees)')
        self.histogram_plot.setLabel('left', 'Distance (cm)')
        self.histogram_plot.setXRange(-100, 100)
        self.histogram_plot.setYRange(0, 10)
        self.histogram_plot.setMouseEnabled(x=False, y=True)
        self.histogram_hist = pg.BarGraphItem(x=[], height=[], width=10, brush='b')
        self.histogram_plot.addItem(self.histogram_hist)
        self.histogram_line = self.histogram_plot.plot([], [], pen=pg.mkPen(color='b', width=2))

        # Layout setup
        layout = QtWidgets.QGridLayout(self)
        layout.addWidget(self.lidar_plot_widget, 0, 0)
        layout.addWidget(self.camera_label, 0, 1)
        layout.addWidget(self.histogram_plot_widget, 1, 0)
        layout.setSpacing(0)
        layout.setContentsMargins(0, 0, 0, 0)

        self.setLayout(layout)

        # Thread setup
        self.lidar_thread = LidarThread(port="/dev/tty.usbserial-0001", baudrate=256000)
        self.lidar_thread.new_data.connect(self.update_lidar_plot)
        self.lidar_thread.new_data.connect(self.update_histogram_plot)
        self.lidar_thread.start()
        
        self.camera_thread = CameraThread()
        self.camera_thread.new_frame.connect(self.update_camera_feed)
        self.camera_thread.start()

    def add_fov_lines(self):
        fov_lines = []
        angles = [51, 309]
        for angle in angles:
            x = 500 * sin(radians(angle))
            y = 500 * cos(radians(angle))
            line = pg.PlotDataItem([0, x], [0, y], pen=pg.mkPen(color=(255, 255, 255), width=1, style=QtCore.Qt.DashLine))
            self.lidar_plot.addItem(line)
            fov_lines.append(line)
        return fov_lines

    def update_camera_feed(self, frame):
        rgb_image = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        self.camera_label.setPixmap(QtGui.QPixmap.fromImage(qt_image))

    def update_lidar_plot(self, x_coords, y_coords, distances, angles):
        self.lidar_plot_data.setData(x_coords, y_coords)

        angle_segments = np.digitize(angles, np.linspace(0, 360, 37)) - 1
        for i in range(36):
            segment_mask = angle_segments == i
            if np.any(segment_mask):
                self.lines[i].setData([0, x_coords[segment_mask].mean()], [0, y_coords[segment_mask].mean()])
                self.text_items[i].setPos(x_coords[segment_mask].mean(), y_coords[segment_mask].mean())
                self.text_items[i].setText(f'{distances[segment_mask].mean():.2f} cm')
            else:
                self.lines[i].setData([], [])
                self.text_items[i].setText('')

    def update_histogram_plot(self, x_coords, y_coords, distances, angles):
        transformed_angles = [(angle - 360) if angle > 180 else angle for angle in angles]
        self.histogram_hist.setOpts(x=transformed_angles, height=distances, width=10)
        self.histogram_line.setData(transformed_angles, distances)

    def closeEvent(self, event):
        self.lidar_thread.stop()
        self.camera_thread.stop()
        event.accept()

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
