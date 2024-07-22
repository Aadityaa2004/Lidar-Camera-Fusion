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
        cap = cv.VideoCapture(0)

        model = YOLO('/Users/aaditya/ALSTOM/Lidar/yolov8n.pt')
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


        if not cap.isOpened():
            print("Couldn't open Camera")
            return

        while not self.stop_flag:
            ret, frame = cap.read()
            if ret:



                
                self.new_frame.emit(frame)
            else:
                print("Failed to read from camera")
                break  # Exit the loop if reading fails

        cap.release()

    def stop(self):
        self.stop_flag = True
        self.wait()

class MainWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('LiDAR, Camera Feed, and Histogram')
        self.resize(1200, 900)

        # LiDAR plot setup
        self.lidar_plot_widget = pg.GraphicsLayoutWidget()
        self.lidar_plot_widget.setFixedSize(600, 450)  # Adjust size for grid layout
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
        self.camera_label.setFixedSize(600, 450)  # Adjust size for grid layout
        self.camera_label.setScaledContents(True)  # Scale the image to fit the label

        # Histogram plot setup
        self.histogram_plot_widget = pg.GraphicsLayoutWidget()
        self.histogram_plot_widget.setFixedSize(600, 450)  # Adjust size for grid layout
        self.histogram_plot = self.histogram_plot_widget.addPlot(title="Distance Histogram")
        self.histogram_plot.setLabel('bottom', 'Angle (degrees)')
        self.histogram_plot.setLabel('left', 'Distance (cm)')
        self.histogram_plot.setXRange(-60, 60)  # Set range for transformed angles
        self.histogram_plot.setYRange(0, 10)   # Adjust range as necessary
        self.histogram_plot.setMouseEnabled(x=False, y=True)
        self.histogram_hist = pg.BarGraphItem(x=[], height=[], width=10, brush='b')
        self.histogram_plot.addItem(self.histogram_hist)
        self.histogram_line = self.histogram_plot.plot([], [], pen=pg.mkPen(color='b', width=2))

        # Layout setup
        layout = QtWidgets.QGridLayout(self)
        layout.addWidget(self.lidar_plot_widget, 0, 0)  # Top-left
        layout.addWidget(self.camera_label, 0, 1)  # Top-right
        layout.addWidget(self.histogram_plot_widget, 1, 0)  # Bottom-left
        # Bottom-right is empty for now
        layout.setSpacing(0)  # Remove spacing between widgets
        layout.setContentsMargins(0, 0, 0, 0)  # Remove layout margins

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

        # Angle limits
        angles = [51, 309]
        for angle in angles:
            x = 500 * sin(radians(angle))
            y = 500 * cos(radians(angle))
            line = pg.PlotDataItem([0, x], [0, y], pen=pg.mkPen(color=(0, 0, 255), width=2))
            self.lidar_plot.addItem(line)
            fov_lines.append(line)
        return fov_lines

    def update_lidar_plot(self, x, y, distances, angles):
        self.lidar_plot_data.setData(x, y)

        for i in range(36):
            angle_target = i * 10
            mask = (angles >= angle_target - 5) & (angles < angle_target + 5)
            valid_distances = distances[mask]
            valid_x = x[mask]
            valid_y = y[mask]

            if valid_distances.size > 0:
                distance = valid_distances[0]  # Use the first distance in the segment
                x_obj = valid_x[0]
                y_obj = valid_y[0]

                if (angle_target < 51 or angle_target > 309):  # Only plot within the range <51 and >309
                    self.lines[i].setData([0, x_obj], [0, y_obj])
                    self.text_items[i].setText(f'{distance:.2f} cm')
                    self.text_items[i].setPos(x_obj / 2, y_obj / 2)
                else:
                    self.lines[i].setData([], [])
                    self.text_items[i].setText('')
            else:
                self.lines[i].setData([], [])
                self.text_items[i].setText('')

    def update_histogram_plot(self, x, y, distances, angles):
        # Transform angles
        transformed_angles = np.where(
            angles > 309,
            angles - 360,  # Map 310 to -50, 320 to -40, etc.
            angles  # Angles < 51 stay as they are
        )

        # Filter angles and distances based on the transformed angles
        filtered_mask = (angles > 309) | (angles < 51)
        filtered_angles = transformed_angles[filtered_mask]
        filtered_distances = distances[filtered_mask] / 2000

        # Define histogram bins
        bin_edges = np.arange(-180, 361, 10)  # Include negative values and positive values
        hist, _ = np.histogram(filtered_angles, bins=bin_edges, weights=filtered_distances)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        # Update histogram plot
        self.histogram_hist.setOpts(x=bin_centers, height=hist)
        self.histogram_plot.setXRange(-60, 60)  # Adjust x-axis range to include negative values

        # Update histogram line plot
        self.histogram_line.setData(bin_centers, hist)

    def update_camera_feed(self, frame):
        rgb_image = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        self.camera_label.setPixmap(QtGui.QPixmap.fromImage(qt_image))

    def closeEvent(self, event):
        self.lidar_thread.stop()
        self.camera_thread.stop()
        event.accept()

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
