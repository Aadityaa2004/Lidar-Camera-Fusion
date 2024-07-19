import sys
import cv2 as cv
import numpy as np
from math import cos, sin, radians
from PySide6 import QtCore, QtWidgets, QtGui  # Added QtGui import
import pyqtgraph as pg
from pyrplidar import PyRPlidar

class LidarThread(QtCore.QThread):
    new_data = QtCore.Signal(np.ndarray, np.ndarray)

    def __init__(self, port, baudrate):
        super().__init__()
        self.lidar = PyRPlidar()
        self.port = port
        self.baudrate = baudrate
        self.stop_flag = False

    def run(self):
        self.lidar.connect(port=self.port, baudrate=self.baudrate, timeout=3)
        self.lidar.set_motor_pwm(500)
        scan_generator = self.lidar.force_scan()

        while not self.stop_flag:
            x_coords = []
            y_coords = []

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

                if count >= 360:  # Emit data every 100 points
                    self.new_data.emit(np.array(x_coords), np.array(y_coords))
                    x_coords = []
                    y_coords = []
                    break

        self.lidar.stop()
        self.lidar.set_motor_pwm(0)
        self.lidar.disconnect()

    def new_data_test(self):
        print(self.new_data)

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
        self.setWindowTitle('LiDAR and Camera Feed')
        self.resize(1600, 800)

        # LiDAR plot setup
        self.lidar_plot_widget = pg.GraphicsLayoutWidget()
        self.lidar_plot_widget.setFixedSize(800, 800)  # Set fixed size
        self.lidar_plot = self.lidar_plot_widget.addPlot(title="LiDAR Points")
        self.lidar_plot.setAspectLocked(True)
        self.lidar_plot.setXRange(-500, 500)
        self.lidar_plot.setYRange(-500, 500)
        self.lidar_plot_data = self.lidar_plot.plot([], [], pen=None, symbolBrush=(255, 0, 0), symbolSize=3, symbolPen=None)

        # Camera label setup
        self.camera_label = QtWidgets.QLabel()
        self.camera_label.setFixedSize(800, 800)  # Set fixed size
        self.camera_label.setScaledContents(True)  # Scale the image to fit the label

        # Layout setup
        layout = QtWidgets.QHBoxLayout(self)
        layout.addWidget(self.lidar_plot_widget)
        layout.addWidget(self.camera_label)
        layout.setSpacing(0)  # Remove spacing between widgets
        layout.setContentsMargins(0, 0, 0, 0)  # Remove layout margins

        self.setLayout(layout)

        # Thread setup
        self.lidar_thread = LidarThread(port="/dev/tty.usbserial-0001", baudrate=256000)
        self.lidar_thread.new_data.connect(self.update_lidar_plot)
        self.lidar_thread.start()
        
        self.camera_thread = CameraThread()
        self.camera_thread.new_frame.connect(self.update_camera_feed)
        self.camera_thread.start()

    def update_lidar_plot(self, x, y):
        self.lidar_plot_data.setData(x, y)

    def update_camera_feed(self, frame):
        frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        height, width, channel = frame.shape
        bytes_per_line = 3 * width
        q_image = QtGui.QImage(frame.data, width, height, bytes_per_line, QtGui.QImage.Format_RGB888)
        self.camera_label.setPixmap(QtGui.QPixmap.fromImage(q_image))

    def closeEvent(self, event):
        self.lidar_thread.stop()
        self.camera_thread.stop()
        event.accept()

if __name__ == "__main__":
    app = QtWidgets.QApplication([])

    main_window = MainWindow()
    main_window.show()

    sys.exit(app.exec())
