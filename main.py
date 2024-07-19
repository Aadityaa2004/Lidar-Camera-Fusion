import sys
import cv2 as cv
import numpy as np
from math import cos, sin, radians
from PySide6 import QtCore, QtWidgets, QtGui  # Added QtGui import
import pyqtgraph as pg
from pyrplidar import PyRPlidar

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
            angles_and_distances = []

            for count, scan in enumerate(scan_generator()):
                line = str(scan)
                line = line.replace('{', ' ').replace('}', ' ').split(',')
                angle = float(line[2].split(':')[1])
                distance = float(line[3].split(':')[1]) / 10
                # if angles < 60:
                #     angles_and_distances.append(angle[count], distance[count])
                # if angles > 300:
                #     angles_and_distances.append(angle[count], distance[count])
                if distance > 0: 
                        if angle < 51 or angle > 309:
                            x = distance * sin(radians(angle))
                            y = distance * cos(radians(angle))
                            x_coords.append(x)
                            y_coords.append(y)
                            angles.append(angle)
                            distances.append(distance)
            

                if count >= 360:  # Emit data every 100 points
                    self.new_data.emit(np.array(x_coords), np.array(y_coords), np.array(angles), np.array(distances))
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

        self.lines = []
        self.text_items = []
        for _ in range(72):
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
        
        # self.camera_thread = CameraThread()
        # self.camera_thread.new_frame.connect(self.update_camera_feed)
        # self.camera_thread.start()

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

        # Draw connecting line between endpoints of FOV lines
        x1, y1 = 500 * sin(radians(51)), 500 * cos(radians(51))
        x2, y2 = 500 * sin(radians(309)), 500 * cos(radians(309))
        outline_line = pg.PlotDataItem([x1, x2], [y1, y2], pen=pg.mkPen(color=(0, 0, 255), width=2))
        self.lidar_plot.addItem(outline_line)
        fov_lines.append(outline_line)

        return fov_lines

    def update_lidar_plot(self, x, y, distances, angles):
        self.lidar_plot_data.setData(x, y)

        for i in range(72):
            angle_target = i * 5
            mask = (angles >= angle_target - 2.5) & (angles < angle_target + 2.5)
            valid_distances = distances[mask]
            valid_x = x[mask]
            valid_y = y[mask]
            
            if valid_distances.size > 0:
                min_distance_idx = np.argmin(valid_distances) # finds the index of the minimum distance
                min_distance = valid_distances[min_distance_idx] # finds the minimum distance
                x_obj = valid_x[min_distance_idx]
                y_obj = valid_y[min_distance_idx]
                
                self.lines[i].setData([0, x_obj], [0, y_obj])
                self.text_items[i].setText(f'{min_distance:.2f} cm')
                self.text_items[i].setPos(x_obj / 2, y_obj / 2)
            else:
                self.lines[i].setData([], [])
                self.text_items[i].setText('')
                

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
