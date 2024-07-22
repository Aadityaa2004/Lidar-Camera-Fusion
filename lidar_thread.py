from PySide6 import QtCore
import numpy as np
from math import cos, sin, radians
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
            x_coords, y_coords, angles, distances = [], [], [], []

            for count, scan in enumerate(scan_generator()):
                angle, distance = self.parse_scan(scan)
                if distance > 0:
                    x = distance * sin(radians(angle))
                    y = distance * cos(radians(angle))
                    x_coords.append(x)
                    y_coords.append(y)
                    angles.append(angle)
                    distances.append(distance)

                if count >= 360:
                    self.new_data.emit(np.array(x_coords), np.array(y_coords), 
                                       np.array(distances), np.array(angles))
                    break

        self.cleanup()

    def parse_scan(self, scan):
        line = str(scan).replace('{', ' ').replace('}', ' ').split(',')
        return float(line[2].split(':')[1]), float(line[3].split(':')[1]) / 10

    def cleanup(self):
        self.lidar.stop()
        self.lidar.set_motor_pwm(0)
        self.lidar.disconnect()

    def stop(self):
        self.stop_flag = True
        self.wait()