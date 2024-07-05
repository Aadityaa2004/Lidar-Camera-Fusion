import sys
import numpy as np
from PyQt5 import QtCore, QtWidgets
import pyqtgraph as pg
from pyrplidar import PyRPlidar
from math import cos, sin, radians

class LidarThread(QtCore.QThread):
    new_data = QtCore.pyqtSignal(np.ndarray, np.ndarray, np.ndarray, np.ndarray)

    def __init__(self, port, baudrate):
        QtCore.QThread.__init__(self)
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
            distances = []
            angles = []
            for _ in range(360):
                if self.stop_flag:
                    break
                scan = next(scan_generator())
                angle = scan.angle
                distance = scan.distance / 10
                if distance > 0:
                    x = distance * sin(radians(angle))
                    y = distance * cos(radians(angle))
                    x_coords.append(x)
                    y_coords.append(y)
                    distances.append(distance)
                    angles.append(angle)
            
            if x_coords and y_coords:
                self.new_data.emit(np.array(x_coords), np.array(y_coords), np.array(distances), np.array(angles))

        self.lidar.stop()
        self.lidar.set_motor_pwm(0)
        self.lidar.disconnect()

    def stop(self):
        self.stop_flag = True
        self.wait()

class Plotting(pg.GraphicsLayoutWidget):
    def __init__(self, parent=None):
        pg.GraphicsLayoutWidget.__init__(self, parent=parent)
        self.setWindowTitle('Real-time Lidar Points')
        self.resize(800, 600)
        self.plot = self.addPlot(title="Lidar Points")
        self.plot.setAspectLocked(True)
        self.plot.setXRange(-500, 500)
        self.plot.setYRange(-500, 500)
        self.plot_data = self.plot.plot([], [], pen=None, symbolBrush=(255, 0, 0), symbolSize=3, symbolPen=None)
        
        self.lines = []
        self.text_items = []
        for _ in range(12):
            line = pg.PlotDataItem([], [], pen=pg.mkPen(color=(0, 255, 0), width=2))
            self.plot.addItem(line)
            self.lines.append(line)
            
            text_item = pg.TextItem('', anchor=(0.5, 1))
            self.plot.addItem(text_item)
            self.text_items.append(text_item)
        
        self.lidar_thread = LidarThread(port="/dev/tty.usbserial-0001", baudrate=256000)
        self.lidar_thread.new_data.connect(self.update_plot)
        self.lidar_thread.start()

    def update_plot(self, x, y, distances, angles):
        self.plot_data.setData(x, y)
        
        for i in range(12):
            angle_target = i * 30
            mask = (angles >= angle_target - 15) & (angles <= angle_target + 15)
            valid_distances = distances[mask]
            valid_x = x[mask]
            valid_y = y[mask]
            
            if valid_distances.size > 0:
                min_distance_idx = np.argmin(valid_distances)
                min_distance = valid_distances[min_distance_idx]
                x_obj = valid_x[min_distance_idx]
                y_obj = valid_y[min_distance_idx]
                
                self.lines[i].setData([0, x_obj], [0, y_obj])
                self.text_items[i].setText(f'{min_distance:.2f} cm')
                self.text_items[i].setPos(x_obj / 2, y_obj / 2)
            else:
                self.lines[i].setData([], [])
                self.text_items[i].setText('')

    def closeEvent(self, event):
        self.lidar_thread.stop()
        event.accept()

def main():
    app = QtWidgets.QApplication(sys.argv)
    pg.setConfigOptions(antialias=True)
    lidar_widget = Plotting()
    lidar_widget.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
