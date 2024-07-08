import sys
import numpy as np
from PyQt5 import QtCore, QtWidgets
import pyqtgraph as pg
import cv2 as cv
import time
from lidar import LidarThread, handDetector

class IntegratedPlotting(pg.GraphicsLayoutWidget):
    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.setWindowTitle('Integrated Lidar and Camera Data')
        self.resize(1600, 600)
        
        # LIDAR plot
        self.lidar_plot = self.addPlot(title="Lidar Points")
        self.lidar_plot.setAspectLocked(True)
        self.lidar_plot.setXRange(-500, 500)
        self.lidar_plot.setYRange(-500, 500)
        self.plot_data = self.lidar_plot.plot([], [], pen=None, symbolBrush=(255, 0, 0), symbolSize=3, symbolPen=None)
        
        self.lines = []
        self.text_items = []
        for _ in range(12):
            line = pg.PlotDataItem([], [], pen=pg.mkPen(color=(0, 255, 0), width=2))
            self.lidar_plot.addItem(line)
            self.lines.append(line)
            
            text_item = pg.TextItem('', anchor=(0.5, 1))
            self.lidar_plot.addItem(text_item)
            self.text_items.append(text_item)

        # Camera view
        self.nextRow()
        self.camera_view = self.addViewBox()
        self.camera_image = pg.ImageItem()
        self.camera_view.addItem(self.camera_image)

        # Hand position highlight
        self.hand_highlight = pg.ScatterPlotItem(size=20, brush=pg.mkBrush(0, 255, 0, 120))
        self.lidar_plot.addItem(self.hand_highlight)

    def update_lidar(self, x, y, distances, angles, hand_position):
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
        
        if hand_position:
            self.hand_highlight.setData([hand_position[0]], [hand_position[1]])
        else:
            self.hand_highlight.setData([], [])

    def update_camera(self, frame):
        self.camera_image.setImage(frame.swapaxes(0, 1))

    def closeEvent(self, event):
        # This method will be called when the window is closed
        event.accept()

class LidarCameraIntegration(QtCore.QObject):
    update_lidar_signal = QtCore.pyqtSignal(np.ndarray, np.ndarray, np.ndarray, np.ndarray, list)
    update_camera_signal = QtCore.pyqtSignal(np.ndarray)

    def __init__(self, lidar_port, lidar_baudrate):
        super().__init__()
        self.lidar_thread = LidarThread(port=lidar_port, baudrate=lidar_baudrate)
        self.hand_detector = handDetector()
        self.camera = cv.VideoCapture(0)
        self.camera.set(3, 640)  # Width
        self.camera.set(4, 480)  # Height
        self.stop_flag = False
        self.hand_position = None

    def run(self):
        self.lidar_thread.new_data.connect(self.process_lidar_data)
        self.lidar_thread.start()

        while not self.stop_flag:
            ret, frame = self.camera.read()
            if not ret:
                print("Failed to grab frame")
                break

            frame, hand_position = self.hand_detector.findHands(frame, return_position=True)
            frame = cv.flip(frame, 0)  # Flip the frame vertically
            self.update_camera_signal.emit(frame)

            if hand_position:
                self.process_hand_position(hand_position)

            if cv.waitKey(1) & 0xFF == ord('q'):
                break

        self.stop()

    def process_lidar_data(self, x, y, distances, angles):
        self.update_lidar_signal.emit(x, y, distances, angles, self.hand_position)

    def process_hand_position(self, hand_position):
        x = hand_position[0] - 320  # Assume camera center is at (320, 240)
        y = -(hand_position[1] - 240)  # Invert Y axis to match LIDAR coordinates
        self.hand_position = [x, y]

    def stop(self):
        self.stop_flag = True
        self.lidar_thread.stop()
        self.camera.release()
        cv.destroyAllWindows()

def main():
    app = QtWidgets.QApplication(sys.argv)
    pg.setConfigOptions(antialias=True)
    
    integrated_widget = IntegratedPlotting()
    integrated_widget.show()

    integration = LidarCameraIntegration("/dev/tty.usbserial-0001", 256000)
    integration.update_lidar_signal.connect(integrated_widget.update_lidar)
    integration.update_camera_signal.connect(integrated_widget.update_camera)

    integration_thread = QtCore.QThread()
    integration.moveToThread(integration_thread)
    integration_thread.started.connect(integration.run)
    integration_thread.start()

    sys.exit(app.exec_())

if __name__ == "__main__":
    main()