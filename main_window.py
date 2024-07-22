from PySide6 import QtWidgets, QtGui
import pyqtgraph as pg
import numpy as np
from lidar_thread import LidarThread
from camera_thread import CameraThread
from config import LIDAR_PORT, LIDAR_BAUDRATE
import cv2 as cv

class MainWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setup_ui()
        self.setup_threads()

    def setup_ui(self):
        self.setWindowTitle('LiDAR, Camera Feed, and Histogram')
        self.resize(1200, 900)

        self.lidar_plot_widget = self.setup_lidar_plot()
        self.camera_label = self.setup_camera_label()
        self.histogram_plot_widget = self.setup_histogram_plot()

        layout = QtWidgets.QGridLayout(self)
        layout.addWidget(self.lidar_plot_widget, 0, 0)
        layout.addWidget(self.camera_label, 0, 1)
        layout.addWidget(self.histogram_plot_widget, 1, 0)
        layout.setSpacing(0)
        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)

    def setup_lidar_plot(self):
        plot_widget = pg.GraphicsLayoutWidget()
        plot_widget.setFixedSize(600, 450)
        plot = plot_widget.addPlot(title="LiDAR Points")
        plot.setAspectLocked(True)
        plot.setXRange(-500, 500)
        plot.setYRange(-500, 500)
        self.lidar_plot_data = plot.plot([], [], pen=None, symbolBrush=(255, 0, 0), symbolSize=3, symbolPen=None)

        self.lines = [plot.plot([], [], pen=pg.mkPen(color=(0, 255, 0), width=2)) for _ in range(36)]
        self.text_items = [pg.TextItem('', anchor=(0.5, 1)) for _ in range(36)]
        for item in self.text_items:
            plot.addItem(item)

        self.fov_lines = self.add_fov_lines(plot)
        return plot_widget

    def setup_camera_label(self):
        label = QtWidgets.QLabel()
        label.setFixedSize(600, 450)
        label.setScaledContents(True)
        return label

    def setup_histogram_plot(self):
        plot_widget = pg.GraphicsLayoutWidget()
        plot_widget.setFixedSize(600, 450)
        plot = plot_widget.addPlot(title="Distance Histogram")
        plot.setLabel('bottom', 'Angle (degrees)')
        plot.setLabel('left', 'Distance (cm)')
        plot.setXRange(-60, 60)
        plot.setYRange(0, 10)
        plot.setMouseEnabled(x=False, y=True)
        self.histogram_hist = pg.BarGraphItem(x=[], height=[], width=10, brush='b')
        plot.addItem(self.histogram_hist)
        self.histogram_line = plot.plot([], [], pen=pg.mkPen(color='b', width=2))
        return plot_widget

    def setup_threads(self):
        self.lidar_thread = LidarThread(port=LIDAR_PORT, baudrate=LIDAR_BAUDRATE)
        self.lidar_thread.new_data.connect(self.update_lidar_plot)
        self.lidar_thread.new_data.connect(self.update_histogram_plot)
        self.lidar_thread.start()
        
        self.camera_thread = CameraThread()
        self.camera_thread.new_frame.connect(self.update_camera_feed)
        self.camera_thread.start()

    def add_fov_lines(self, plot):
        fov_lines = []
        for angle in [51, 309]:
            x = 500 * np.sin(np.radians(angle))
            y = 500 * np.cos(np.radians(angle))
            line = pg.PlotDataItem([0, x], [0, y], pen=pg.mkPen(color=(0, 0, 255), width=2))
            plot.addItem(line)
            fov_lines.append(line)
        return fov_lines

    def update_lidar_plot(self, x, y, distances, angles):
        self.lidar_plot_data.setData(x, y)
        for i, line in enumerate(self.lines):
            angle_target = i * 10
            mask = (angles >= angle_target - 5) & (angles < angle_target + 5)
            valid_distances = distances[mask]
            valid_x, valid_y = x[mask], y[mask]
            if valid_distances.size > 0 and (angle_target < 51 or angle_target > 309):
                line.setData([0, valid_x[0]], [0, valid_y[0]])
                self.text_items[i].setText(f'{valid_distances[0]:.2f} cm')
                self.text_items[i].setPos(valid_x[0] / 2, valid_y[0] / 2)
            else:
                line.setData([], [])
                self.text_items[i].setText('')

    def update_histogram_plot(self, x, y, distances, angles):
        transformed_angles = np.where(angles > 309, angles - 360, angles)
        filtered_mask = (angles > 309) | (angles < 51)
        filtered_angles = transformed_angles[filtered_mask]
        filtered_distances = distances[filtered_mask] / 2000

        hist, bin_edges = np.histogram(filtered_angles, bins=np.arange(-180, 361, 10), weights=filtered_distances)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        self.histogram_hist.setOpts(x=bin_centers, height=hist)
        self.histogram_line.setData(bin_centers, hist)

    def update_camera_feed(self, frame):
        rgb_image = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        qt_image = QtGui.QImage(rgb_image.data, w, h, ch * w, QtGui.QImage.Format_RGB888)
        self.camera_label.setPixmap(QtGui.QPixmap.fromImage(qt_image))

    def closeEvent(self, event):
        self.lidar_thread.stop()
        self.camera_thread.stop()
        event.accept()