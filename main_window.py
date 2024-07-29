from PySide6 import QtWidgets, QtGui
import pyqtgraph as pg
import numpy as np
from camera_thread import CameraThread
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
        layout.addWidget(QtWidgets.QLabel(''), 1, 1)
        self.setLayout(layout)

    def setup_lidar_plot(self):
        '''Create a plot widget'''
        plot_widget = pg.GraphicsLayoutWidget()
        plot_widget.setFixedSize(600, 450)
        plot = plot_widget.addPlot(title="LiDAR Points")
        plot.setAspectLocked(True)
        plot.setXRange(-500, 500)
        plot.setYRange(-500, 500)

        '''Create a scatter plot for LiDAR data'''
        self.lidar_plot_data = plot.plot([], [], pen=None, symbolBrush=(255, 0, 0), symbolSize=3, symbolPen=None)

        '''Create lines for each angle'''
        self.lines = []
        for _ in range(36):
            self.lines.append(plot.plot([], [], pen=pg.mkPen(color=(0, 255, 0), width=2)))
        # self.lines = [plot.plot([], [], pen=pg.mkPen(color=(0, 255, 0), width=2)) for _ in range(36)

        '''Create text items for each angle'''
        self.text_items = []
        for _ in range(36):
            self.text_items.append(pg.TextItem('', anchor=(0.5, 1)))
        # self.text_items = [pg.TextItem('', anchor=(0.5, 1)) for _ in range(36)]
        for item in self.text_items:
            plot.addItem(item)

        '''Create lines for field of view'''
        self.fov_lines = self.add_fov_lines(plot)

        return plot_widget

    def setup_camera_label(self):
        label = QtWidgets.QLabel()
        label.setFixedSize(600, 450)
        label.setScaledContents(True)

        return label

    def setup_histogram_plot(self):
        '''Create a plot widget for the histogram'''
        plot_widget = pg.GraphicsLayoutWidget()
        plot_widget.setFixedSize(600, 450)
        plot = plot_widget.addPlot(title="Distance Histogram")
        plot.setLabel('bottom', 'Angle (degrees)')
        plot.setLabel('left', 'Distance (divided by 1000)')
        plot.setXRange(-60, 60)
        plot.setYRange(0, 10)
        plot.setMouseEnabled(x=False, y=True) # Disable panning in x-direction
        self.histogram_hist = pg.BarGraphItem(x=[], height=[], width=10, brush='b')
        plot.addItem(self.histogram_hist)
        self.histogram_line = plot.plot([], [], pen=pg.mkPen(color='b', width=2))

        return plot_widget

    def setup_threads(self):
        self.camera_thread = CameraThread()
        self.camera_thread.new_frame.connect(self.update_camera_feed)
        self.camera_thread.new_lidar_data.connect(self.update_lidar_plot)
        self.camera_thread.new_lidar_data.connect(self.update_histogram_plot)
        self.camera_thread.start()

    def update_camera_feed(self, frame):
        height, width, channel = frame.shape
        bytes_per_line = 3 * width
        qt_image = QtGui.QImage(frame.data, width, height, bytes_per_line, QtGui.QImage.Format_RGB888).rgbSwapped()
        self.camera_label.setPixmap(QtGui.QPixmap.fromImage(qt_image))

    def add_fov_lines(self, plot):
        fov_lines = []
        for angle in [51, 309, 0]:
            x = 500 * np.sin(np.radians(angle))
            y = 500 * np.cos(np.radians(angle))
            line = pg.PlotDataItem([0, x], [0, y], pen=pg.mkPen(color=(0, 0, 255), width=2))
            plot.addItem(line)
            fov_lines.append(line)
        return fov_lines
    
    # def update_lidar_plot(self, x, y, distances, angles):
    #     self.lidar_plot_data.setData(x, y)
    #     for i, line in enumerate(self.lines):
    #         angle_target = i * 10
    #         mask = (angles >= angle_target - 5) & (angles < angle_target + 5)
    #         valid_distances = distances[mask]
    #         valid_x, valid_y = x[mask], y[mask]
    #         if valid_distances.size > 0 and (angle_target < 51 or angle_target > 309):
    #             line.setData([0, valid_x[0]], [0, valid_y[0]])
    #             self.text_items[i].setText(f'{valid_distances[0]:.2f} cm')
    #             self.text_items[i].setPos(valid_x[0] / 2, valid_y[0] / 2)
    #         else:
    #             line.setData([], [])
    #             self.text_items[i].setText('')
    
    # def update_lidar_plot(self, x, y, distances, angles):
    #     '''Plot the graph of LiDAR data'''
    #     self.lidar_plot_data.setData(x, y)


    #     '''Update the lines and text items for each angle'''
    #     for i, line in enumerate(self.lines):
    #         angle_target = i * 10
    #         valid_x, valid_y, valid_distances = [], [], []
    #         for j, angle in enumerate(angles):
    #             if angle_target - 5 <= angle < angle_target + 5:
    #                 valid_x.append(x[j])
    #                 valid_y.append(y[j])
    #                 valid_distances.append(distances[j])
    #         if valid_distances and (angle_target < 51 or angle_target > 309):
    #             line.setData([0, valid_x[0]], [0, valid_y[0]])
    #             self.text_items[i].setText(f'{valid_distances[0]:.2f} cm')
    #             self.text_items[i].setPos(valid_x[0] / 2, valid_y[0] / 2)
    #         else:
    #             line.setData([], [])
    #             self.text_items[i].setText('')

    def update_lidar_plot(self, x, y, distances, angles):
        '''Plot the graph of LiDAR data'''
        self.lidar_plot_data.setData(x, y)
        
        target_angles = [310, 320, 330, 340, 350, 360, 0, 10, 20, 30, 40, 50]
        distance_values = np.zeros(12)  # Array to store distances at specified angles
        
        '''Here we are trying to first get all the angles in +- 5 range of the specified angles
        and then we are trying to get the closest object to the LiDAR sensor at that angle'''
        for i, angle_target in enumerate(target_angles):
            valid_x, valid_y, valid_distances = [], [], []
            for j, angle in enumerate(angles):
                if angle_target - 5 <= angle < angle_target + 5 or (angle_target == 360 and 355 <= angle < 360) or (angle_target == 0 and 0 <= angle < 5):
                    valid_x.append(x[j])
                    valid_y.append(y[j])
                    valid_distances.append(distances[j])
            
            if valid_distances:
                min_distance_index = valid_distances.index(min(valid_distances))  # Get the index of the closest object
                closest_x = valid_x[min_distance_index]
                closest_y = valid_y[min_distance_index]
                closest_distance = valid_distances[min_distance_index]
                
                line = self.lines[i]
                line.setData([0, closest_x], [0, closest_y])
                self.text_items[i].setText(f'{closest_distance:.2f} cm')
                self.text_items[i].setPos(closest_x / 2, closest_y / 2)
                
                distance_values[i] = closest_distance  # Store the closest distance in the array
            else:
                line = self.lines[i]
                line.setData([], [])
                self.text_items[i].setText('')
                distance_values[i] = None  # No valid distance found
        
        # print("Distance values at specified angles:", distance_values)
        return distance_values  # Optionally return the array


    def update_histogram_plot(self, x, y, distances, angles):
        # Adjust angles to be in the range of -180 to 180 degrees
        '''If angles is greater than 309, subtract 360 from the angle, otherwise keep the angle as is'''
        transformed_angles = np.where(angles > 309, angles - 360, angles)

        '''Filter out angles that are not within the range of 51 to 309 degrees'''
        filtered_mask = (angles > 309) | (angles < 51)

        '''Get the filtered angles and distances'''
        filtered_angles = transformed_angles[filtered_mask]
        filtered_distances = distances[filtered_mask] / 1000

        # Create histogram
        hist, bin_edges = np.histogram(filtered_angles, bins=np.arange(-180, 361, 10), weights=filtered_distances)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        # Update the histogram plot
        self.histogram_hist.setOpts(x=bin_centers, height=hist, width=5)
        self.histogram_line.setData(bin_centers, hist)

    def closeEvent(self, event):
        self.camera_thread.stop()
        event.accept()

if __name__ == '__main__':
    app = QtWidgets.QApplication([])
    window = MainWindow()
    window.show()
    app.exec()