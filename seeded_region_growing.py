import sys
import numpy as np
from PySide6 import QtWidgets, QtCore
import pyqtgraph as pg
from math import sin, cos, radians, sqrt

# Import your existing classes
from main import LidarThread, MainWindow, CameraThread

class SeededRegionGrowing:
    def __init__(self):
        self.EPSILON = 0.03  # Distance threshold from point to line
        self.DELTA = 0.1  # Distance threshold from point to point
        self.SNUM = 6  # Number of points in a seed segment
        self.PMIN = 10  # Minimum number of points in a line segment
        self.LMIN = 0.6  # Minimum length of a line segment
        self.GMAX = 0.1  # Maximum gap between points

    def dist_point2point(self, point1, point2):
        return sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

    def dist_point2line(self, params, point):
        A, B, C = params
        return abs(A * point[0] + B * point[1] + C) / sqrt(A**2 + B**2)

    def fit_line(self, points):
        x = [p[0] for p in points]
        y = [p[1] for p in points]
        A = np.vstack([x, np.ones(len(x))]).T
        m, c = np.linalg.lstsq(A, y, rcond=None)[0]
        return m, c

    def line_params(self, m, c):
        # Convert to general form Ax + By + C = 0
        A = -m
        B = 1
        C = -c
        return A, B, C

    def seed_segment_detection(self, laser_points):
        for i in range(len(laser_points) - self.SNUM):
            seed_segment = laser_points[i:i+self.SNUM]
            m, c = self.fit_line(seed_segment)
            
            A, B, C = self.line_params(m, c)
            
            is_valid_seed = True
            for point in seed_segment:
                if self.dist_point2line((A, B, C), point) > self.EPSILON:
                    is_valid_seed = False
                    break
            
            if is_valid_seed:
                return seed_segment, (A, B, C)
        
        return None, None

    def region_growing(self, laser_points, seed_segment, line_params):
        if seed_segment is None:
            return None

        line_segment = seed_segment.copy()
        start_index = laser_points.index(seed_segment[0])
        end_index = laser_points.index(seed_segment[-1])

        # Grow forward
        for i in range(end_index + 1, len(laser_points)):
            if self.dist_point2line(line_params, laser_points[i]) <= self.EPSILON:
                if self.dist_point2point(laser_points[i], line_segment[-1]) <= self.GMAX:
                    line_segment.append(laser_points[i])
                else:
                    break
            else:
                break

        # Grow backward
        for i in range(start_index - 1, -1, -1):
            if self.dist_point2line(line_params, laser_points[i]) <= self.EPSILON:
                if self.dist_point2point(laser_points[i], line_segment[0]) <= self.GMAX:
                    line_segment.insert(0, laser_points[i])
                else:
                    break
            else:
                break

        if len(line_segment) >= self.PMIN:
            if self.dist_point2point(line_segment[0], line_segment[-1]) >= self.LMIN:
                return line_segment

        return None

    def extract_line_segments(self, laser_points):
        line_segments = []
        remaining_points = laser_points.copy()

        while len(remaining_points) > self.SNUM:
            seed_segment, line_params = self.seed_segment_detection(remaining_points)
            if seed_segment is None:
                break

            line_segment = self.region_growing(remaining_points, seed_segment, line_params)
            if line_segment:
                line_segments.append(line_segment)
                remaining_points = [p for p in remaining_points if p not in line_segment]
            else:
                remaining_points = remaining_points[1:]

        return line_segments

class SeededRegionGrowingWindow(MainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('LiDAR Seeded Region Growing')
        self.srg = SeededRegionGrowing()

        # Modify the update_lidar_plot method to use seeded region growing
        self.lidar_thread.new_data.disconnect(self.update_lidar_plot)
        self.lidar_thread.new_data.connect(self.update_lidar_plot_with_srg)

    def update_lidar_plot_with_srg(self, x, y, distances, angles):
        self.lidar_plot_data.setData(x, y)

        laser_points = list(zip(x, y))
        line_segments = self.srg.extract_line_segments(laser_points)

        # Clear previous lines
        for line in self.lines:
            line.setData([], [])

        # Plot new line segments
        for i, segment in enumerate(line_segments):
            if i < len(self.lines):
                x_seg, y_seg = zip(*segment)
                self.lines[i].setData(x_seg, y_seg)

if __name__ == "__main__":
    app = QtWidgets.QApplication([])
    main_window = SeededRegionGrowingWindow()
    main_window.show()
    sys.exit(app.exec())
