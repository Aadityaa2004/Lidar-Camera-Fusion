# import sys
# from PyQt5 import QtWidgets
# import pyqtgraph as pg

# # Create the application
# app = QtWidgets.QApplication(sys.argv)

# # Set configuration options for pyqtgraph
# pg.setConfigOptions(antialias=True)

# # Create a window and a plot
# win = pg.GraphicsLayoutWidget(show=True, title="PyQtGraph Example")
# plot = win.addPlot(title="Antialiased Plot")

# # Add some data to the plot
# x = [1, 2, 3, 4, 5]
# y = [10, 20, 15, 30, 25]
# plot.plot(x, y)

# # Start the Qt event loop
# sys.exit(app.exec_())

import sys
import random
from PySide6 import QtCore, QtWidgets, QtGui

class MyWidget(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()

        self.hello = ["Hallo Welt", "Hei maailma", "Hola Mundo", "Привет мир"]

        self.button = QtWidgets.QPushButton("Click me!")
        self.text = QtWidgets.QLabel("Hello World",
                                     alignment=QtCore.Qt.AlignCenter)

        self.layout = QtWidgets.QVBoxLayout(self)
        self.layout.addWidget(self.text)
        self.layout.addWidget(self.button)

        self.button.clicked.connect(self.magic)

    @QtCore.Slot()
    def magic(self):
        self.text.setText(random.choice(self.hello))

if __name__ == "__main__":
    app = QtWidgets.QApplication([])

    widget = MyWidget()
    widget.resize(800, 600)
    widget.show()

    sys.exit(app.exec())