# ##################################
# Copyright (c) 2021, Mattis Jaksch
#
# This Python file uses the following encoding: utf-8
# ##################################

import sys
from PySide6.QtWidgets import QApplication, QMainWindow, QPushButton, QWidget, QTabWidget, QVBoxLayout
from PySide6.QtGui import QFont

import SubWindow

class Main(QMainWindow):
	def __init__(self, arguments):
		super().__init__()
		self.title = "Neural Network - Prediction Probability"
		self.left = 80
		self.top = 60
		self.width = 800
		self.height = 600
		self.setFont(QFont('hack', 16))
		self.setWindowTitle(self.title)
		self.setGeometry(self.left, self.top, self.width, self.height)

		self.table_widget = TableWidget(self, arguments)
		self.setCentralWidget(self.table_widget)

		self.show()

class TableWidget(QWidget):
	def __init__(self, parent, arguments):
		super(TableWidget, self).__init__(parent)
		self.layout = QVBoxLayout(self)

		# Initialize tabs
		self.tabs = QTabWidget()
		self.tab_dat = SubWindow.DataInput(self, arguments[1])
		self.tab_nn = SubWindow.NeuralNetwork(self, self.tab_dat)
		self.tab_pre = SubWindow.Prediction(self, self.tab_dat, self.tab_nn)

		# Add tabs
		self.tabs.addTab(self.tab_dat, "Data")
		self.tabs.addTab(self.tab_nn, "Network")
		self.tabs.addTab(self.tab_pre, "Prediction")

		# Create data tab
#		self.tab_dat.layout = QVBoxLayout(self)
#		self.pushButton1 = QPushButton("Some Button")
#		self.tab_dat.layout.addWidget(self.pushButton1)
#		self.tab_dat.setLayout(self.tab_dat.layout)

		# Add tabs to widget
		self.layout.addWidget(self.tabs)
		self.setLayout(self.layout)

if __name__ == "__main__":
	app = QApplication(sys.argv)
	window = Main(app.arguments())
	window.show()
	sys.exit(app.exec())
