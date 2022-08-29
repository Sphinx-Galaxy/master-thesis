# ##################################
# Copyright (c) 2021, Mattis Jaksch
#
# This Python file uses the following encoding: utf-8
# ##################################

from PySide6.QtCore import QObject, Slot
from PySide6.QtWidgets import QWidget, QPushButton, QVBoxLayout, QHBoxLayout, QGridLayout, QLineEdit, QCheckBox, QLabel, QGroupBox, QButtonGroup, QRadioButton, QSpinBox, QSizePolicy, QTextEdit
import pyqtgraph as pg

import tensorflow as tf
import numpy as np

import SubModule

class DataInput(QWidget):
	def __init__(self, parent, filename):
		super(DataInput, self).__init__(parent)

		# Init data module
		self.dataModule = SubModule.DataModule(self, filename, self.update_plot)

		# Vertical main layout
		self.mainLayout = QVBoxLayout(self)

		# Horizontal top layout with cleaning options
		self.topLayout = QHBoxLayout()
		## Options
		self.optionBox = QGroupBox("Options")
		self.optionBox.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Minimum)
		self.optionLayout = QGridLayout()

		self.fftBox = QCheckBox()
		self.fftBox.toggled.connect(self.dataModule.set_fft)
		self.optionLayout.addWidget(QLabel("FFT:"), 0, 0)
		self.optionLayout.addWidget(self.fftBox, 0, 1)
		
		self.smoothBox = QSpinBox()
		self.smoothBox.setRange(0, self.dataModule.get_entries)
		self.smoothBox.setSingleStep(2)
		self.smoothBox.valueChanged.connect(self.dataModule.set_smooth)
		self.fftBox.toggled.connect(self.smoothBox.setReadOnly)
		self.optionLayout.addWidget(QLabel("Smooth:"), 1, 0)
		self.optionLayout.addWidget(self.smoothBox, 1, 1)

		self.validBox = QSpinBox()
		self.validBox.setRange(0, self.dataModule.get_entries)
		self.validBox.setSingleStep(1)
		self.validBox.setValue(int(self.dataModule.get_entries*0.8))
#		self.validBox.valueChanged.connect(self.dataModule.set_smooth)
		self.optionLayout.addWidget(QLabel("Valid:"), 2, 0)
		self.optionLayout.addWidget(self.validBox, 2, 1)
		
		self.testBox = QSpinBox()
		self.testBox.setRange(0, self.dataModule.get_entries)
		self.testBox.setSingleStep(1)
		self.testBox.setValue(int(self.dataModule.get_entries*0.9))
#		self.testBox.valueChanged.connect(self.dataModule.set_smooth)
		self.optionLayout.addWidget(QLabel("Test:"), 3, 0)
		self.optionLayout.addWidget(self.testBox, 3, 1)		

		self.optionBox.setLayout(self.optionLayout)

		## Features
		self.featureBox = QGroupBox("Features")
		self.featureBox.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum)
		self.featureLayout = QGridLayout()

		self.featureLayout.addWidget(QLabel("Deterministic:"), 1, 0)
		self.featureLayout.addWidget(QLabel("Aleatoric:"), 2, 0)
		self.featureLayout.addWidget(QLabel("Prediction:"), 3, 0)
		self.featureLayout.addWidget(QLabel("(None):"), 4, 0)
		self.featureLayout.addWidget(QLabel(""), 5, 0)
		self.featureLayout.addWidget(QLabel("One-Hot:"), 6, 0)

		### Add buttons to the features
#		buttonGroupPrediction = QButtonGroup(self)
		
		buttonGroupOneHot = QButtonGroup(self)
		buttonGroupOneHot.setExclusive(False)

		for i,name in enumerate(self.dataModule.get_features, 2):
			#### One-hot checkbox				
			oneHotBox = QCheckBox()
			oneHotBox.setEnabled(False)
			buttonGroupOneHot.addButton(oneHotBox)
			buttonGroupOneHot.setId(oneHotBox, i-2)
			self.featureLayout.addWidget(oneHotBox, 6, i)
			
			self.featureLayout.addWidget(QLabel(name), 0, i)
			buttonGroupFeature = QButtonGroup(self)

			for j in range(1, 5):
				button = QRadioButton()
				buttonGroupFeature.addButton(button)
				buttonGroupFeature.setId(button, 4*(i-2) + (j-1)) # 4 States: Deterministic, Aleatoric, Prediction, None
				if j == 1: # Deterministic
					button.toggled.connect(oneHotBox.setEnabled)

				self.featureLayout.addWidget(button, j, i)

			buttonGroupFeature.buttonClicked.connect(self.dataModule.set_features)

			
#			#### Select output feature
#			button = QRadioButton()
#			buttonGroupPrediction.addButton(button)
#			self.featureLayout.addWidget(button, 6, i)
#			buttonGroupPrediction.setId(button, i-2)
#			buttonGroupPrediction.buttonClicked.connect(self.dataModule.set_prediction_feature)
	
		buttonGroupOneHot.buttonClicked.connect(self.dataModule.set_one_hot)

		self.featureBox.setLayout(self.featureLayout)

		self.topLayout.addWidget(self.optionBox)
		self.topLayout.addWidget(self.featureBox)

		# Plot to view the data
		self.plotBox = QGroupBox("Plot")
		self.plotLayout = QHBoxLayout()

		self.graph = pg.PlotWidget()
		
		self.update_plot()

		self.plotLayout.addWidget(self.graph)

		self.plotBox.setLayout(self.plotLayout)

		# Put layout together
		self.mainLayout.addLayout(self.topLayout)
		self.mainLayout.addWidget(self.plotBox)
		self.setLayout(self.mainLayout)
		
	@property
	def get_dataModule(self):
		return self.dataModule
		
	def update_plot(self):
		self.graph.clear()
		
		# Plot all the data columns that are not (none)
		for column in self.dataModule.get_features:			
			if self.fftBox.isChecked() == True:
				self.graph.plot(self.dataModule.get_fft_data[column])
			else:
				self.graph.plot(self.dataModule.get_data[column])
		

class NeuralNetwork(QWidget):
	def __init__(self, parent, tab_dat):
		super(NeuralNetwork, self).__init__(parent)

		# Init NN module
		self.nnModule = SubModule.NNModule(self, self.get_modelText, tab_dat.get_dataModule)

		# Vertical main layout
		self.mainLayout = QVBoxLayout(self)

		# Top results
		self.resultBox = QGroupBox("Result")
		self.resultLayout = QHBoxLayout()
		
		self.resultLayout.addWidget(QLabel("Basline:"))
		self.baseLineEdit = QLineEdit()
		self.baseLineEdit.setReadOnly(True)
		self.baseLineEdit.setText(str(0.000000))
		self.nnModule.baseline_changed.connect(self.baseLineEdit.setText)
		self.resultLayout.addWidget(self.baseLineEdit)
		
		self.resultLayout.addWidget(QLabel("Model:"))
		self.modelLineEdit = QLineEdit()
		self.modelLineEdit.setReadOnly(True)
		self.modelLineEdit.setText(str(0.000000))
		self.nnModule.eval_changed.connect(self.modelLineEdit.setText)
		self.resultLayout.addWidget(self.modelLineEdit)

		self.resultBox.setLayout(self.resultLayout)

		# Horizontal top layout with options
		self.optionBox = QGroupBox("Options")
		self.optionLayout = QHBoxLayout()
		
		## Model storage
		self.storageBox = QGroupBox("Storage")
		self.storageLayout = QHBoxLayout()
		
		self.loadButton = QPushButton("Load")
		self.loadButton.clicked.connect(self.nnModule.load)
		self.storageLayout.addWidget(self.loadButton)
		
		self.saveButton = QPushButton("Save")
		self.saveButton.clicked.connect(self.nnModule.save)
		self.storageLayout.addWidget(self.saveButton)
		
		self.storageBox.setLayout(self.storageLayout)
		self.optionLayout.addWidget(self.storageBox)
		
		## Model compilation, training and evaluation
		self.etcBox = QGroupBox("Model")
		self.etcLayout = QHBoxLayout()
		
		self.compileButton = QPushButton("Compile")
		self.compileButton.clicked.connect(self.nnModule.compile)
		self.etcLayout.addWidget(self.compileButton)
		
		self.trainButton = QPushButton("Train")
		self.trainButton.clicked.connect(self.nnModule.train)
		self.etcLayout.addWidget(self.trainButton)

		self.evalButton = QPushButton("Evaluate")
		self.evalButton.clicked.connect(self.nnModule.eval)
		self.etcLayout.addWidget(self.evalButton)

		self.epochBox = QSpinBox()
		self.epochBox.setRange(1, 100)
		self.epochBox.setValue(5)
		self.epochBox.valueChanged.connect(self.nnModule.set_epoch)
		self.etcLayout.addWidget(QLabel("Epoch:"))
		self.etcLayout.addWidget(self.epochBox)

		self.lrBox = QSpinBox()
		self.lrBox.setRange(1, 1000)
		self.lrBox.setValue(20)
		self.lrBox.valueChanged.connect(self.nnModule.set_lr)
		self.etcLayout.addWidget(QLabel("LR (1e-3):"))
		self.etcLayout.addWidget(self.lrBox)
		
		self.pastBox = QSpinBox()
		self.pastBox.setRange(1, 1000)
		self.pastBox.setValue(24)
		self.pastBox.valueChanged.connect(self.nnModule.set_past_points)
		self.etcLayout.addWidget(QLabel("Past Points:"))
		self.etcLayout.addWidget(self.pastBox)
		
		self.futureBox = QSpinBox()
		self.futureBox.setRange(1, 1000)
		self.futureBox.setValue(24)
		self.futureBox.valueChanged.connect(self.nnModule.set_future_points)
		self.etcLayout.addWidget(QLabel("Future Points:"))
		self.etcLayout.addWidget(self.futureBox)
		
		self.etcBox.setLayout(self.etcLayout)
		self.optionLayout.addWidget(self.etcBox)
		
		self.optionBox.setLayout(self.optionLayout)

		# Texteditor for models
		self.modelBox = QGroupBox("Texteditor")
		self.modelLayout = QHBoxLayout()

		self.modelEdit = QTextEdit()
		self.modelLayout.addWidget(self.modelEdit)
		self.modelEdit.setPlainText(
			"encoder = tf.keras.layers.LSTM(24, return_state=True)\n" +
			"encoder_outputs, state_h, state_c = encoder(past_inputs)\n" +
			"\n" +
			"decoder_lstm = tf.keras.layers.LSTM(24, return_sequences=True)\n" +
			"x = decoder_lstm(future_inputs, initial_state=[state_h, state_c])\n" +
			"x = tf.keras.layers.Dense(24, activation='relu')(x)\n" + 
			"x = tf.keras.layers.Dense(12, activation='relu')(x)\n")

		self.modelText = self.modelEdit.toPlainText()

		self.modelBox.setLayout(self.modelLayout)

		# Put layout together
		self.mainLayout.addWidget(self.resultBox)
		self.mainLayout.addWidget(self.optionBox)
		self.mainLayout.addWidget(self.modelBox)
		self.setLayout(self.mainLayout)

#	@property
	def get_modelText(self):
		return self.modelEdit.toPlainText()

	@property
	def get_nnModule(self):
		return self.nnModule

class Prediction(QWidget):
	def __init__(self, parent, tab_dat, tab_nn):
		super(Prediction, self).__init__(parent)

		self.predictionModule = SubModule.PredictionModule(self, tab_dat.get_dataModule, tab_nn.get_nnModule, self.update_plot)
		
		self.dataModule = tab_dat.get_dataModule
		self.nnModule = tab_nn.get_nnModule

		# Vertical main layout
		self.mainLayout = QVBoxLayout(self)

		# Horizontal top layout with options
		self.optionBox = QGroupBox("Options")
		self.optionLayout = QHBoxLayout()

#		self.saveButton = QPushButton("Save Plot")
#		self.saveButton.clicked.connect(self.savePlot)
#		self.optionLayout.addWidget(self.saveButton)
		
		self.nextButton = QPushButton("Next Example")
		self.nextButton.clicked.connect(self.predictionModule.nextExample)
		self.optionLayout.addWidget(self.nextButton)

		self.optionBox.setLayout(self.optionLayout)

		# Plotgraph for predictions
		self.plotBox = QGroupBox("Plot")
		self.plotLayout = QHBoxLayout()

		self.graph = pg.PlotWidget()

		self.plotLayout.addWidget(self.graph)

		self.plotBox.setLayout(self.plotLayout)

		# Put layout together
		self.mainLayout.addWidget(self.optionBox)
		self.mainLayout.addWidget(self.plotBox)
		self.setLayout(self.mainLayout)
		
		self.sample_n = 0
		
		self.sd = []

	def update_plot(self, data):
		self.graph.clear()
		
		if self.nnModule.get_model is None:
			return
		
		(past, future), truth = data
		
		# Scale stuff
		t = np.array(range(SubModule.forecast_len))
		
		(min, max) = self.dataModule.get_scale()
		mean = lambda x: x.mean().numpy().flatten() * (max - min) + min
		sd = lambda x: x.stddev().numpy().flatten() * (max - min) + min
		truth = truth * (max - min) + min
		
		pred = self.nnModule.get_model((past, future))
		
		print("Standard Deviation (Variance):")
		self.sd.append(np.mean(sd(pred)))
		print(np.mean(self.sd))
		
		self.graph.plot(t, mean(pred), label='Prediction', pen=pg.mkPen('r'))
		self.graph.plot(t, truth.numpy().flatten(), label='True value', pen=pg.mkPen('b'))
		
		phigh = pg.PlotCurveItem(t, mean(pred) + 2*sd(pred), pen = pg.mkPen('g'))
		plow = pg.PlotCurveItem(t, mean(pred) - 2*sd(pred), pen = pg.mkPen('g'))
		pfill = pg.FillBetweenItem(phigh, plow, brush = (0,255,0,50))

		self.graph.addItem(pfill)
