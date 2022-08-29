# ##################################
# Copyright (c) 2021, Mattis Jaksch
#
# This Python file uses the following encoding: utf-8
# ##################################

from PySide6.QtCore import QObject, Slot, Signal
from PySide6.QtWidgets import QAbstractButton, QFileDialog

import tensorflow as tf
import tensorflow_probability as tfp

import pandas as pd
import numpy as np
import math

window_len = 24
forecast_len = 24
batch_size = 32

class DataModule(QObject):
	def __init__(self, parent, filename, cb_update_plot):
		super(DataModule, self).__init__(parent)

		self.filename = filename
		self.update_plot = cb_update_plot

		self.df = pd.read_csv(filename, sep=',')
		self.fft_data = []

		self.features = self.df.columns.tolist()
		self.deterministic_features = []
		self.n_deterministic_features = None
		self.aleatoric_features = []
		self.none_features = []
		self.prediction_feature = None
		
		self.one_hot_exclusion = self.df.columns.tolist()

		self.data = self.df.copy()
		
		for column in (self.features):
			min = np.min(self.df[column])
			max = np.max(self.df[column])
			self.data[column] = (self.data[column] - min) / (max - min)
		
		self.forecast_len = forecast_len
		self.window_len = window_len
		
	@property
	def get_features(self):
		return self.features

	@property
	def get_n_features(self):
		return len(self.aleatoric_features) + self.n_deterministic_features + 1 # Prediction accounts for one

	@property
	def get_deterministic_features(self):
		return self.deterministic_features

	@property
	def get_n_deterministic_features(self):
		return self.n_deterministic_features
		
	@property
	def get_aleatoric_features(self):
		return self.aleatoric_features
		
	@property
	def get_prediction_feature(self):
		return self.prediction_feature
		
	@property
	def get_entries(self):
		return len(self.df)

	@property
	def get_data(self):
		return self.data

	@property
	def get_fft_data(self):
		return self.fft_data

	@property
	def get_train(self):
		df = self.prepare_data(self.df)
		return self.create_dataset(df.iloc[:int(len(self.df)*0.8)], self.window_len, self.forecast_len, batch_size)
		
	@property
	def get_valid(self):
		df = self.prepare_data(self.df)
		return self.create_dataset(df.iloc[int(len(self.df)*0.8):int(len(self.df)*0.9)], self.window_len, self.forecast_len, batch_size)

	@property
	def get_test(self):
		df = self.prepare_data(self.df)
		return self.create_dataset(df.iloc[int(len(self.df)*0.9):], self.window_len, self.forecast_len, batch_size=1)
		
	def get_scale(self):
		max = np.max(self.df[self.prediction_feature])
		min = np.min(self.df[self.prediction_feature])
		
		return (min, max)

	def set_past_points(self, points):
		self.window_len = points
		
	def set_future_points(self, points):
		self.forecast_len = points

	@Slot(QAbstractButton)
	def set_features(self, button):
		button = button.group().id(button)
		feature_id = int(button / 4) # 4 States: Deterministic, Aleatoric, Prediction, None
		state_id = button % 4
#		print("Feature ID: " + str(feature_id))
#		print("State ID: " + str(state_id))

		# Remove feature from list
		if self.df.columns[feature_id] in self.features:
			self.features.remove(self.df.columns[feature_id])
		
		if self.df.columns[feature_id] in self.deterministic_features:
			self.deterministic_features.remove(self.df.columns[feature_id])
		if self.df.columns[feature_id] in self.aleatoric_features:
			self.aleatoric_features.remove(self.df.columns[feature_id])
		if self.df.columns[feature_id] == self.prediction_feature:
			self.prediction_feature = None
		if self.df.columns[feature_id] in self.none_features:
			self.none_features.remove(self.df.columns[feature_id])
		
		# Add feature to list
		if state_id == 0:
			self.deterministic_features.append(self.df.columns[feature_id])
			self.features.append(self.df.columns[feature_id])
		elif state_id == 1:
			self.aleatoric_features.append(self.df.columns[feature_id])
			self.features.append(self.df.columns[feature_id])
		elif state_id == 2:
			self.prediction_feature = self.df.columns[feature_id]
			self.features.append(self.df.columns[feature_id])
		elif state_id == 3:
			self.none_features.append(self.df.columns[feature_id])
			
		self.update_plot()
		
	@Slot(QAbstractButton)
	def set_prediction_feature(self, button):
		self.prediction_feature = self.df.columns[button.group().id(button)]

	@Slot(int)
	def set_smooth(self, value):
		if value != 0:
			for column in self.df.columns:
				self.data[column] = self.smooth_column(self.df[column], value)

		else:
			self.data = self.df
						
		self.update_plot()

	@Slot(bool)
	def set_fft(self, checked):
		fft = []
		for column in self.df.columns:
			fft.append(np.array(np.log10(np.abs(np.fft.fft(self.df[column])))))
		
		fft = np.transpose(np.array(fft))
		
		self.fft_data = pd.DataFrame(data=fft, columns=self.df.columns)
		
		self.update_plot()
		
	@Slot(QAbstractButton)
	def set_one_hot(self, button):
		feature_id = button.group().id(button)
#		print("Set one hot index: " + str(feature_id))
		
		if button.isChecked() == True:
			self.one_hot_exclusion.remove(self.df.columns[feature_id])
		else:
			self.one_hot_exclusion.append(self.df.columns[feature_id])
		
	def smooth_column(self, data, value):
		window = np.ones(value,'d')
		curve = np.r_[data[value-1:0:-1], data, data[-2:-value-1:-1]]
		x = np.convolve(window/window.sum(), curve, mode='valid')

		return x[int(value/2)-1:-int(value/2)]

	def one_hot_encode(self, df):
		df = df.copy()
		
		int_cols = [col for col in df.select_dtypes(include=['int']) if col not in self.one_hot_exclusion]

		df.loc[:, int_cols] = df.loc[:, int_cols].astype('str')
		
		df_encoded = pd.get_dummies(df)
		
#		print(df_encoded)
		
		return df_encoded

	def prepare_data(self, df):
		# Scale other features	
		for column in (self.features):
			if column in self.one_hot_exclusion:
				max = np.max(self.df[column])
				min = np.min(self.df[column])
				df[column] = (df[column] - min) / (max - min)

		# Re-order features
		features = self.aleatoric_features.copy()
		
		for f in self.deterministic_features:
			features.append(f)
		
		features.insert(0, self.prediction_feature)
		df = df.reindex(columns=features)

		# One hot encoding
		df = self.one_hot_encode(df)
				
#		print(df.columns)
		print(df)

		return df

	def create_dataset(self, df, window_size, forecast_size, batch_size):
		shuffle_buffer_size = len(df)
		
		total_size = window_size + forecast_size
	
		# Put features into dataset
		data = tf.data.Dataset.from_tensor_slices(df.values)
		
		data = data.window(total_size, shift=1, drop_remainder=True)
		data = data.flat_map(lambda k: k.batch(total_size))
		
		data = data.shuffle(shuffle_buffer_size, seed=42)
		
		# Prediction feature accounts for one
		self.n_deterministic_features = len(df.columns) - len(self.aleatoric_features) - 1
		
		print(df.columns)
		print(self.aleatoric_features)
		
		# Past, Future, Label
		data = data.map(lambda k: ((k[:-forecast_size], k[-forecast_size:, -self.n_deterministic_features:]),
			k[-forecast_size:, 0]))
			
		print(data)
			
		return data.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)

class NNModule(QObject):
	baseline_changed = Signal(str)
	eval_changed = Signal(str)
	
	def __init__(self, parent, modelText, dataModule):
		super(NNModule, self).__init__(parent)
		
		self.dataModule = dataModule
		self.modelText = modelText
					
		self.epoch = 5
		self.lr = 0.02
		self.window_len = window_len
		self.forecast_len = forecast_len
		
		self.model = None
		
	@Slot()
	def compile(self):
		# Generate dataset to publish input size
		train = self.dataModule.get_train
	
		# Add prefix with (past, future) inputs 
		model_prefix = 	("past_inputs = tf.keras.Input(shape=(self.window_len, self.dataModule.get_n_features), name=\'past_inputs\')\n" +
				"future_inputs = tf.keras.Input(shape=(self.forecast_len, self.dataModule.get_n_deterministic_features), name=\'future_inputs\')\n")
		
		# Add suffix with output and probability distribution
		model_suffix =	("x = tf.keras.layers.Dense(2, activation=\'relu\')(x)\n"
				"tfd = tfp.distributions\n" +
				"output = tfp.layers.DistributionLambda(lambda t: tfp.distributions.Normal(loc=t[...,0], scale=0.01*tf.math.softplus(t[...,1])), name=\'normal_dist\')(x)\n" +
				"self.model = tf.keras.models.Model(inputs=[past_inputs, future_inputs], outputs=output)\n" +
				"self.model.compile(loss=self.neglogik, optimizer=tf.keras.optimizers.Adam(self.lr), metrics=[\"mae\"])\n")

		# Execute model steps
		exec(model_prefix)				
		exec(self.modelText())
		exec(model_suffix)
		
		exec("print(self.model.summary())")
		
	@Slot()
	def train(self):
		fit_model = "self.history = self.model.fit(self.dataModule.get_train, epochs=self.epoch, validation_data=self.dataModule.get_valid)\n"
		
		exec(fit_model)

		# Debug info	
		exec("print(self.model.summary())")
		
	@Slot()
	def eval(self):
		print("Evaluate model")
		
		eval_model = "self.eval_result = self.model.evaluate(self.dataModule.get_test)"
		
		exec(eval_model)
		
		self.baseline_eval()
		self.eval_changed.emit(str(self.eval_result[1]))
		
	@property
	def get_model(self):
		if self.model is not None:
			return self.model
		
	@Slot()
	def load(self):
		# Open file select		
		(path, filetype) = QFileDialog.getOpenFileName()
		
		self.model = tf.keras.models.load_model(path, custom_objects={"neglogik": self.neglogik})
	
	@Slot()
	def save(self):
		if self.model is not None:
			(path, filetype) = QFileDialog.getSaveFileName()
			
			self.model.save(path)	
	@Slot(int)
	def set_epoch(self, epoch):
		self.epoch = epoch
		
	@Slot(int)
	def set_lr(self, lr):
		self.lr = lr / 1000.0
		
	@Slot(int)
	def set_past_points(self, points):
		self.window_len = points
		
		self.dataModule.set_past_points(points)
		
	@Slot(int)
	def set_future_points(self, points):
		self.forecast_len = points
		
		self.dataModule.set_past_points(points)
		
	def neglogik(self, y, p_y):
		return -p_y.log_prob(y)
	
	def baseline_eval(self):
		label = self.dataModule.get_data[self.dataModule.get_prediction_feature]
		
		
		mae = 0
		for i in range(0, len(label)-forecast_len):
			mae += abs(label[i] - label[i+24])
		
		mae /= (len(label) - forecast_len)
		
		self.baseline_changed.emit(str(mae))

class PredictionModule(QObject):
	def __init__(self, parent, dataModule, nnModule, cb_update_plot):
		super(PredictionModule, self).__init__(parent)

		self.dataModule = dataModule
		self.nnModule = nnModule
		
		self.update_plot = cb_update_plot
		
		self.sample_n = 0
	@Slot()
	def nextExample(self):
		for i, data in enumerate(self.dataModule.get_test.take(100)):
			if i == self.sample_n:
				self.update_plot(data)

		self.sample_n += 1
