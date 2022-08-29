#!/usr/bin/env python
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

### Permanent class stuff ###
# Data window handler
class WindowGenerator():
	def __init__(self, input_width, label_width, shift, 
		train_df, val_df, test_df,
		label_columns=None):
		
		# Store the raw data
		self.train_df = train_df
		self.val_df = val_df
		self.test_df = test_df
		
		# Labels
		self.label_columns = label_columns
		
		if label_columns is not None:
			self.label_columns_indices = {name: i for i, name in enumerate(label_columns)}
			
		self.column_indices = {name: i for i, name in enumerate(train_df.columns)}
		
		# Window parameters
		self.input_width = input_width
		self.label_width = label_width
		self.shift = shift
		
		self.total_window_size = input_width + shift
		
		self.input_slice = slice(0, input_width)
		self.input_indices = np.arange(self.total_window_size)[self.input_slice]
		
		self.label_start = self.total_window_size - self.label_width
		self.label_slice = slice(self.label_start, None)
		self.label_indices = np.arange(self.total_window_size)[self.label_slice]
		
	def split_window(self, features):
		inputs = features[:, self.input_slice, :]
		labels = features[:, self.label_slice, :]
		
		if self.label_columns is not None:
			labels = tf.stack([labels[:, :, self.column_indices[name]] for name in self.label_columns], axis=-1)
			
		inputs.set_shape([None, self.input_width, None])
		labels.set_shape([None, self.label_width, None])
	
		return inputs, labels
		
	def make_dataset(self, data):
		data = np.array(data, dtype=np.float32)
		ds = tf.keras.preprocessing.timeseries_dataset_from_array(
			data=data,
			targets=None,
			sequence_length=self.total_window_size,
			sequence_stride=1,
			shuffle=True,
			batch_size=128,)
			
		ds = ds.map(self.split_window)
		
		return ds
	
	def plot(self, plot_col, model=None, max_subplots=4):
		inputs, labels = self.example
		plt.figure(figsize=(12, 8))
		plot_col_index = self.column_indices[plot_col]
		max_n = min(max_subplots, len(inputs))
		for n in range(max_n):
			inputs, labels = self.example
	
			plt.subplot(max_n, 1, n+1)
			plt.ylabel(f'{plot_col} [normed]')
			plt.plot(self.input_indices, inputs[n, :, plot_col_index], label='Inputs', marker='.', zorder=-10)
			
			if self.label_columns:
				label_col_index = self.label_columns_indices.get(plot_col, None)
			else:
				label_col_index = plot_col_index
				
			if label_col_index is None:
				continue
				
			plt.scatter(self.label_indices, labels[n, :, label_col_index], edgecolors='k', label='Labels', c='#2ca02c', s=64)
			
			if model is not None:
				predictions = model(inputs)
				plt.scatter(self.label_indices, predictions[n, :, label_col_index], marker='X', edgecolors='k', label='Predictions', c='#ff7f0e', s=64)
				
			if n == 0:
				plt.legend()
		
					
		plt.xlabel('Time [h]')
		
	@property
	def train(self):
		return self.make_dataset(self.train_df)

	@property
	def val(self):
		return self.make_dataset(self.val_df)

	@property
	def test(self):
		return self.make_dataset(self.test_df)

	@property
	def example(self):
		result = getattr(self, '_example', None)
		
		if result is None:
			result = next(iter(self.train))
			self._example = result
			
		return result

	def __repr__(self):
		return '\n'.join([
			f'Total window size: {self.total_window_size}',
			f'Input indices: {self.input_indices}',
			f'Label indices: {self.label_indices}',
			f'Label column name(s): {self.label_columns}'])
			
def plot_input(name, train, val, test):
	# Plotting
	plt.clf()
	fig = plt.figure(figsize=(20,10))
	plt.title(name)
	
	plt.plot(train)
	plt.plot(range(len(train), len(train)+len(val)), val)
	plt.plot(range(len(train)+len(val), len(train)+len(val)+len(test)), test)

	plt.savefig(name.replace(' ', '_') + ".png")

	# CSV
	train_df = pd.DataFrame({"Time" : range(len(train)), "Train" : train})
	val_df = pd.DataFrame({"Time" : range(len(train), len(train)+len(val)), "Validation" : val})
	test_df = pd.DataFrame({"Time" : range(len(train)+len(val), len(train)+len(val)+len(test)), "Test" : test})

	csvframe = train_df
	csvframe = pd.merge_ordered(csvframe, val_df, how='outer')
	csvframe = pd.merge_ordered(csvframe, test_df, how='outer')
	
	csvframe.to_csv(name + ".csv", mode="w", index=False)
	
	return None
	
def plot_output(name, test, baseline, prediction=None):
	# Plotting
	plt.clf()
	fig = plt.figure(figsize=(20,10))
	plt.title(name)
	
	plt.plot(test)
	plt.plot(range(len(test)-len(baseline), len(test)), baseline)

	if prediction is not None:
		plt.plot(range(len(test)-len(prediction), len(test)), prediction)		

	plt.savefig(name.replace(' ', '_') + ".png")
	
	# CSV
	test_df = pd.DataFrame({"Time" : range(len(test)), "Test" : test})
	base_df = pd.DataFrame({"Time" : range(len(test)-len(baseline), len(test)), "Baseline" : baseline})
	
	if prediction is not None:
		pred_df = pd.DataFrame({"Time" : range(len(test)-len(prediction), len(test)), "Prediction" : prediction})
	
	csvframe = test_df
	csvframe = pd.merge_ordered(csvframe, base_df, how='outer')
	
	if prediction is not None:
		csvframe = pd.merge_ordered(csvframe, pred_df, how='outer')
	
	csvframe.to_csv(name + ".csv", mode="w", index=False)
	
	return None

def plot_learning(name, history):
	loss = history.history['loss']
	val_loss = history.history['val_loss']
	epochs = range(1, len(loss) + 1)

	plt.clf()
	fig = plt.figure(figsize=(20,10))
	
	plt.plot(epochs, loss, 'bo', label='Training loss')
	plt.plot(epochs, val_loss, 'b', label='validation loss')
	plt.title("Training and validation loss")
	plt.legend()

	plt.savefig(name.replace(' ', '_') + "_history.png")
