#!/usr/bin/env python
# -*- coding: utf-8 -*-

import AIHelp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
import tensorflow as tf

# Load data
df = pd.read_csv('rwa_prune.csv')

#df.pop('Time')

# Split data
n = len(df)
train_df = df[0:int(n*0.8)]
val_df = df[int(n*0.8):int(n*0.95)]
test_df = df[int(n*0.95):]

num_features = df.shape[1]

# Normalize data
train_mean = train_df.mean()
train_std = train_df.std()

train_df = (train_df - train_mean) / train_std
val_df = (val_df - train_mean) / train_std
test_df = (test_df - train_mean) / train_std

train_lin = train_df.values[: , 1]
val_lin = val_df.values[: , 1]
test_lin = test_df.values[: , 1]

AIHelp.plot_input("RWL Raw", train_lin, val_lin, test_lin)

SHIFT_STEPS = 48
OUT_STEPS = 72
IN_STEPS = 48
start = OUT_STEPS + IN_STEPS - SHIFT_STEPS

### Create baseline model ###
linear_fit = np.polyfit(range(len(train_lin)), train_lin, 1)
y_fit_train =np.array(range(0, len(train_lin)) * linear_fit[0]) + linear_fit[1]
y_fit_test = np.array(range(0, SHIFT_STEPS) * linear_fit[0]) + test_lin[start-1]

y_test = test_lin
mae = 0
for i in range(start, len(test_lin) - SHIFT_STEPS - 1):
	y_fit = np.array(range(0, SHIFT_STEPS) * linear_fit[0]) + y_test[i-1]

	mae += np.mean(np.abs(y_fit - y_test[i : i + SHIFT_STEPS]))

mae /= len(test_df) - SHIFT_STEPS - 1 - start

test_values = test_lin[0:(IN_STEPS+OUT_STEPS)]
AIHelp.plot_output("RWL Baseline", test=test_values, baseline=y_fit_test)
print("Baseline slope: " + str(linear_fit[0]) + " Baseline offset: " + str(linear_fit[1]))
print("Baseline train_mae:", np.mean(np.abs(train_lin - y_fit_train)))
print("Baseline predict_mae:", mae)

### Create AI model ###

### Tensorflow feedback model ###
class FeedBack(tf.keras.Model):
	def __init__(self, units, out_steps):
		super().__init__()
		self.out_steps = out_steps
		self.units = units
		self.lstm_cell = tf.keras.layers.LSTMCell(units)
		# Also wrap the LSTMCell in an RNN to simplify the `warmup` method.
		self.lstm_rnn = tf.keras.layers.RNN(self.lstm_cell, return_state=True)
		self.dense = tf.keras.layers.Dense(1, activation='linear')
    
	def warmup(self, inputs):
		# inputs.shape => (batch, time, features)
		# x.shape => (batch, lstm_units)
		x, *state = self.lstm_rnn(inputs)

		# predictions.shape => (batch, features)
		prediction = self.dense(x)
		return prediction, state
  
	def call(self, inputs, training=None):
		# Use a TensorArray to capture dynamically unrolled outputs.
		predictions = []
		# Initialize the LSTM state.
		prediction, state = self.warmup(inputs)

		# Insert the first prediction.
		predictions.append(prediction)

		# Run the rest of the prediction steps.
		for n in range(1, self.out_steps):
			# Use the last prediction as input.
			x = prediction
			# Execute one lstm step.
			x, state = self.lstm_cell(x, states=state,
						              training=training)
			# Convert the lstm output to a prediction.
			prediction = self.dense(x)
			# Add the prediction to the output.
			predictions.append(prediction)

		# predictions.shape => (time, batch, features)
		predictions = tf.stack(predictions)
		# predictions.shape => (batch, time, features)
		predictions = tf.transpose(predictions, [1, 0, 2])
		return predictions

# Tensor flow model handler
def compile_and_fit(model, window, patience=3):
	early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience, mode='min')
	
#	model.compile(loss=tf.losses.MeanAbsoluteError(), optimizer=tf.optimizers.SGD(learning_rate=3e-2), metrics=[tf.metrics.MeanAbsoluteError()])
#, decay=1e-10
#, clipvalue=1.0
#	print(feedback_model.summary())
	model.compile(loss=tf.losses.MeanSquaredError(), optimizer=tf.optimizers.Adam(), metrics=[tf.metrics.MeanAbsoluteError()])

#	print(model.summary())

	history = model.fit(window.train, epochs=30, validation_data=window.val, callbacks=[early_stopping])

	print(model.summary())
	
	return history

#model = FeedBack(units=256, out_steps=OUT_STEPS)

CONV_WIDTH = 24
### Single Shot model ###
model = tf.keras.models.Sequential()
#model.add(tf.keras.layers.Reshape([OUT_STEPS, num_features]))
model.add(tf.keras.layers.InputLayer(input_shape=(IN_STEPS, num_features)))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(int(sys.argv[1]), activation='linear'))
model.add(tf.keras.layers.Dense(OUT_STEPS, activation='linear'))
model.add(tf.keras.layers.Reshape([OUT_STEPS, 1]))

### Stackoverflow test ###

#model = tf.keras.Model(inputs=inputs, outputs=outputs)

### Evaluation
# Window definition
multi_window = AIHelp.WindowGenerator(input_width=IN_STEPS, label_width=OUT_STEPS, shift=SHIFT_STEPS, train_df=train_df, val_df=val_df, test_df=test_df, label_columns=['rwl_est_frict_torque'])
print(multi_window)
#print(multi_window.example[0].shape)

history = compile_and_fit(model, multi_window)
loss = model.evaluate(multi_window.test)
	
i = 0
for batch in multi_window.test:
	inputs, labels = batch

	prediction = model.predict(inputs)
	
	if i == 0:
		print("Inputs:", inputs.shape)
		print("Labels:", labels.shape)
		print("Prediction:", prediction.shape)


	print("MAE (" + str(i) + "):" + str(np.mean(np.abs(prediction[0, :, 0] - labels[0, :, 0]))))
	
	test_values = np.concatenate([np.array(inputs[0, :, 1]).flatten(), np.array(labels[0, :, 0]).flatten()])
	y_fit_test = np.array(range(0, OUT_STEPS) * linear_fit[0]) + inputs[0, -1, 1]
	AIHelp.plot_output(name="RWL Prediction " + str(i), test=test_values, baseline=y_fit_test, prediction=prediction[0, :, 0])
	
	i += 1
	
	if i == 6:
		break
	
AIHelp.plot_learning("RWL Single Shot", history)
