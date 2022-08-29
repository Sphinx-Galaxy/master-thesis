#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import sys 
import pandas as pd

### Get arguments ###
if len(sys.argv) < 2:
	print("Not enough arguments!")
	sys.exit()

filename = str(sys.argv[1])
timeframe = 672
timepre = 96

### Fetch dataset ###
dataset = pd.read_csv(filename)

length = len(dataset.values)

train = dataset.values[:int(length*0.9)]
valid = dataset.values[int(length*0.9):int(length*0.95)]
test = dataset.values[int(length*0.95):]

print("Train: " + str(train.shape))
print("Valid: " + str(valid.shape))
print("Test: " + str(test.shape))

plt.clf()
name = "LSTM Training Data"
fig = plt.figure(figsize=(20,10))
plt.title(name)
plt.plot(train)
plt.plot(range(int(length*0.9), int(length*0.95)), valid)
plt.plot(range(int(length*0.95), int(length)), test)

plt.savefig(name.replace(' ', '_') + ".png")

### Transform data ###
def transform(arr, seq):
    x, y = [], []
    for i in range(len(arr) - seq - timepre):
        x_i = arr[i : i + seq]
        y_i = arr[i + timepre : i + seq + timepre]
        x.append(x_i)
        y.append(y_i)
    x_arr = np.array(x).reshape(-1, seq)
    y_arr = np.array(y).reshape(-1, seq)
    
    return x_arr, y_arr

x_train, y_train = transform(train, timeframe)
x_valid, y_valid = transform(valid, timeframe)
x_test, y_test = transform(test, timeframe)

### Build a model ###
# LSTM
inputs = tf.keras.Input(shape=(timeframe, 1))

x = tf.keras.layers.LSTM(units=timepre)(inputs)
x = tf.keras.layers.Dropout(0.2)(x)
outputs = tf.keras.layers.Dense(1, activation='linear')(x)

#lstm = tf.keras.Model(inputs=inputs, outputs=outputs)

#lstm = tf.keras.Sequential([
#	tf.keras.Input(shape=(timeframe, 1)),
#	tf.keras.layers.LSTM(units=timepre),
#	tf.keras.layers.Dropout(0.2)
#	tf.keras.layers.Dense(1, activation='linear')])

#lstm = tf.keras.Sequential([
#	tf.keras.layers.Conv1D(filters=32, kernel_size=(timepre,), activation='relu'),
#	tf.keras.layers.Dense(units=32, activation='relu'),
#	tf.keras.layers.Dense(units=1),
#])

#lstm.compile(optimizer="adam", loss="mean_squared_error")

#lstm.fit(x_train, y_train, epochs=3, batch_size=512, shuffle=True, validation_data=(x_valid, y_valid))
#lstm.fit(x_train, epochs=3, batch_size=512, shuffle=True)

### Save the model ###
#converter = tf.lite.TFLiteConverter.from_keras_model(lstm)
#tflite_model = converter.convert()

# Save the model.
#lstm.save('lstm_model')
  
### open the model ###
lstm = tf.keras.models.load_model('lstm_model')
	
lstm.summary();

### Test the model ###
pre = lstm.predict(x_test)

plt.clf()
name = "LSTM Prediction"
fig = plt.figure(figsize=(20,10))
plt.title(name)
plt.plot(train)
plt.plot(range(int(length*0.9), int(length*0.95)), valid)
plt.plot(range(int(length*0.95), int(length)), test)
plt.plot(range(timeframe+int(length*0.95), int(length)-timepre), pre, marker='o')

plt.savefig(name.replace(' ', '_') + "_1.png")

plt.clf()
name = "LSTM Prediction"
fig = plt.figure(figsize=(20,10))
plt.title(name)
plt.plot(test)
plt.plot(pre, marker='o')

plt.savefig(name.replace(' ', '_') + "_2.png")
