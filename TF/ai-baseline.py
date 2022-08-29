#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import sys 
import pandas as pd
from contextlib import redirect_stdout

### Get arguments ###
if len(sys.argv) < 2:
	print("Not enough arguments!")
	sys.exit()

filename = str(sys.argv[1])
timeframe = 168
timepre = 24

### Fetch dataset ###
dataset = pd.read_csv(filename)

length = len(dataset.values)

y_train = dataset.values[:int(length*0.8)]
y_valid = dataset.values[int(length*0.8):int(length*0.95)]
y_test = dataset.values[int(length*0.95):]

x_train = range(0, len(y_train))
x_valid = range(len(y_train), len(y_train) + len(y_valid))
x_test = range(len(y_train) + len(y_valid), len(y_train) + len(y_valid) + len(y_test))

#print("Train: " + str(y_train.shape))
#print("Valid: " + str(y_valid.shape))
#print("Test: " + str(y_test.shape))

linear_fit = np.polyfit(x_train, y_train.flatten(), 1)

print(linear_fit)

### Show training ###
y_fit = x_train * linear_fit[0] + linear_fit[1]

plt.plot(x_train, y_fit)
plt.plot(x_train, y_train)
#plt.show()

#print("Time; test; fit")
#for i in range(0, len(x_test)):
#	print(str(x_train[i]*60*60) + ";" + str(y_train.flatten()[i]) + ";" + str(y_fit[i]))

y_loss = np.mean(np.abs(y_fit-y_test))
print(y_loss)

### Show test ###
y_fit = (np.array(x_test[timeframe:timeframe + (24*7)]) - x_test[timeframe]) * linear_fit[0] + y_test[timeframe]

plt.plot(x_test[timeframe:timeframe + (24*7)], y_fit)
plt.plot(x_test[timeframe:], y_test[timeframe:])
plt.show()

y_loss = np.mean(np.abs(y_fit-y_test[0:(24*7)]))
print(y_loss)
