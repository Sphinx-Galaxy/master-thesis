#!/usr/bin/env python
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys

def get_missing_elements(x):
	res = 0
	
	for i in range(0, len(x)):
		if pd.isna(x[i]) == False:
			res += 1
	
	return res
	
def get_time_bin(x, y):
	# 0 to 120 secs (120 == else)
	time_bin = np.zeros(100)
	last_t = 0
	
	for i in range(1, len(x)):
		if pd.isna(y[i]) == True:
			continue
			
		time = int(x[i] - last_t)
		last_t = int(x[i])
		
		if time < 500:
			time_bin[int(time/5)] += 1
		else:
			time_bin[99] += 1
	
	return time_bin
	
def get_time(x, y):
	min_t = 60*60*24
	max_t = 0
	avg_t = x[1] - x[0]
	cnt_t = 1
	last_t = 0
	
	for i in range(2, len(x)):
		if pd.isna(y[i]) == True:
			continue
	
		t = x[i] - last_t
		last_t = x[i]
		
		if t < min_t:
			min_t = t
			
		if t > max_t:
			max_t = t
			
		avg_t += t
		cnt_t += 1
		
	return str(min_t) + ";" + str(max_t) + ";"  + str(avg_t/cnt_t)

### Get arguments ###
if len(sys.argv) < 2:
	print("Please specify a file")
	sys.exit()

### Read CSV file ###
filename = sys.argv[1]
csv_file = pd.read_csv(filename)

### Show total number of datapoints ###
output = str(len(csv_file.values[: ,0])) + ";;"

### Show number of missing points ###
for i in range(1, len(csv_file.columns)):
	output += str(get_missing_elements(csv_file.values[: , i])) + ";"

output += ";"

### Show time min max ###
for i in range(1, len(csv_file.columns)):
	output += str(get_time(csv_file.values[: , 0], csv_file.values[: , i])) + ";;"

output += ";"

### Bin for average timing ###
time_bin = get_time_bin(csv_file.values[: , 0], csv_file.values[: , 1])
for i in range(2, len(csv_file.columns)):
	time_bin += get_time_bin(csv_file.values[: , 0], csv_file.values[: , i])

for i in range(0, len(time_bin)):
	output += str(time_bin[i]) + ";"
	
### Print information
name = filename.split('.')[0]

print(name + ";" + filename + ";" + output)
