#!/usr/bin/env python
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys

### Get arguments ###
if len(sys.argv) < 2:
	print("Please specify a file")
	sys.exit()

### Read CSV file ###
filename = sys.argv[1]
csv_file = pd.read_csv(filename)

result = []
gap = []

### Average to XX minutes ###
avg_time = 60*60
i = 0
quarter_cnt = 0
while i < len(csv_file.values[: , 0]):
	
	gap_row = []
	result_row = []
	t_start = csv_file.values[i, 0]
	result_row.append(t_start)
	gap_row.append(False)
	k = 0

	# Get columns #
	for j in range(1, len(csv_file.columns)):
		res = 0.0
		count = 0
		k = 0
		t = 0

		while t < avg_time and i+k < len(csv_file.values[: , 0]):
			if pd.isna(csv_file.values[i+k, j]) == False:
				res += csv_file.values[i+k, j]
				count += 1
			
			t = csv_file.values[i+k, 0] - avg_time*quarter_cnt
			k += 1
			
		result_row.append(res/count if count != 0 else 0)
		gap_row.append(False if count != 0 else True)
		
	result.append(np.array(result_row))
	gap.append(np.array(gap_row))
			
	i += k + 1
	quarter_cnt += 1
	
result = np.array(result)
gap = np.array(gap)

### Fill the gaps ###
for i in range(1, len(gap[0 , :])):
	print("Checking column: " + str(i))	

	# Get rows of column #
	j = 0
	while j < len(gap[: , i]):
		
		# Check for gap #
		if gap[j][i] == True:
			
			start = j-1
			stop = j+1
			
			for k in range(j+1, len(gap[: , i])):
				if gap[k][i] == False:
					stop = k
					break
		
			# Interpolate #
			m = (result[stop][i] - result[start][i]) / (stop - start)

#			print("Gap size: " + str(stop-start) + " at " + str(result[start][0]))
#			print("Interpolate: " + str(result[start][i]) + " -> " + str(result[stop][i]))
			
			for k in range(start+1, stop):
				result[k][i] = m*(k-start) + result[start][i]
#				print(result[k][i])
			
#			print(result[stop][i])

			# Correct counter #
			j = stop+1
		
		else:
			j += 1

### Low pass ###
#lowpass = np.concatenate([np.ones(int(len(result[: , 0])/10)), np.zeros(int(len(result[: , 0])*9/10) + 1)])
lowpass = [1/28, 3/28, 5/28, 10/28, 5/28, 3/28, 1/28]
N = len(lowpass)-1

for i in range(1, len(csv_file.columns)):
#	result[: , i] = np.fft.ifft(np.fft.fft(result[: , i])*lowpass)
	result[: , i] = np.convolve(result[: , i], lowpass)[int(N/2):-int(N/2)]

#### Save to csv ###
dataframe = []
for i in range(1, len(csv_file.columns)):
	dataframe.append(pd.DataFrame({"Time" : result[int(N/2):-int(N/2) , 0], csv_file.columns[i] : result[int(N/2):-int(N/2) , i]}))

csv_frame = dataframe[0]
for i in range(1, len(csv_file.columns)-1):
	csv_frame = pd.merge_ordered(csv_frame, dataframe[i], how='outer')

csv_frame.to_csv(filename.split('.')[0] + "_prune.csv", mode="w", index=False)
