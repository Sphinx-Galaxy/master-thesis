#!/usr/bin/env python
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
from numpy.fft import fft
import pandas as pd
import sys

### Get arguments ###
if len(sys.argv) < 2:
	print("Please specify a file")
	sys.exit()

### Read csv file ###
filename = sys.argv[1]
csv_file = pd.read_csv(filename)
ws = 3*4*24*30 # 30 day window

dataframe = []

### Iterate over all columns ###
for i in range(1, len(csv_file.columns)):

	### Windowed FFT ###
	wn = 0
	X = np.zeros(ws)
	y = csv_file.values[: , i]
	y /= np.linalg.norm(y)
	y -= np.average(y)
	
	while (ws + 0.5*ws*wn) < len(csv_file.values[: , i]):
		x = y[int(wn*ws*0.5):int(ws + 0.5*ws*wn)]
		
		X += np.abs(fft(x)[0:int(len(x/2))])
		wn += 1
		
	freq = np.arange(0, 1 / (15*60*2), 1 / (ws*15*60))
	
	### Plot and save figure ###
	fig = plt.figure(figsize=(20,10))
	
	name = filename.split('.')[0] + "_" + str(csv_file.columns[i]) + "_FFT"

	plt.title(name)
	plt.plot(freq, X, marker='o')
	plt.xscale('log')
	plt.yscale('log')
#	plt.show()
	plt.savefig(name.replace(' ', '_') + ".png")
	
	### Put into dataframe ###
	dataframe.append(pd.DataFrame({"Freq" : freq, csv_file.columns[i] : X}))
	
### Merge ###
csvframe = dataframe[0]
	
for i in range(1, len(csv_file.columns)-1):
	csvframe = pd.merge_ordered(csvframe, dataframe[i], how='outer')

### Save to csv ###
name = filename.split('.')[0] + "_FFT"
csvframe.to_csv(name + ".csv", mode="w", index=False)
