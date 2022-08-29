#!/usr/bin/env python
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import sys

### Get arguments ###
if len(sys.argv) < 2:
	print("Please specify a file (+period)")
	sys.exit()

### Read csv file ###
filename = sys.argv[1]
period = (24*90) if len(sys.argv) < 3 else int(sys.argv[2])
csv_file = pd.read_csv(filename)

### Create individual parameter

### Plot and save the figure ###
dataframe = []

for i in range(1, len(csv_file.columns)):
	plt.rcParams['figure.figsize'] = [20, 10]

	res = sm.tsa.seasonal_decompose(csv_file.values[: , i], period=period)

	observed = res.observed[period:-period]
	seasonal = res.seasonal[period:-period]
	trend = res.trend[period:-period]
	residual = res.resid[period:-period]

	time = np.arange(0, len(observed)) * 3600
	dataframe.append(pd.DataFrame({"Time" : time,
		csv_file.columns[i] + "_observed" : observed,
		csv_file.columns[i] + "_seasonal" : seasonal,
		csv_file.columns[i] + "_trend" : trend,
		csv_file.columns[i] + "_residual" : residual}))

	resplot = res.plot()

	name = filename.split('.')[0] + "_" + str(csv_file.columns[i]) + "_" + str(period)
	plt.title(name)
	plt.savefig(name + ".png")
	plt.clf()

### Merge ###
csvframe = dataframe[0]
	
for i in range(1, len(csv_file.columns)-1):
	csvframe = pd.merge_ordered(csvframe, dataframe[i], how='outer')

### Save to csv ###
name = filename.split('.')[0] + "_X"
csvframe.to_csv(name + ".csv", mode="w", index=False)
