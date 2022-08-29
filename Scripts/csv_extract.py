#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import sys

### Get arguments and init ###
if len(sys.argv) < 3:
	print("Please a file and a column")
	sys.exit()

filename = str(sys.argv[1])
column = int(sys.argv[2])

### Read CSV file ###
filename = sys.argv[1]
csv_file = pd.read_csv(filename)

### Extract column ###
result = csv_file.values[: , column]
cname = csv_file.columns[column]

### Normalize ###
result_mean = np.mean(result)
result_std = np.std(result)

print("Mean:", result_mean)
print("Std:", result_std)

result_max = np.max(result)
result_min = np.min(result)

print("Max:", result_max)
print("Min:", result_min)

result = (result - result_mean) / result_std
#result = (result - result_min) / (result_max - result_min)

if min(result) < 0:
	print("Warning! You are using a dataset with negative values!")

### Save to csv ###
dataframe = pd.DataFrame(result)

csv_name = filename.split('.')[0] + "_" + cname + ".csv"

dataframe.to_csv(csv_name, mode="w", index=False, header=False)
