#!/usr/bin/env python
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys

### Get arguments ###
if len(sys.argv) < 4:
	print("Please specify a file, a column and your value")
	sys.exit()

### Read CSV file ###
filename = sys.argv[1]
column = int(sys.argv[2])
value = float(sys.argv[3])
epsilon = 1e-1

csv_file = pd.read_csv(filename)
counter = 0

for i in range(0, len(csv_file.values[: , column])):
	
	if value + epsilon > csv_file.values[i, column] and value - epsilon < csv_file.values[i, column]:
		print(csv_file.loc[i]) 
		counter += 1
		
print("Found your value: " + str(value) + " exactly " + str(counter) + " times")
