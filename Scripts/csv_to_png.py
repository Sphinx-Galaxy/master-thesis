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

### Plot and save the figure ###
for i in range(1, len(csv_file.columns)):
	fig = plt.figure(figsize=(20,10))
	
	name = filename.split('.')[0] + "_" + str(csv_file.columns[i])

	plt.title(name)

	plt.plot(csv_file.values[:, i], marker='o')
	plt.savefig(name.replace(' ', '_') + ".png")
