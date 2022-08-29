#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numbers
import pandas as pd
import sys

#Parameter
colors = ['blue', 'red', 'black', 'orange']
marks = ['*', 'x', 'triangle', 'square']

#Get arguments
if len(sys.argv) < 2:
	print("Not enough arguments!")
	print("1: filename.csv")
	print("2: ylabel")
	print("3: ymin")
	print("4; ymax")
	sys.exit()

#Read CSV file	
csv_file = pd.read_csv(sys.argv[1], sep=';')

#TIKZ header
print("\\begin{tikzpicture}")

#Plot axis information
print("\t\\begin{axis}[")

print("\t\theight=6cm,")
print("\t\twidth=12cm,")

print("\t\txlabel={" + str(csv_file.columns[0]) + "},")

if len(sys.argv) > 2:
	print("\t\tylabel={" + str(sys.argv[2]) + "},")
	
print("\t\taxis x line=bottom,")
print("\t\taxis y line=left,")

if len(sys.argv) > 4:
	print("\t\tymin=" + str(sys.argv[3]) + ",")
	print("\t\tymax=" + str(sys.argv[4]) + ",")

print("]")

#Create graphs
for i in range(1, len(csv_file.columns)):
	print("\t\\addplot[only marks, mark size=1.5pt, color=" + colors[i-1] + ", mark=" + marks[i-1] + "] plot coordinates {")

	last_value = 0
	last_index = 0
	r_ratio = 2
	delta_t = max(csv_file.index) / 100
	
	for j in range(0, len(csv_file.values[: , i])):
			
		if pd.isna(csv_file.iat[j, i]) == False:
			if abs(last_value) > 1e-9:
				if (abs(csv_file.iat[j, i] / last_value) > (1 + r_ratio) or abs(csv_file.iat[j, i] / last_value) < (1 - r_ratio)) or csv_file.index[j] - last_index > delta_t:
					print("\t\t(" + str(csv_file.iat[j, 0]) + ", " + str(csv_file.iat[j, i]) + ")")
					last_value = csv_file.iat[j, i]
					last_index = csv_file.index[j]
			elif abs(csv_file.iat[j, i]) > 1e-9:
				if abs((last_value / csv_file.iat[j, i]) > (1 + r_ratio) or abs(last_value / csv_file.iat[j, i]) < (1 - r_ratio)) or csv_file.index[j] - last_index > delta_t:
					print("\t\t(" + str(csv_file.iat[j, 0]) + ", " + str(csv_file.iat[j, i]) + ")")
					last_value = csv_file.iat[j, i]
					last_index = csv_file.index[j]

#	print(cnt)
	print("\t};")
	print("\t\\addlegendentry{" + str(csv_file.columns[i]) + "}")

#End TIKZ
print("\t\end{axis}")
print("\end{tikzpicture}")
