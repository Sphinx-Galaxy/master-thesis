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
csv_file = pd.read_csv(sys.argv[1], index_col=0)

#TIKZ header
print("\\begin{tikzpicture}")

#Plot axis information
print("\t\\begin{axis}[")
print("\t\txlabel={" + str(csv_file.index.name) + "},")

print("\t\theight=6cm,")
print("\t\twidth=12cm,")

if len(sys.argv) > 2:
	print("\t\tylabel={" + str(sys.argv[2]) + "},")
	
print("\t\taxis x line=bottom,")
print("\t\taxis y line=left,")

if len(sys.argv) > 4:
	print("\t\tymin=" + str(sys.argv[3]) + ",")
	print("\t\tymax=" + str(sys.argv[4]) + ",")

print("]")

#Create graphs
for i in range(0, len(csv_file.columns)):
	print("\t\\addplot[only marks, mark size=1.5pt, color=" + colors[i] + ", mark=" + marks[i] + "] plot coordinates {")
	
	for j in range(0, csv_file.shape[0]):
		if pd.isna(csv_file.iat[j, i]) == False:
			if isinstance(csv_file.iat[j, i], numbers.Number):
				print("\t\t(" + str(csv_file.index[j]) + ", " + str(csv_file.iat[j, i]) + ")")
	
	print("\t};")
	print("\t\\addlegendentry{" + str(csv_file.columns[i]) + "}")

#End TIKZ
print("\t\end{axis}")
print("\end{tikzpicture}")
