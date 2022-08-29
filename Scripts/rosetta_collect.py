#!/usr/bin/env python
# -*- coding: utf-8 -*-

import datetime
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import sys

### Conversion of the given date ###
def date_to_time(date):
	year = int(date.split("-")[0])
	month = int(date.split("-")[1])
	day = int(date.split("-")[2].split("T")[0])

	date = date.split("T")[1]
	hour = int(date.split(":")[0])
	minute = int(date.split(":")[1])
	second = int(date.split(":")[2].split(".")[0])
	
	#print("Year: " + str(year) + " Month: " + str(month) + " Day: " + str(day))
	#print("Hour: " + str(hour) + " Minute: " + str(minute) + " Second: " + str(second))
	
	return datetime.datetime(year, month, day, hour, minute, second)

### Get arguments and init ###
if len(sys.argv) < 5:
	print("Please specify folder, name, year and number of possible parameters")
	sys.exit()

location = str(sys.argv[1])
name = str(sys.argv[2])
year = str(sys.argv[3])
parameter = int(sys.argv[4])

### Setup arrays ###
value_name = []
value = []
time = []
timebase = []
narray = []

for i in range(0, parameter):
	value.append([])
	time.append([])
	timebase.append(0)
	narray.append(False)

### Collect the data ###
counter = 0
for folder in sorted(os.listdir(location)):

	value_name.append(folder)

	folder = location + "/" + folder
	print("Checking folder: " + folder)
	
	for subfolder in sorted(os.listdir(folder)):

		subfolder = folder + "/" + subfolder
		print("Checking subfolder. " + subfolder)

		if year in subfolder:
			break
			
		for filename in sorted(os.listdir(subfolder)):
		
			if ".tab" not in filename or name not in filename:
				continue
				
			filename = subfolder + "/" + filename
			print("Checking file: " + str(filename))
			
			# Every file in the folder is another wheel or thingy
			with open(filename) as f:
				content = f.readlines()

				for line in content:
					value[counter].append(float(line.split(',')[1]))
					t = date_to_time(line.split(',')[0])
					
					if timebase[counter] == 0:
						timebase[counter] = t
					
					time[counter].append((t - timebase[counter]).total_seconds())
	
	if timebase[counter] == 0:
		narray[counter] = True
	
	counter += 1

### Correct arrays ###
k = 0
for i in range(0, len(narray)):
	if narray[i] == True:
		del value[k]
		del timebase[k]
		del time[k]
		del value_name[k]
		k -= 1
	k += 1

### Correct time offset ###
timebase = np.array(timebase)

t_min = min(timebase)

for i in range(0, len(timebase)):
	time[i] = np.array(time[i])
	time[i] += (timebase[i] - t_min).total_seconds()
	
### Merge ###
dataframe = []

for i in range(0, len(time)):
	dataframe.append(pd.DataFrame({"Time" : time[i], value_name[i] : value[i]}))


csvframe = dataframe[0]
	
for i in range(1, len(time)):
	csvframe = pd.merge_ordered(csvframe, dataframe[i], how='outer')

### Save to csv ###
csvframe.to_csv(name + "_" + year + ".csv", mode="w", index=False)
