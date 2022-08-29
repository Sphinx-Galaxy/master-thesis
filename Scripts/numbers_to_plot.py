#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys

### Get arguments ###
if len(sys.argv) < 2:
	print("Please specify a file")
	sys.exit()

filename = sys.argv[1]

### x axis ###
year = 2004

print("symbolic x coords={")

with open(filename) as f:
	content = f.readlines()

	output = ""

	for line in content:
		output += "$" + str(year) + "$, "
		year += 1	
		
print(output[:-2] + "}")

### coords ###
year = 2004

with open(filename) as f:
	content = f.readlines()

	for line in content:
		print("($" + str(year) + "$, " + line.strip('\n') + ")")

		year += 1
