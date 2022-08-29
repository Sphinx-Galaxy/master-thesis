#!/usr/bin/env python
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np

x = np.arange(0, 100);
y_sine = 0.45*np.sin(2 * 3.141 * x / 100) + 0.45

y_bat_1 = np.array(0.45*np.sin(2* 3.141 * np.arange(0, 10) / 40) + 0.45)
y_bat_2 = np.array(0.9 * np.ones(40))
y_bat_3 = np.array(0.9 - np.arange(0, 50) / (50/0.45))

print(y_bat_1)
print(y_bat_1.shape)

print(y_bat_2)
print(y_bat_2.shape)

y_bat = np.concatenate((y_bat_1, y_bat_2, y_bat_3))

plt.plot(y_sine)
plt.plot(y_bat)
plt.show()

for i, val in enumerate(y_sine):
	print("(" + str(i) + ", " + str(val) + ")")
	
for i, val in enumerate(y_bat):
	print("(" + str(i) + ", " + str(val) + ")")
