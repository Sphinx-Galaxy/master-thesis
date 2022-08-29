#!/usr/bin/env python
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np

x = np.arange(0, 200);
y = np.sin(2 * 3.141 * x / 100)
y_noise = y + np.random.normal(0, 0.4, len(x))

plt.plot(y_noise)
#plt.show()

y_fft = np.fft.fft(y_noise)

plt.plot(np.log10(y_fft[0:int(len(x)/2)]))
#plt.show()

y_lowpass_1 = np.fft.ifft(y_fft[0:int(len(x)/4)])
y_lowpass_2 = np.fft.ifft(y_fft[0:int(len(x)/10)])

y_mean = []
y_mean.append(y_noise[0])

y_last = []
y_last.append(y_noise[0])

for i in range(1, len(x)):
	y_last.append(y_noise[i])
	
	if len(y_last) > 10:
		del y_last[0]

	y_mean.append(np.sum(y_last) / len(y_last))

print(y_mean)
plt.plot(y_noise)
#plt.plot(np.arange(0, 200, 4), y_lowpass_1/2.0)
plt.plot(np.arange(0, 200, 10), y_lowpass_2/5.0)
plt.plot(y_mean)
plt.show()

for i in range(0, len(x)):
	print("(" + str(i) + ", " + str(y_noise[i]) + ")")
		
for i in range(0, int(len(x)/10)):
#	print("(" + str(i*10) + ", " + str(2*np.real(y_lowpass_2[i]/5.0) * np.imag(y_lowpass_2[i]/5.0)) + ")")
	print("(" + str(i*10) + ", " + str(np.real(y_lowpass_2[i]/5.0)) + ")")
	
for i in range(0, len(x)):
	print("(" + str(i) + ", " + str(y_mean[i]) + ")")

