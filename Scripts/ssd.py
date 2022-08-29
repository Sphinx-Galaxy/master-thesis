#!/usr/bin/env python
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import numpy.fft as fft
import pandas as pd
from pyts.decomposition import SingularSpectrumAnalysis
import sys

### Get arguments ###
if len(sys.argv) < 3:
	print("Please specify a file and column")
	sys.exit()

### Read csv file ###
filename = sys.argv[1]
column = int(sys.argv[2])
csv_file = pd.read_csv(filename)
x = csv_file.values[: , column]
x = x[0::(4*24)]
x = np.sin(3.141*np.arange(200)/80)# + np.arange(-1, 1, 0.01)
x += np.sin(3.141*np.arange(200)/20)
x += np.random.normal(0, 0.1, len(x))


### Embedding Dimension ###
N = len(x)
M = int(N/3)
M = 40

print("Length M: " + str(M))
print(x.shape)

X = []
#for i in range(0, M):
#	X_ = np.zeros(N)
#	X_[M-i:M+1] = x[0:i+1]
#	X.append(X_)

#X.append(x)

#for i in range(0, M):
#	X_ = np.zeros(N)
#	X_[0:N-i-1] = np.roll(x, -i-1)[0:N-i-1]
#	X.append(X_)

for i in range(0, M):
	X.append(x[i:N-M+i+1])	
	
X = np.array(X)
print(X.shape)

ssa = SingularSpectrumAnalysis(window_size=40, groups=3)
X_ssa = ssa.transform(X)

print(X_ssa.shape)

### Plot ###
plt.figure(figsize=(16, 6))

ax1 = plt.subplot(121)
ax1.plot(X[0], 'o-', label='Original')
ax1.legend(loc='best', fontsize=14)

ax2 = plt.subplot(122)
for i in range(3):
    ax2.plot(X_ssa[0, i], 'o--', label='SSA {0}'.format(i + 1))
ax2.legend(loc='best', fontsize=14)

plt.suptitle('Singular Spectrum Analysis', fontsize=20)

plt.tight_layout()
plt.subplots_adjust(top=0.88)
plt.show()
