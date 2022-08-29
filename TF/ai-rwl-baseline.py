#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys 
import pandas as pd

filename = str(sys.argv[1])
df = pd.read_csv(filename, index_col='Time')

dfs = df[['rwl_est_frict_torque']]
dfs['rwl_est_frict_torque'] = dfs['rwl_est_frict_torque'] / dfs['rwl_est_frict_torque'].max()

n = len(dfs)

mae = 0
for i in range(int(n*0.9), n-24):
	mae += abs(dfs.values[i, 0] - dfs.values[i+24, 0])
	
scaling_factor = df.rwl_est_frict_torque.max()
print("MAE: " + str((mae / (n - 24 - int(n*0.9)))))
print("MAE: " + str(scaling_factor * (mae / (n - 24 - int(n*0.9)))))
