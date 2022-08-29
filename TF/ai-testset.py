#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import sys 
import pandas as pd

n = 65536
time = np.arange(0, n)
data = time/100 + np.random.normal(0, 0.01, n)

dataframe = pd.DataFrame({"Time" : time, "Data" : data})
dataframe.to_csv("testset.csv", mode="w", index=False)
