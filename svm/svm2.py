# -*- coding: utf-8 -*-
"""
Created on Sat Sep 30 10:56:36 2023

@author: David Martin
"""

#%%
# See svm1.py for other exploratory analysis on this data set including
# plotting of the spectra and the auto-scaled spectra etc.

#%%

#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#%%
#from sklearn.decomposition import PCA;
from sklearn.preprocessing import StandardScaler;
from sklearn.svm import SVC;

#%%

filename = r"ptspectra.csv";

#%%

df = pd.read_csv(filename, sep=",")

#%%

Xo = df.iloc[:,1:];  # all data rows and columns, Xo = X original

X = StandardScaler().fit_transform(Xo);   # autoscaled X

Y = df['class'];     # extract class names

#%%



