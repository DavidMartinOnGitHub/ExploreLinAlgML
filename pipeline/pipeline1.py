# -*- coding: utf-8 -*-
"""
Created on Sat Sep 30 10:56:36 2023

@author: David Martin
"""

#%%
# This code experiments with FunctionTransformer


#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#%%
#from sklearn.decomposition import PCA;
from sklearn.preprocessing import StandardScaler, FunctionTransformer;
from sklearn.svm import SVC;
from sklearn.model_selection import train_test_split;
from sklearn.pipeline import Pipeline; # For setting up pipeline

#%%

def CustomFunction(x):
    return np.log10(x);

#%%




#%%

df = pd.read_csv('https://bit.ly/48CQOZS', sep=",", header=0, encoding='utf-8');

#%%

X = df.iloc[:,1:];  # all data rows and columnsl

#%%
x0 = X.iloc[0,:];
x0 = x0 + 5.0; # add 5.0 to all elements to eliminate negatives values


#%%

# calculate directly through the custom function

t1 = CustomFunction(x0);

#%%

# calculate using the built-in function and verify
# the output matches t1.

t2 = np.log10(x0);  # expected to be equal to t1. It is.
print(np.allclose(t1, t2, rtol= 0.0, atol = 1e-6)); # expected True


#%%

# create a transformer pipeline using the custom function
# then verify that it transforms as expected
pipe = Pipeline([
('custom', FunctionTransformer(CustomFunction)),
]);

t3 = pipe.transform(x0);
print(np.allclose(t1, t3, rtol= 0.0, atol = 1e-6)); # expected True





