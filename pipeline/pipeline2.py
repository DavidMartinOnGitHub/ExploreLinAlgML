# -*- coding: utf-8 -*-
"""
Created on Sun Oct 1 09:11:36 2023

@author: David Martin
"""

#%%
# This code expands upon pipeline1.py by creating a preprocessing
# step with parameters.


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

def CustomFunction(x, offset):
    xo = x + offset;
    return np.log10(xo);


#%%

df = pd.read_csv('https://bit.ly/48CQOZS', sep=",", header=0, encoding='utf-8');

#%%

X = df.iloc[:,1:];  # all data rows and columnsl

#%%
x0 = X.iloc[0,:];

#%%
t1 = CustomFunction(x0, 5.0);


#%%

# create a transformer pipeline using the custom function
# then verify that it transforms as expected
pipe = Pipeline([
('custom', FunctionTransformer(CustomFunction,kw_args={"offset" : 5.0})),
]);

#%%

# use the pipeline with kw_args parameters to transform the data
t3 = pipe.transform(x0);

#%%
# verify that the two calculation methods produce the same results.

assert(np.allclose(t1, t3, rtol= 0.0, atol = 1e-6)); # expected True



