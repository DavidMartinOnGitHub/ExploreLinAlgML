# -*- coding: utf-8 -*-
"""
Created on Sat Sep 30 10:56:36 2023

@author: David Martin
"""

#%%
# This code builds upon svm2.py bu adding a prediction pipeline.


#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#%%
#from sklearn.decomposition import PCA;
from sklearn.preprocessing import StandardScaler;
from sklearn.svm import SVC;
from sklearn.model_selection import train_test_split;
from sklearn.pipeline import Pipeline; # For setting up pipeline

#%%

df = pd.read_csv('https://bit.ly/48CQOZS', sep=",", header=0, encoding='utf-8');

#%%

Xo = df.iloc[:,1:];  # all data rows and columns, Xo = X original

Y = df['class'];     # extract class names



#%%
# Split the Xo and Y data into training and test sets.

Xtrain, Xtest, Ytrain, Ytest = train_test_split(Xo, Y, random_state=0, test_size=0.25 );

#%%

pipe = Pipeline([
('scaler', StandardScaler()),
('classifier', SVC(kernel = 'linear',gamma = 'scale', shrinking = False))
]);

#%%

pipe.fit(Xtrain, Ytrain);


#%%


print('Training set score: ' + str(pipe.score(Xtrain,Ytrain)))
print('Test set score: ' + str(pipe.score(Xtest, Ytest)))

