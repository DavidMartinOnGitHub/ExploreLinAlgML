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
from sklearn.model_selection import train_test_split;

#%%

df = pd.read_csv('https://bit.ly/48CQOZS', sep=",", header=0, encoding='utf-8')

#%%

Xo = df.iloc[:,1:];  # all data rows and columns, Xo = X original

Y = df['class'];     # extract class names



#%%
# Split the Xo and Y data into training and test sets.

Xtrain, Xtest, Ytrain, Ytest = train_test_split(Xo, Y, random_state=0, test_size=0.25 )

#%%

# scale the X Training Set, then use the same transformer to
# scale the X Test Set, using the columns means and standard
# deviations from the X training set.

scaler = StandardScaler();
scaler.fit(Xtrain);
Xt = scaler.transform(Xtrain);
Xp = scaler.transform(Xtest);

#%%

# build the SVM model using the auto-scaled X training set and the Y training set.
svc = SVC(kernel = 'linear',gamma = 'scale', shrinking = False);
svc.fit(Xt, Ytrain);

#%%

# perform prediction on all rows in the auto-scaled test set.
predictions = svc.predict(Xp);

#%%

# print True if all predictions are correct
print((Ytest == predictions).all());

#%%

# alternatively use the score
print("Test Set Accuracy = ", svc.score(Xtest,Ytest)*100.0, "%")

