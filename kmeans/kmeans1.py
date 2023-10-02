# -*- coding: utf-8 -*-
"""
Created on Sun Oct  1 16:07:58 2023

@author: David Martin
"""

#%%
import pandas as pd
import numpy as np

#%%
#from sklearn.decomposition import PCA;
from sklearn.preprocessing import StandardScaler;
from sklearn.cluster import KMeans;
from sklearn.model_selection import train_test_split;
from sklearn.pipeline import Pipeline; # For setting up pipeline
from sklearn.metrics import confusion_matrix;

#%%

df = pd.read_csv('https://bit.ly/48CQOZS', sep=",", header=0, encoding='utf-8');

#%%

Xo = df.iloc[:,1:];  # all data rows and columns, Xo = X original

Y = df['class'];     # extract class names

#%%

N = len(Y.unique());

#%%
# Split the Xo and Y data into training and test sets.

Xtrain, Xtest, Ytrain, Ytest = train_test_split(Xo, Y, random_state=0, test_size=0.25 );

#%%

pipe = Pipeline([
('scaler', StandardScaler()),
('classifier', KMeans(n_clusters=N, init='k-means++', n_init=10, max_iter=100, random_state = 42))
]);

#%%

pipe.fit(Xtrain, Ytrain);


#%%

actual = pipe.predict(Xtest);

# K-means is not supervised learning,
# so it's not really possible to get class
# names of the predicted rows, you just
# get the cluster id the row was assigned to.
# The row class does not necessarily correlate to a specific cluster id.

# the following code does not work because actual is
# an array of numbers not class names

#%%


# m = confusion_matrix(Ytest, actual);
# print("Confusion Matrix:\n", m);

#%%
# create a DataFrame with 'actual' and 'expected' predictions

# df = pd.DataFrame()
# df['actual'] = actual
# df['expected'] = np.array(Ytest)

#%%

# # verify that all predicts are correct
# assert((df['actual'] == df['expected']).all())


