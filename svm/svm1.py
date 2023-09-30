# -*- coding: utf-8 -*-
"""
Created on Fri Sep 29 09:57:59 2023

@author: David Martin
"""
#%% REFERENCES
# https://www.tutorialspoint.com/scikit_learn/scikit_learn_support_vector_machines.htm

#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#%%
#from sklearn.decomposition import PCA;
from sklearn.preprocessing import StandardScaler;
from sklearn.svm import SVC;

#%%

def PlotSpectra(x, y, colors):
    # x is the X matrix of spectra, 
    # rows are samples and 
    # columns are intensities at wavelengths
    # y is the list of class names one per sample
    # colors is a cross-reference color dictionary, key is the class name and value is the color
    
    
    N = len(x);
    xidx = list(range(len(df.columns)-1));
    # xidx is a list of the 'x-values' to be plotted

    fig, ax = plt.subplots(figsize=(5, 3.0), dpi=200)
    for index in range(N):
        ax.plot(xidx, x.iloc[index,:], color=colors[y[index]]);
    
    ax.set_title("Spectra")  # Add a title to the axes.
        
    return
#%%

# filename = r"C:\Users\Admin\source\repos\LinearAlgebra\svm\ptspectra.csv";
filename = r"ptspectra.csv";

#%%

df = pd.read_csv(filename, sep=",")


#%%

# extract the class names from the data frame
Y = df['class'];

#%%

# extract all data columns and all rows of the DataFrame
# i.e. the spectra data.

X = df.iloc[:,1:];

#%%

# plot each spectra class in a different color

colordictionary = {"citric-acid" : "red",
                   "flour" : "blue",
                   "salt" : "yellow",
                   "cream-tartar" : "green",
                   "sugar" : "black",
                   };

#%%

PlotSpectra(X, Y, colordictionary);



#%%

names = Y.unique();

#%%
x_scaled = StandardScaler().fit_transform(X);

print("mean =", x_scaled[:,120].mean().round(8))  # expect ~= 0.0
print("std deviation = ", x_scaled[:,120].std().round(8))   # expect ~= 1.0



#%%

# plot the auto-scaled spectra

PlotSpectra(pd.DataFrame(data=x_scaled), Y, colordictionary);


#%%

svc = SVC(kernel = 'linear',gamma = 'scale', shrinking = False);
svc.fit(x_scaled, Y);

#%%

index = 45;

array1d = x_scaled[index,:];
array2d= [array1d];
         
#%%

# svc.predict requires a 2D array, since the indexed x_scaled 
# is a 1D array we wrap the 1D array in [] to make it into 2D.

# svc.predict returns an array, since we're doing one prediction
# we get an array of one element so assign p to be the 0th element.

p = svc.predict(array2d)[0];


#%%

# perform prediction on all rows in the training set.

sp = svc.predict(x_scaled);

print((sp == Y).all()) # True if all predictions match the original class

#%%

# exp[ore the support vectors, they are rows in the training set that define
# the hyperspace boundaries between the classes. 
v = 0;
index = svc.support_[v];
print(np.allclose(x_scaled[index,:], 
                  svc.support_vectors_[v], 
                  rtol=0.0, 
                  atol=1e-6)); 
# expect the above equality test to be True

print("Number of Support Vectors for each class", svc.n_support_);

#%%





