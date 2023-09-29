# -*- coding: utf-8 -*-
"""
Created on Fri Sep 29 09:57:59 2023

@author: David Martin
"""

#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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

filename = r"C:\Users\Admin\source\repos\LinearAlgebra\pca\ptspectra.csv";

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

# highlight the 'salt' spectra

highlightdictionary = {"citric-acid" : "grey",
                   "flour" : "grey",
                   "salt" : "red",
                   "cream-tartar" : "grey",
                   "sugar" : "grey",
                   };

#%%

PlotSpectra(X, Y, highlightdictionary);






