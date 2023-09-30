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
from sklearn.decomposition import PCA;

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

df = pd.read_csv('https://bit.ly/48CQOZS', sep=",", header=0, encoding='utf-8')

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

#%%

names = Y.unique();

#%%

from sklearn.preprocessing import StandardScaler;

#%%
x_scaled = StandardScaler().fit_transform(X);

print("mean =", x_scaled[:,120].mean().round(8))  # expect ~= 0.0
print("std deviation = ", x_scaled[:,120].std().round(8))   # expect ~= 1.0

#%%

npca = 5; # number of PC components

pca = PCA(n_components=npca);
xpca = pca.fit_transform(x_scaled);


#%%
# TODO use the n_components to generate the column headers by concatenating 'PC' and index... 

pccolumns =  ['PC1','PC2','PC3','PC4','PC5' ];

#%%
pcadf = pd.DataFrame(data = xpca, columns = pccolumns);

#%%

pcadf['target'] = Y;



#%%

# Explained Variance

# cumulative explained variance x-axis data
cevx = range(1, (npca+1));

# cumulative explained variance Y-axis data
cevy = np.cumsum(pca.explained_variance_ratio_)*100.0; 


fig1, ax1 = plt.subplots(figsize=(5, 3.0), dpi=200)

# plot the cumulative explained variance
ax1.plot(cevx, cevy, c='red');
ax1.set_title('Cumulative Explained Variance');
ax1.set_xlabel('PC Component');
ax1.set_xticks(cevx);
ax1.set_ylabel('Percent');

#%%

pcx = pcadf['PC1'];
pcy = pcadf['PC2'];


#%%

colordictionary = {"citric-acid" : "red",
                   "flour" : "blue",
                   "salt" : "yellow",
                   "cream-tartar" : "green",
                   "sugar" : "black",
                   };

colors = pcadf['target'].map(colordictionary)


#%%

fig2, ax2 = plt.subplots(figsize=(5, 3.0), dpi=200);
ax2.scatter(pcx,pcy, marker='o',c=colors);
ax2.set_title('Principal Components');
ax2.set_xlabel('PC1');
ax2.set_ylabel('PC2');


#%%

#%%





