# -*- coding: utf-8 -*-
"""
Created on Sun Oct  1 12:23:49 2023

@author: David Martin
"""

#%%


# This git repository has two data sets. One is raw spectra and the other is 
# pretreated spectra. These data sets are in the '\data' directory.
# Since the repository is hosted on GitHub they are accessible via URLs:
# * https://raw.githubusercontent.com/DavidMartinOnGitHub/ExploreLinAlgML/main/data/spectra.csv
# * https://raw.githubusercontent.com/DavidMartinOnGitHub/ExploreLinAlgML/main/data/ptspectra.csv
# The GitHub 'raw' view is used otherwise the URL content will be HTML and unparseable
# I used bit.ly to shorten the URLs:
# * https://bit.ly/3EXvRLN
# * https://bit.ly/48CQOZS
# The bit.ly is mapped as follows:
# https://bit.ly/3EXvRLN => spectra.csv
# https://bit.ly/48CQOZS => ptspectra.csv

#%%
import pandas as pd

#%%

# use the local file to import the data
filename = r'.\..\data\spectra.csv';
df = pd.read_csv(filename, sep=",", header=0, encoding='utf-8');


#%%

# Extract the X data matrix and Y class names 
# First column in df is 'class', the rest of the columns are X1,X2,..,X125

X = df.iloc[:,1:];  # extract the X1..X125 columns all rows
Y = df['class'];    # extract the class name column

#%%

# use the URL to download and import the data
url = r'https://raw.githubusercontent.com/DavidMartinOnGitHub/ExploreLinAlgML/main/data/spectra.csv';
df2 = pd.read_csv(url, sep=",", header=0, encoding='utf-8');

#%%

# use the bit.ly shortened URL to import the data
df3 = pd.read_csv('https://bit.ly/3EXvRLN', sep=",", header=0, encoding='utf-8');

#%%
