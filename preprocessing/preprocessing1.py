# -*- coding: utf-8 -*-
"""
Created on Sun Oct  1 11:29:11 2023

@author: David Martin
"""

#%%
import pandas as pd
import numpy as np

#%%

def SGConvolution(x):
    filter = np.array([-2.0, -1.0, 0.0, +1.0, +2.0])/10.0;
    reversefilter = np.flip(filter);
    
    if len(x.shape) == 1:
        Xsg = np.convolve(x, reversefilter, mode='valid');
    elif len(x.shape) == 2 and isinstance(x, pd.core.frame.DataFrame):
        Xsg = [];
        rows = x.shape[0];
        for row in range(0, (rows)):
            data = x.iloc[row,:];
            Xsg.append(SGConvolution(data));
        Xsg = np.array(Xsg);
    elif len(x.shape) == 2 and isinstance(x, np.ndarray):
        Xsg = [];
        rows = x.shape[0];
        for row in range(0, (rows)):
            data = x[row];
            Xsg.append(SGConvolution(data));
        Xsg = np.array(Xsg);
    else:
        print(x.shape)
        print(len(x.shape))
        print(type(x))
        print(isinstance(x, type(pd.core.frame.DataFrame)))
        print(isinstance(x, np.ndarray))
        raise ValueError("Illegal dimensions of 'x' argument");
    
    return Xsg;


#%%

def SNV(x):
    if len(x.shape) == 1:
        m = x.mean();
        s = x.std(ddof=1);
        snv = (x - m)/s;
    elif len(x.shape) == 2 and isinstance(x, pd.core.frame.DataFrame):
        snv = [];
        rows = x.shape[0];
        for row in range(0, (rows)):
            data = x.iloc[row,:];
            snv.append(SNV(data));
        snv = np.array(snv);
    elif len(x.shape) == 2 and isinstance(x, np.ndarray):
        snv = [];
        rows = x.shape[0];
        for row in range(0, (rows)):
            data = x[row];
            snv.append(SNV(data));
        snv = np.array(snv);
    else:
        print(x.shape);
        print(type(x));
        raise ValueError("Illegal dimensions of 'x' argument");
    return snv;

#%%

# import raw spectra, all rows all data columns
X = pd.read_csv('https://bit.ly/3EXvRLN', header=0).iloc[:,1:];

#%%

# import pretreated spectra
Xpte = pd.read_csv('https://bit.ly/48CQOZS', header=0).iloc[:,1:];
Xpte = np.array(Xpte);

#%%
X0 = np.array(X);
X1 = SGConvolution(X0);
Xpta = SNV(X1);

assert np.allclose(Xpte, Xpta, rtol=0.0, atol=1e-6)

#%%
Xpta = SNV(SGConvolution(X));

assert np.allclose(Xpte, Xpta, rtol=0.0, atol=1e-6)

#%%
print(type(X0))
print(type(X))
#%%

# verify that the SNV and SGConvolution handle the DataFrame and ndarray
# data types and 1D or 2D

assert np.allclose(SNV(X0), SNV(X), rtol=0.0, atol=1e-6)
assert np.allclose(SGConvolution(X0), SGConvolution(X), rtol=0.0, atol=1e-6)

assert np.allclose(SNV(X0[0]), SNV(X.iloc[0,:]), rtol=0.0, atol=1e-6)
assert np.allclose(SGConvolution(X0[0]), SGConvolution(X.iloc[0,:]), rtol=0.0, atol=1e-6)


#%%



