# -*- coding: utf-8 -*-
"""
Created on Wed Sep 27 15:27:30 2023

@author: David Martin

"""


#%%
import numpy as np;
import pandas as pd;
import matplotlib.pyplot as plt

#%%

df = pd.read_csv('https://bit.ly/3goOAnt', delimiter=",")

#%%

x = df.values[:, :-1].flatten();  # 
y = df.values[:, -1];   # vecor of length 10

#%%
X = df.values[:, 0];  # vector of length 10
Y = df.values[:, 1];   # vector of length 10


#%%
# augment the X matrix with a column filled with ones, to accomodate 
# the Beta0 term.

X1 = np.vstack([X, np.ones(len(X))]).transpose()


#%%


Q, R = np.linalg.qr(X1);

#%%

# beta = R'.QT.y # R' is inverse of R, QT = transpose of Q

beta = np.linalg.inv(R) @ Q.transpose() @ y;


#%%

# calculate to y-predicted values i.e. the regression line
yp = X1 @ beta;


#%%

plt.title('Linear Regression using scatter plot')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')

plt.grid()
plt.plot(X,Y, '.r ')  # red dots
plt.plot(X,yp, 'xb-') # blue line
plt.show()






