# -*- coding: utf-8 -*-
"""
Created on Wed Sep 27 16:39:43 2023

@author: Admin
"""

#%%

from scipy.stats import norm;
import matplotlib.pyplot as plt
import numpy as np



#%%
N = ((25.0-12.0)/0.1)
X = np.linspace(12.0, 25.0, int(N+1))




#%%
Y = norm.cdf(X, 18.0, 1.5)

#%%
plt.title('CDF plot')
plt.xlabel('X-axis')
plt.ylabel('cdf')

plt.grid()
plt.plot(X,Y, '.r ')  # red dots
plt.show()


#%%
x = np.linspace(norm.ppf(0.000001), norm.ppf(0.999999), 10000);

mean = 0.0;
std_dev = 1.0;
y1 = norm.pdf(x, mean, std_dev);

mean = 0.0;
std_dev = 1.5;
y2 = norm.pdf(x, mean, std_dev);



#%%
plt.title('density plot')
plt.xlabel('X-axis')
plt.ylabel('pdf')

plt.grid()
plt.plot(x,y1, '-r')  # red line
plt.plot(x,y2, '-b')  # blue line
plt.show()




#%%

x = np.linspace(norm.ppf(0.000001), norm.ppf(0.999999), 10000);

y3 = norm.ppf(x, mean, std_dev);

#%%
plt.title('PPF plot')
plt.xlabel('X-axis')
plt.ylabel('ppf')

plt.grid()
plt.plot(x,y3, '-r')  # red line
plt.show()





