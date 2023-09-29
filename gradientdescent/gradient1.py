# -*- coding: utf-8 -*-
"""
Created on Wed Sep 27 20:05:02 2023

Use Gradient Descent to determine the minima of a quadratic function.

@author: Admin
"""

#%%

import numpy as np;
import matplotlib.pyplot as plt
import random;

#%%

# our function of x
def f(x):
    return (x-3)**2 + 4;

#%%
# derivative wrt x of our function
def dfdx(x):
    return 2 * (x-3);

#%%

L = 0.001 # Learning Rate

#%%

iterations = 100000;

#%%
xstart = -3.0;
xend = +9.0;

X = np.linspace(xstart, xend, 1000);
Y = f(X);


#%%
plt.title('X~Y plot')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')

plt.grid()
plt.plot(X,Y, '-r')  # red line
plt.xlim((-5.0, +12.0));
plt.ylim((0.0, +42.0));
plt.show()

# shows the minima at X ~= 3.0 with Y ~= 4.0


#%%
# use a seed value so that the random number is consistent
random.seed(42);

# get a random initial value for x
x = random.randint(-15.0, +15.0);

#%%

values = np.zeros((iterations+1));
values[0] = x;


#%%

for i in range(iterations):
    slope = dfdx(x);
    x -= L * slope;
    values[i+1] = x;
    
#%%

for i in range(iterations+1):
    if((i) % 10 == 0) : print(i, values[i]);
    
#%%

print("After " + str(iterations) + " iterations the minima is estimated to be at x = " + str(x) +", y = " + str(f(x)) )
    
#%%
plt.title('Convergence Plot')
plt.xlabel('iteration')
plt.ylabel('x-value')


xchart = np.linspace(0, iterations, (iterations+1));

plt.grid()
plt.plot(xchart,values, '-r')  # red line
plt.show()

#%%
plt.title('Convergence Plot')
plt.xlabel('iteration')
plt.ylabel('x-value')


plt.grid()
plt.plot(xchart[0:100],values[0:100], '-b')  # blue line
plt.show()


