# -*- coding: utf-8 -*-
"""
Created on Fri Aug 11 11:09:33 2023

@author: Admin
"""

#%%
import numpy as np
import inspect



#%%

# usage : print(retrieve_name(y))

def retrieve_name(var):
    callers_local_vars = inspect.currentframe().f_back.f_back.f_locals.items()    
    return [var_name for var_name, var_val in callers_local_vars if var_val is var][0]

#%%
def AreMatricesEqual(a, b) :
    ismatch = np.allclose(a, b, rtol=0.0, atol=1e-08);
    
    print(retrieve_name(a) + ":\n" , a)
    print(retrieve_name(b) + ":\n" , b)

    prefix = "Matrices " + "'" + retrieve_name(a) + "'" + " and " + "'" + retrieve_name(b) + "'" + " are " 
    result = "Equal" if ismatch else "NOT Equal"
    print(prefix + result);
    return;
    


#%%
a = np.array([[1.0, 2.0, 3.0],  [4.0, 5.0, 6.0]], np.float64);
b = np.array([[2.0, 3.0],  [5.0, 6.0], [4.0, 1.0]], np.float64);

expected = np.array([[24.0, 18.0],  [57.0, 48.0]], np.float64);

#%%

c = np.matmul(a, b);


#%%

AreMatricesEqual(c, expected); # should be Equal
#%%

bad = np.array([[1.0, 18.0],  [57.0, 48.0]], np.float64);
AreMatricesEqual(bad, expected); # should NOT be equal


