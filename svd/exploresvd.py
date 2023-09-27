# -*- coding: utf-8 -*-
"""
Created on Wed Aug  9 13:15:39 2023

@author: David Martin
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
def PrintShape(m) :  
    dim1 = str(m.shape[0]);
    dim2 = "1" if (len(m.shape) == 1) else str(m.shape[1]);
    size = "[" + dim1 + " x " + dim2 + "]";
    name = "'" + retrieve_name(m) + "'";
    msg = "Shape of " + name + " is " + size;
    
    print(msg);
    #print("Shape " + "'" + retrieve_name(m) + "'" + " is " + "[" + str(m.shape[0]) + " x " + str(m.shape[1]) + "]")
    
    
    return;
    
# PrintShape(a)


#%%

A = np.array([[1.0, 2.0, 3.0],  [3.0, 4.0, 5.0]], np.float64);

PrintShape(A);


#%%
u, s, vh = np.linalg.svd(A, full_matrices=True)

# A is a [mxn] matrix
# u is an [mxn]
# s is a vector
# vh is a [nxn] matrix

# to re-create the decomposition the 's' vector must be transformed to a diagonal
# matrix. It's returned as a vector because the diagonal matrix
# is sparse and doing so would be somewhat redundant.
# However, it's not a true diagonal matrix i.e. it's not [2x2] for A
# we must create a [2x3] matrix and fill the principal diagonal with the 
# 's' vector elements. Thus 'S' becomes a [2x3] matrix.



#%%
# convert the 's' vector to a diagonal matrix 'S'

S = np.zeros(A.shape);
np.fill_diagonal(S, s)

#%%

PrintShape(u);
PrintShape(s);
PrintShape(S);
PrintShape(vh);


#%%

B = u @ S @ vh

PrintShape(B);


#%%

AreMatricesEqual(A, B)






