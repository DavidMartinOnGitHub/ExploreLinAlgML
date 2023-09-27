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
    
    return;
    

#%%
def PrintMatrix(m) :  
    print(retrieve_name(m) + ":\n" , m)   
    return;
    


#%%

A = np.array([[1.0, 2.0, 3.0],  [3.0, 4.0, 5.0], [1.0, 4.0, 3.0]], np.float64);

PrintShape(A);

#%%
Q, R = np.linalg.qr(A);

PrintShape(Q);
PrintShape(R);

#%%
# re-create the original matrix from the decompose Q and R matrices.

B = Q @ R;
PrintShape(B);

#%%

# A and B are expected to be Equal
AreMatricesEqual(A, B);


#%%

PrintMatrix(Q);
PrintMatrix(R);

#%%
print("R is an upper triangle matrix");
print("Q is an Orthogal matrix");

#%%
# because Q is Orthogonal the Q @ Q' is an I matrix (Q' is inverse of Q)

C = Q @ np.linalg.inv(Q);
I = np.rint(C); # round the matrix to integers so 
                # it more obviously appears as an eye-matrix.
print("I:\n" , I);




