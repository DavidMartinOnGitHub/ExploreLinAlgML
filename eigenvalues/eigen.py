# -*- coding: utf-8 -*-
"""
Created on Wed Sep 27 09:16:12 2023

@author: David Martin
"""

#%%
import numpy as np

#%%

A = np.array([
    [1,2],
    [4,5]
    ]);

#%%
# calculate the eigenvalues and eigenvectors of A
eigenvalues , eigenvectors = np.linalg.eig(A);

#%%
# the eigenvectors are the columns 
v1 = eigenvectors[:,0]
v2 = eigenvectors[:,1]

#%%
# the eigenvalues are the elements
y1 = eigenvalues[0]
y2 = eigenvalues[1]



#%%
# for the first eigenvalue/eigenvector

# A.v1 = y1*v1  
# where A.v1 is matrix muliplication 
# and y1*v1 is scalar multiplication
L1 = np.matmul(A, v1)
R1 = (y1 * v1)

print("L1")
print(L1)
print("R1")
print(R1)
# L1 and R1 are expected to be equal

#%%
# for the second eigenvalue/eigenvector

L2 = np.matmul(A, v2)
R2 = (y2 * v2)

print("L2")
print(L2)
print("R2")
print(R2)
# L2 and R2 are expected to be equal


#%%

L = np.matmul(A,eigenvectors) # matrix multiplication
R = eigenvalues*eigenvectors  # scalar multiplication

print("L")
print(L)
print("R")
print(R)
# L and R are expected to be equal



#%%


# A = Q.Y.Q' 
# where
# Q are eigenvectors 
# Y are the eigenvalues in diagonal matrix form
# Q' is the inverse of Q,

Q = eigenvectors
Y = np.diag(eigenvalues)

B = np.matmul(np.matmul(Q, Y), np.linalg.inv(Q))

print("A")
print(A)
print("B")
print(B)

#%%
# To eliminate the scalar multiplication and use matrix multiplcation
# we put the eigenvalues into a diagonal matrix and multiply on the right
R3 = np.matmul(eigenvectors, np.diag(eigenvalues))
print("L")
print(L)
print("R3")
print(R3)
# L and R3 are expected to be equal

