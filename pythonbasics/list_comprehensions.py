# -*- coding: utf-8 -*-
"""
Created on Sun Oct  1 10:45:31 2023

@author: Admin
"""

#%%
# get the square of items in a list

x2 = [x**2 for x in [1,2,3,4,5]];
print("Squares : ", x2);

#%%

# flatten a mult-dimension list using nested list comprehensions
M = [[1,2,3],[4,5,6],[7,8,9,10]];

m = [item for row in M for item in row];

print("Flat List : ", m);

#%%

# using a condition on a list of lists

actors = [
    ['John', 29, True],
    ['Rachel', 56, False],
    ['Smiley', 72, True],
    ];

males = [actor[0] for actor in actors if actor[2] == True];

print("Male Actors = ", males);

#%%

