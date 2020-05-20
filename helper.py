"""
Python code for the MCL-based Optimum-Path Forest

Reference: H. Bostani, M. Sheikhan, “Modification of Optimum-Path Forest using 
           Markov Cluster Process Algorithm,” In Proc. 2nd International Conference on
           Signal Processing and Intelligent Systems (ICSPIS’2016), 
           pp. 1-5,2016. (Winner of the Outstanding Paper Award)
           DOI: 10.1109/ICSPIS.2016.7869874           

Coded by: Hamid Bostani (st_h_bostani@azad.ac.ir), 2020.

Purpose: Defining various auxiliary functions used in the main code 

Code compatible: Python: 3.*
"""

import collections

def flatten(x):
    if isinstance(x, collections.Iterable):
        return [a for i in x for a in flatten(i)]
    else:
        return [x]
    
def appendToTuples(x,y):
    z=[]
    for i in range(len(x)):
        temp=list(x[i])
        temp.append(y[i])
        z.append(tuple(temp))
    return z