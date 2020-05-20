"""
Python code for the MCL-based Optimum-Path Forest

Reference: H. Bostani, M. Sheikhan, “Modification of Optimum-Path Forest using 
           Markov Cluster Process Algorithm,” In Proc. 2nd International Conference on
           Signal Processing and Intelligent Systems (ICSPIS’2016), 
           pp. 1-5,2016. (Winner of the Outstanding Paper Award)
           DOI: 10.1109/ICSPIS.2016.7869874           

Coded by: Hamid Bostani (st_h_bostani@azad.ac.ir), 2020.

Purpose: Defining classify function of OPF

Code compatible: Python: 3.*
"""

import numpy as np
from scipy.spatial import distance

def classify(Z1,Z2,Model):
    Classification=np.zeros((np.shape(Z2)[0],4))
    for k in range(np.shape(Z2)[0]):
        Classification[k,0]=k
        i=0
        dst = distance.euclidean(Z1[int(Model[i,0]),1:10], Z2[k,1:10])      
        max_cost=max(Model[i,1],dst)
        Classification[k,1]=max_cost
        Classification[k,2]=Model[i,2]
        Classification[k,3]=Model[i,3]
        #print("k: ", k)
        while i<len(Model)-1 and max_cost>Model[i+1,1]:
           #print("i+1: ",i+1)
           temp_dst = distance.euclidean(Z1[int(Model[i+1,0]),1:10], Z2[k,1:10])      
           temp_max_cost=max(Model[i+1,1],temp_dst)
           if temp_max_cost<max_cost:
               min_cost=temp_max_cost
               Classification[k,1]=min_cost
               Classification[k,2]=Model[i+1,2]
               Classification[k,3]=Model[i+1,3]
           i+=1
    return Classification
    