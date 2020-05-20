"""
Python code for the MCL-based Optimum-Path Forest

Reference: H. Bostani, M. Sheikhan, “Modification of Optimum-Path Forest using 
           Markov Cluster Process Algorithm,” In Proc. 2nd International Conference on
           Signal Processing and Intelligent Systems (ICSPIS’2016), 
           pp. 1-5,2016. (Winner of the Outstanding Paper Award)
           DOI: 10.1109/ICSPIS.2016.7869874           

Coded by: Hamid Bostani (st_h_bostani@azad.ac.ir), 2020.

Purpose: Defining a priority queue used in training and classification phases of OPF 

Code compatible: Python: 3.*
"""

import numpy as np

class MyPriorityQueue:    
    def push(self,node,weight,parent,label,root):
        if hasattr(self, 'queue'):
           self.queue=np.append(self.queue,[[node,weight,parent,label,root]],axis=0)
        else:
            self.queue=np.array([[node,weight,parent,label,root]])
    def pop(self,priority_index):
        if len(self.queue) > 0:
            weights=self.queue[:,priority_index]
            min_weight=np.where(weights == np.amin(weights))
            min_weight_index=min_weight[0][0]
            result=self.queue[min_weight_index,:]
            self.queue=np.delete(self.queue,min_weight_index,axis=0)
        else:
            result=[]
        return result
    def remove(self,node_id):
        row_index=np.where(self.queue[:,0]==node_id)
        self.queue=np.delete(self.queue,row_index,axis=0)
    def empty(self):
        return len(self.queue) == 0        
   
        
        