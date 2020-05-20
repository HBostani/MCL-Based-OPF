"""
Python code for the MCL-based Optimum-Path Forest

Reference: H. Bostani, M. Sheikhan, “Modification of Optimum-Path Forest using 
           Markov Cluster Process Algorithm,” In Proc. 2nd International Conference on
           Signal Processing and Intelligent Systems (ICSPIS’2016), 
           pp. 1-5,2016. (Winner of the Outstanding Paper Award)
           DOI: 10.1109/ICSPIS.2016.7869874           

Coded by: Hamid Bostani (st_h_bostani@azad.ac.ir), 2020.

Purpose: Defining train function of OPF

Code compatible: Python: 3.*
"""
import numpy as np
import MyPriorityQueue as queue
from scipy.spatial import distance

def train(Z1,Prototypes):
    Model=np.zeros((np.shape(Z1)[0],5))
    """
    index 0: node id, index 1: cost, index 2: parent, index 3: label, 
    index 4: root
    """
    Model[:,0]=range(np.shape(Model)[0])
    Model[:,1]=10000
    Model[Prototypes[:,0],1]=0
    Model[Prototypes[:,0],3]=Prototypes[:,1]
    Model[Prototypes[:,0],4]=Prototypes[:,0]
    priority_queue = queue.MyPriorityQueue()
    for item in Prototypes:
        priority_queue.push(item[0],0,0,item[1],item[0])
    
    while priority_queue.empty() == False:
        node,cost,parent,label,root=priority_queue.pop(1)
        for k in range(np.shape(Z1)[0]):        
            dst = distance.euclidean(Z1[int(node),1:10], Z1[k,1:10])      
            max_cost=max(cost,dst)
            if max_cost<Model[k,1]:
                if Model[k,1]!=10000:
                    priority_queue.remove(Model[k,0])
                Model[k,:]=[Model[k,0],max_cost,int(node),label,int(root)]
                priority_queue.push(Model[k,0],Model[k,1],Model[k,2],Model[k,3],Model[k,4])
    Model=Model[Model[:,1].argsort()]
    return Model