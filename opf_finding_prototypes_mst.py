"""
Python code for the MCL-based Optimum-Path Forest

Reference: H. Bostani, M. Sheikhan, “Modification of Optimum-Path Forest using 
           Markov Cluster Process Algorithm,” In Proc. 2nd International Conference on
           Signal Processing and Intelligent Systems (ICSPIS’2016), 
           pp. 1-5,2016. (Winner of the Outstanding Paper Award)
           DOI: 10.1109/ICSPIS.2016.7869874           

Coded by: Hamid Bostani (st_h_bostani@azad.ac.ir), 2020.

Purpose: Finding the Prototypes (key nodes) of the input graph derived from the input data set
         based on Minimum Spanning Tree

Code compatible: Python: 3.*
"""

import numpy as np
import networkx as nx
from sklearn.metrics.pairwise import euclidean_distances
import itertools
import helper

def finding_prototypes_mst(Z1):
    G=nx.Graph()
    result=euclidean_distances(Z1[:,0:10],Z1[:,0:10])
    wList=np.round(result[np.triu_indices(np.shape(result)[0])],4)
    wList=wList[wList!=0]
    nodes=list(range(np.shape(result)[0]))
    elist=list(itertools.combinations(nodes, 2))
    weighted_elist=helper.appendToTuples(elist,wList)
    G.add_weighted_edges_from(weighted_elist)
    MST=nx.minimum_spanning_tree(G)    
    MST_edgelist=list(MST.edges())    
    Z1Lable=[i[10] for i in Z1]
    temp=[Z1Lable[i[0]]!=Z1Lable[i[1]] for i in MST_edgelist]
    temp_index=[i for i,e in enumerate(temp) if e==True]
    Prototypes_temp1=[(MST_edgelist[e][0],int(Z1Lable[MST_edgelist[e][0]])) for i,e in enumerate(temp_index)]
    Prototypes_temp2=[(MST_edgelist[e][1],int(Z1Lable[MST_edgelist[e][1]])) for i,e in enumerate(temp_index)]
    Prototypes=np.array(list(set(Prototypes_temp1+Prototypes_temp2)))
    Prototypes=Prototypes[Prototypes[:,0].argsort()]
    return Prototypes