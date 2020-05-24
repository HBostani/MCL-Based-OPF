"""
Python code for the MCL-based Optimum-Path Forest

Reference: H. Bostani, M. Sheikhan, “Modification of Optimum-Path Forest using 
           Markov Cluster Process Algorithm,” In Proc. 2nd International Conference on
           Signal Processing and Intelligent Systems (ICSPIS’2016), 
           pp. 1-5,2016. (Winner of the Outstanding Paper Award)
           DOI: 10.1109/ICSPIS.2016.7869874           

Coded by: Hamid Bostani (st_h_bostani@azad.ac.ir), 2020.

Purpose: Defining the learning function of OPF

Code compatible: Python: 3.*
"""

import numpy as np
import opf_train as tr
import opf_classify as cf
import random

def learning(Z1,Z2,Prototypes,iteration_count,labels):
    max_acc=-1
    Model_best=[]
    for iteration in range(iteration_count):
        print("The current iteration number of the learning phase of OPF is %s" %(iteration+1))
        Model=tr.train(Z1,Prototypes)
        Classification=cf.classify(Z1,Z2,Model)   
        FP=np.zeros(shape=(1,2))
        FN=np.zeros(shape=(1,2))
        NZ2=np.zeros(shape=(1,2))
        sw=True
        E=0
        for i in range(len(labels)):
            if sw == True:
                NZ2[0,0]=labels[i]
                cmp=[Z2[j,10] == labels[i] for j,e in enumerate(Z2)]
                NZ2[0,1]=cmp.count(True)                
                FP[0,0]=labels[i]
                cmp=[Z2[j,10] != labels[i] and Classification[j,3] == labels[i] for j,e in enumerate(Z2)]
                FP[0,1]=cmp.count(True)                
                FN[0,0]=labels[i]
                cmp=[Z2[j,10] == labels[i] and Classification[j,3] != labels[i] for j,e in enumerate(Z2)]
                FN[0,1]=cmp.count(True) 
                sw=False                     
            else:
                cmp=[Z2[j,10] == labels[i] for j,e in enumerate(Z2)]
                NZ2=np.append(NZ2,[[labels[i],cmp.count(True)]],axis=0)                
                cmp=[Z2[j,10] != labels[i] and Classification[j,3] == labels[i] for j,e in enumerate(Z2)]
                FP=np.append(FP,[[labels[i],cmp.count(True)]],axis=0)                
                cmp=[Z2[j,10] == labels[i] and Classification[j,3] != labels[i] for j,e in enumerate(Z2)]
                FN=np.append(FN,[[labels[i],cmp.count(True)]],axis=0)
            e1=FP[i,1]/(len(Z2)-NZ2[i,1])
            e2=FN[i,1]/(NZ2[i,1])
            e=e1+e2
            E+=e
        L=1-(E/(2*len(labels)))
        if L>max_acc:
            max_acc=L
            Model_best=Model        
        cmp=[Z2[i,10] != Classification[i,3] for i,e in enumerate(Z2)]
        wrong_labels_index=[i for i,e in enumerate(cmp) if e==True]
        Z1_selected_indices=[]
        for i in range(len(wrong_labels_index)):
            index=[j for j,e in enumerate(Z1) if e[10]==Z2[wrong_labels_index[i],10] and j not in Prototypes[:,0] and j not in Z1_selected_indices]
            if len(index) == 0:
                index=[j for j,e in enumerate(Z1) if e[10]==Z2[wrong_labels_index[i],10] and j not in Prototypes[:,0]]
            Z2_temp=np.array(Z2[wrong_labels_index[i],:])
            Z1_current_selected_index=random.choice(index)
            Z1_selected_indices.append(Z1_current_selected_index)
            Z1_temp=np.array(Z1[Z1_current_selected_index,:])
            Z2[wrong_labels_index[i],:]=Z1_temp
            Z1[Z1_current_selected_index,:]=Z2_temp
    return Model_best
 