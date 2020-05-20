# -*- coding: utf-8 -*-
"""
Python code for the MCL-based Optimum-Path Forest

Reference: H. Bostani, M. Sheikhan, “Modification of Optimum-Path Forest using 
           Markov Cluster Process Algorithm,” In Proc. 2nd International Conference on
           Signal Processing and Intelligent Systems (ICSPIS’2016), 
           pp. 1-5,2016. (Winner of the Outstanding Paper Award)
           DOI: 10.1109/ICSPIS.2016.7869874           

Coded by: Hamid Bostani (st_h_bostani@azad.ac.ir), 2020.

Purpose: The main file of comparing standard OPF and the proposed MCL-based OPF

Code compatible: Python: 3.*
"""

import numpy as np
import opf_finding_prototypes_mst as pr_mst
import opf_train as tr
import opf_classify as cf

Dataset=np.genfromtxt('magic04Normalized.csv',delimiter=',')
LabelIndex=11
Z1=Dataset[0:500]
Z2=Dataset[500:1000]

Prototypes=pr_mst.finding_prototypes_mst(Z1)
Model=tr.train(Z1,Prototypes)
Classification=cf.classify(Z1,Z2,Model)
cmp_classification=[Z2[i,10] == Classification[i,3] for i,e in enumerate(Z2)]
accuracy_classification=(cmp_classification.count(True)/np.shape(Z2)[0])*100
print("Accuracy of classification: ",accuracy_classification)


