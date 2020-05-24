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
import opf_finding_prototypes_mcl as pr_mcl
import opf_classify as cf
import opf_learning as le

Dataset=np.genfromtxt('magic04Normalized.csv',delimiter=',')
LabelIndex=11
Z1=Dataset[0:1000]
Z2=Dataset[2000:2500]
Z3=Dataset[1000:3000]
labels=list(set([i[10] for i in Z1]))
iteration_count=5

prototypes_identification_method="MST"
if prototypes_identification_method == "MST":
    Prototypes=pr_mst.finding_prototypes_mst(Z1)
else:
    Prototypes=pr_mcl.finding_prototypes_mcl(Z1)
Model_best=le.learning(Z1,Z2,Prototypes,iteration_count,labels)
Classification=cf.classify(Z1,Z3,Model_best)            
cmp_classification=[Z3[i,10] == Classification[i,3] for i,e in enumerate(Z3)]
accuracy_classification=(cmp_classification.count(True)/np.shape(Z3)[0])*100
print("The Accuracy of classification: ",accuracy_classification)


