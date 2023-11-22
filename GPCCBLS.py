# -*- coding: utf-8 -*-
"""
Created on Sun Nov  5 16:13:06 2023

@author: YX_W
"""
# %%
from IPython.display import Image, display
import pydotplus
import gplearn
from gplearn.genetic import SymbolicTransformer
import numpy as np
import scipy.io as scio
from BroadLearningSystem import CCR_BLS
import graphviz
# from BroadLearningSystem1 import BLS
# %%
dataFile = 'data.mat'
data = scio.loadmat(dataFile)
traindata = np.double(data['trainx'])
trainlabel = np.double(data['trainy'])
testdata = np.double(data['testx'])
testlabel = np.double(data['testy'])
print((traindata.shape, testdata.shape))

# %%

# gplearn
function_set = ['add', 'sub', 'mul', 'div',
                'sqrt', 'log', 'abs', 'neg', 'inv',
                'max', 'min', 'sin', 'cos', 'tan']

gp = SymbolicTransformer(generations=18, population_size=3000,
                         hall_of_fame=100, n_components=20,
                         function_set=function_set,
                         parsimony_coefficient=0.0005,
                         max_samples=0.9, verbose=1,
                         random_state=0, n_jobs=3)
# gp = SymbolicTransformer(generations=20, population_size=3000,
#                          hall_of_fame=100, n_components=50,
#                          function_set=function_set,
#                          parsimony_coefficient=0.0005,
#                          max_samples=0.9, verbose=1,
#                          random_state=1234, n_jobs=3)
gp.fit(traindata, np.argmax(trainlabel, axis=1))

# %%
# new features
origin_features = np.vstack((traindata, testdata))
gp_features = gp.transform(origin_features)
# %%
print(gp_features.shape)  
new_data = np.hstack((origin_features, gp_features))
traindata_new = new_data[:traindata.shape[0], :]
testdata_new = new_data[traindata.shape[0]:, :]
print((traindata_new.shape, testdata_new.shape))  
# %%

N1 = 23  # of nodes belong to each window
N2 = 20  # of windows -------Feature mapping layer
N3 = 478  # of enhancement nodes -----Enhance layer
s = 0.8  # shrink coefficient
C = 2**-30  # Regularization coefficient


print('--------------------CCR_BLS---------------------------')
metrics = CCR_BLS(traindata_new, trainlabel, testdata_new, testlabel,
                  s, C, N1, N2, N3, sparse='CCR_sparse_bls')
print(metrics) #[Sensitivity, Specificity, Precision, F1-score, Accuracy, AUC]


