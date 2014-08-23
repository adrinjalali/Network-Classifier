import sys;
import os;
import numpy as np;
import graph_tool as gt;
from graph_tool import draw;
from graph_tool import spectral;
from graph_tool import stats;
from sklearn import svm;
from sklearn import cross_validation as cv;
from sklearn.metrics import roc_auc_score;
from sklearn.grid_search import GridSearchCV
import sklearn.ensemble
import sklearn.tree
from collections import defaultdict
import time
from joblib import Parallel, delayed, logger
import pickle
import uuid
import re


from constants import *;
from misc import *
from rat import *

#working_dir = '/scratch/TL/pool0/ajalali/ratboost/data_8/TCGA-LAML/risk_group'
#working_dir = '/scratch/TL/pool0/ajalali/ratboost/data_8/TCGA-BRCA/T'
working_dir = '/scratch/TL/pool0/ajalali/ratboost/data_9/TCGA-BRCA/ER'
method = 'ratboost_linear_svc'
cv_index = 10


data_file = np.load(working_dir + '/npdata.npz')
tmpX = data_file['tmpX']
X_prime = data_file['X_prime']
y = data_file['y']
sample_annotation = data_file['sample_annotation']
feature_annotation = data_file['feature_annotation']
g = gt.load_graph(working_dir + '/graph.xml.gz')
cvs = pickle.load(open(working_dir + '/cvs.dmp', 'rb'))

#choosing only one cross-validation fold
tmp = list()
tmp.append((cvs[cv_index]))
cvs = tmp

with open("./rat.py") as f:
    code = compile(f.read(), "rat.py", 'exec')
    exec(code)
rat = Rat(learner_count = 4,
    learner_type = 'linear svc',
    regularizer_index = 10,
    n_jobs = 1)

rat.fit(tmpX[cvs[0][0],],y[cvs[0][0]])
print(roc_auc_score(y[cvs[0][0]], rat.decision_function(
    tmpX[cvs[0][0],])),
    roc_auc_score(y[cvs[0][1]], rat.decision_function(
    tmpX[cvs[0][1],])))


high_C = 1
for pre in np.arange(5) * 0.05 + 0.55:
    low_C = 0
    while(True):
        
        learner = LogisticRegression(penalty = 'l1',
                                     dual = False,
                                     C = high_C,
                                     fit_intercept = True)
        learner = sklearn.svm.LinearSVC(C = high_C,
            penalty = 'l1',
            dual = False)
        learner.fit(tmpX[cvs[0][0],],y[cvs[0][0]])
        
        feature_count = len(learner.coef_[learner.coef_ != 0])
        score = roc_auc_score(y[cvs[0][0]], learner.decision_function(
            tmpX[cvs[0][0],]))
        print(feature_count, score, low_C, high_C)
        if (pre - score > 0.005):
            low_C = high_C
            high_C = high_C * 2
        elif (pre - score < -0.005):
            high_C = (low_C + high_C) / 2
        else:
            break
    print("#################", feature_count, score)
