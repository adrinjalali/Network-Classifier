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
working_dir = '/scratch/TL/pool0/ajalali/ratboost/data_21_sep_2014/TCGA-THCA/ajcc_pathologic_tumor_stage'
method = 'ratboost_linear_svc'
cv_index = 2
batch_based_cv = True

data_file = np.load(working_dir + '/data.npz');
X = data_file['X']
X_prime = data_file['X_prime']
y = data_file['y']
sample_annotation = data_file['patient_annot']
data_file = np.load(working_dir + '/../genes.npz')
feature_annotation = data_file['genes']
g = gt.load_graph(working_dir + '/../graph.xml.gz')
if (batch_based_cv):
    cvs = pickle.load(open(working_dir + '/batch_cvs.dmp', 'rb'))
else:
    cvs = pickle.load(open(working_dir + '/normal_cvs.dmp', 'rb'))

#choosing only one cross-validation fold
tmp = list()
tmp.append((cvs[cv_index]))
cvs = tmp

with open("./rat.py") as f:
    code = compile(f.read(), "rat.py", 'exec')
    exec(code)
    
rat = Rat(learner_count = 4,
    learner_type = 'linear svc',
    regularizer_index = 18,
    n_jobs = 40)

rat.fit(X[cvs[0][0],],y[cvs[0][0]])

print(roc_auc_score(y[cvs[0][0]], rat.decision_function(
    X[cvs[0][0],])),
    roc_auc_score(y[cvs[0][1]], rat.decision_function(
    X[cvs[0][1],])))


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



# testing RBFard on synthesized data

import scipy.io
import pyGPs
import numpy as np
data_dir = '/TL/stat_learn/work/ajalali/Network-Classifier/synthesized_results-1'
bnet_count = 5
feature_noise = .1
tmp = scipy.io.loadmat('%s/data-bnet_count-%d-feature_noise-%g.mat' % (data_dir, bnet_count, feature_noise))
Xtrain = tmp['Xtrain']
ytrain = tmp['ytrain'].reshape(-1)
Xtest = tmp['Xtest']
ytest = tmp['ytest'].reshape(-1)
train_feature_noise = tmp['train_feature_noise']
test_feature_noise = tmp['test_feature_noise']
train_feature_bnet = tmp['train_feature_bnet']
test_feature_bnet = tmp['test_feature_bnet']

model = pyGPs.GPR()
#k = pyGPs.cov.RBFard(log_ell_list=[0.05,0.17], log_sigma=1.)
k = pyGPs.cov.RBFard(D=1030)#D = Xtrain.shape[1], log_sigma = 0)
mean = pyGPs.mean.Const(0)
model.setPrior(kernel=k, mean = mean)
model.optimize(Xtrain[:,1:], Xtrain[:,0])
np.hstack((np.hstack(model.predict(Xtrain[1:40,1:])[0:2]), Xtrain[1:40,0:1]))

model.setData(Xtrain[:,1:], Xtrain[:,0])
#model.plotData_2d(x1,x2,t1,t2,p1,p2)

#model.getPosterior()
model.optimize(Xtrain[:,1:], Xtrain[:,0])
np.hstack((np.hstack(model.predict(Xtrain[1:40,1:])[0:2]), Xtrain[1:40,0:1]))
model.predict(Xtrain[1:3,1:30])


import sklearn
import sklearn.ensemble
def gradientboostingclassifier_get_feature_importances(model):
    tmp = np.vstack((np.arange(len(machine.feature_importances_))[machine.feature_importances_ != 0], machine.feature_importances_[machine.feature_importances_ != 0]));
    tmp = tmp[:,np.argsort(tmp)[1,]];
    [print('%d, %g' % (tmp[0,i], tmp[1,i])) for i in range(tmp.shape[1])];
        
machine = sklearn.ensemble.GradientBoostingRegressor(
    max_features = 5,
    subsample=.8,
    max_depth = 3,
    n_estimators = 400)
machine.fit(Xtrain[:,1:], Xtrain[:,0])
gradientboostingclassifier_get_feature_importances(machine)


from minepy import MINE
def _evaluate_single(data, target_feature):
    mine = MINE(alpha=0.4, c=15)
    MICs = list()
    for i in range(data.shape[1]):
        mine.compute_score(target_feature,data[:,i])
        MICs.append((mine.mic(), mine.mas(), mine.mev(), mine.mcn(), mine.mcn_general()))
    return(MICs)
mics = np.array(_evaluate_single(Xtrain[:,:], Xtrain[:,1]))
values = np.array([mics[i][0] for i in range(len(mics))])
tmp = np.vstack((np.arange(len(values))[values != 0], values[values != 0]));
tmp = tmp[:,np.argsort(tmp)[1,]];
[print('%g, %g' % (tmp[0,i], tmp[1,i])) for i in range(tmp.shape[1])];


clf = sklearn.linear_model.Ridge(alpha=100, copy_X=True, fit_intercept=True,
                                 normalize=True, solver='auto', max_iter=10000)

from sklearn import linear_model
clf = linear_model.ARDRegression(normalize = False, copy_X = True, verbose = True)

from sklearn.linear_model import SGDRegressor
clf = SGDRegressor(n_iter=5000)

from sklearn.linear_model import PassiveAggressiveRegressor
clf = PassiveAggressiveRegressor(n_iter=1000, epsilon=0.01)

import sklearn.linear_model
clf = sklearn.linear_model.TheilSenRegressor(fit_intercept=True, copy_X=True, max_subpopulation=10000.0, n_subsamples=None, max_iter=1000, tol=0.001, random_state=None, n_jobs=1, verbose=False)

clf = sklearn.linear_model.LinearRegression(fit_intercept=True, normalize=True, copy_X=True, n_jobs=10)

clf = sklearn.linear_model.BayesianRidge(n_iter=300, tol=0.001, alpha_1=1e-06, alpha_2=1e-06, lambda_1=1e-06, lambda_2=1e-06, compute_score=False, fit_intercept=True, normalize=False, copy_X=True, verbose=False)

import sklearn
import sklearn.kernel_ridge
clf = sklearn.kernel_ridge.KernelRidge(alpha=10, kernel='linear', gamma=None, degree=3, coef0=1, kernel_params=None)

clf.fit(Xtrain[:,1:], Xtrain[:,0])
print(sum(abs(Xtrain[:,0] - clf.predict(Xtrain[:,1:]))))
print(sum(abs(Xtest[:,0] - clf.predict(Xtest[:,1:]))))
print(sum(np.abs(clf.coef_)))
