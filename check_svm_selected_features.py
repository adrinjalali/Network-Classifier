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
import matplotlib as mpl
#mpl.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde


from constants import *;
from misc import *
from rat import *

#working_dir = '/scratch/TL/pool0/ajalali/ratboost/data_8/TCGA-LAML/risk_group'
#working_dir = '/scratch/TL/pool0/ajalali/ratboost/data_8/TCGA-BRCA/T'
working_dir = '/scratch/TL/pool0/ajalali/ratboost/data_7/TCGA-LAML-GeneExpression/risk_group'


data_file = np.load(working_dir + '/npdata.npz')
tmpX = data_file['tmpX']
X_prime = data_file['X_prime']
y = data_file['y']
sample_annotation = data_file['sample_annotation']
feature_annotation = data_file['feature_annotation']
g = gt.load_graph(working_dir + '/graph.xml.gz')
cvs = pickle.load(open(working_dir + '/cvs.dmp', 'rb'))


def get_l1_svm(X, y, fc):
    high = 1/1000
    low = 0
    while (True):
        learner = sklearn.svm.LinearSVC(penalty = 'l1',
            dual = False, C = high, verbose=False)
        learner.fit(X, y)
        t_fc = np.sum(learner.coef_ != 0)
        if (t_fc > fc):
            high = (low + high) / 2
        elif (t_fc < fc):
            low = high
            high = high * 2
        else:
            break
    return learner

def get_features(learner):
    return np.arange(learner.coef_.shape[1])[learner.coef_.reshape(-1) != 0]

def fit_and_get_features(X, y, fc):
    learner = get_l1_svm(X, y, fc)
    return get_features(learner)

def get_learner_score(learner, X, y):
    return roc_auc_score(y, learner.decision_function(X))

def get_l1_svm_heuristic(X, y):
    fc = 1
    old_score = 0
    while (True):
        learner = get_l1_svm(X, y, fc)
        score = get_learner_score(learner, X, y)
        print (old_score, score, fc)
        if (score > 0.9 and score > old_score + 0.03):
            break
        if (score > 0.95):
            break
        fc = fc + 1
        old_score = score
        
        
    return learner
    
def exclude_cols(X, cols):
    ''' exludes indices in cols, from columns of X '''
    return(X[:,~np.in1d(np.arange(X.shape[1]), cols)])

def convert_to_real_features(fl, ex, max):
    return np.array(list(set(np.arange(max)) - set(ex)))[fl]

def get_decision_values(learners, ls_features, X, y_priors, l_count):
    log_probas = learners[0].predict_log_proba(X[:, ls_features[0]])
    for i in range(1, l_count):
        log_probas = log_probas + \
            learners[i].predict_log_proba(X[:, ls_features[i]])
    return log_probas[:,1] + l_count * np.log(y_priors[1]) - \
      (log_probas[:,0] + l_count * np.log(y_priors[0]))
    
from sklearn.metrics import precision_recall_fscore_support
tr_score = list()
te_score = list()
i = 1
for train, test in cvs:
    print(i)
    i = i + 1

    learners = list()
    l2_learners = list()
    ex = np.ndarray(shape=(0), dtype=int)
    ex_list = list()
    ls_features = list()
    l_count = 4
    for k in range(l_count):
        ex_list.append(ex.copy())
        my_X = exclude_cols(tmpX, ex)
        learner = get_l1_svm_heuristic(my_X[train,], y[train])
        learners.append(learner)
        l_features = convert_to_real_features(get_features(learner),
                                              ex,
                                              tmpX.shape[1])
        ls_features.append(l_features)
        print(sorted(feature_annotation[l_features]))
        ex = np.array(list(set(ex).union(l_features)))
        
        machine = svm.NuSVC(nu=.25,
            kernel='linear',
            verbose=False,
            probability=True)
        machine.fit(my_X[train,][:,l_features], y[train])
        l2_learners.append(machine)

        p_y0 = sum(y[train] == -1) / len(train)
        p_y1 = sum(y[train] ==  1) / len(train)
    for k in range(4):
        print(roc_auc_score(y[train],
                get_decision_values(l2_learners, ls_features, tmpX[train,],
                    [p_y0, p_y1], k)),
              roc_auc_score(y[test],
                get_decision_values(l2_learners, ls_features, tmpX[test,],
                    [p_y0, p_y1], k)))
    
        score_l1 = roc_auc_score(y[train],
            learner.decision_function(ex_X[train,]))
        score_l2 = roc_auc_score(y[train],
                                 machine.decision_function(
                                     ex_X[train,][:,learner.coef_[0] != 0]))
        print('AUC over signed distance from hyperplane - l1:', score_l1, 'l2:', score_l2)
        score_l1 = precision_recall_fscore_support(y[train],
            learner.predict(ex_X[train,]), average='micro')[2]
        score_l2 = precision_recall_fscore_support(y[train],
                                 machine.predict(
                                     ex_X[train,][:,learner.coef_[0] != 0]),
                                     average=None)
        print('F-score over prediction label -            l1:', score_l1, 'l2:', score_l2)


    in_cvs = cv.StratifiedShuffleSplit(y[train], n_iter = 20, test_size = 0.2)
    scores = np.zeros(tmpX.shape[1])
    for i_train, i_test in in_cvs:
        machine = svm.NuSVC(nu=.25,
            kernel='linear',
            verbose=False,
            probability=False)

        machine.fit(tmpX[train,][i_train,], y[train][i_train])
        density = gaussian_kde(abs(machine.coef_))
        xs = np.linspace(0, max(max(abs(machine.coef_))), 200)
        density.covariance_factor = lambda : .25
        density._compute_covariance()
        #find threshold for top X% of area under the density curve
        low =  0
        high = max(max(machine.coef_))
        target_top = 0.001
        while(True):
            mid = (low + high) / 2
            mid_v = density.integrate_box_1d(mid, 1e4)
            if (abs(mid_v - target_top) < 1e-4):
                break
            if (mid_v > target_top):
                low = mid
            elif (mid_v < target_top):
                high = mid

        indices = (abs(machine.coef_) > mid).reshape(-1)
        #print(mid, mid_v, sum(indices))
        scores[indices] = scores[indices] + 1

    machine = svm.NuSVC(nu=.25,
        kernel='linear',
        verbose=False,
        probability=False)

    machine.fit(tmpX[train,][:,scores>10], y[train])
    tr_score.append(roc_auc_score(y[train],
        machine.decision_function(tmpX[train,][:,scores>10])))
    te_score.append(roc_auc_score(y[test],
        machine.decision_function(tmpX[test,][:,scores>10])))

    density = gaussian_kde(scores[scores > 10])
    xs = np.linspace(0, max(scores), 200)
    density.covariance_factor = lambda : .25
    density._compute_covariance()
    plt.plot(xs, density(xs))
    plt.show()
    plt.figure()
    fig = plt.gcf()
    fig.set_size_inches(7,7)
    plt.xlim([0,100])
    plt.axvline(mid, c='g')
    plt.savefig('tmp/density-%s-%s-%d.eps' %
                        (data, target, regularizer_index // 2), dpi=100)
    plt.close()



machine.fit(tmpX, y)
    
density = gaussian_kde(abs(machine.coef_))
xs = np.linspace(0, np.max(abs(machine.coef_)), 200)
density.covariance_factor = lambda : .25
density._compute_covariance()

low =  0
high = max(max(machine.coef_))
target_top = 0.01
while(True):
    mid = (low + high) / 2
    mid_v = density.integrate_box_1d(mid, 1e4)
    if (abs(mid_v - target_top) < 1e-4):
        break
    if (mid_v > target_top):
        low = mid
    elif (mid_v < target_top):
        high = mid

print(np.sum(abs(machine.coef_) > mid))
plt.axvline(mid, c='g')
plt.plot(xs, density(xs))
plt.show()
