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


from constants import *;
from misc import *
import read_nordlund1 
import read_nordlund2
import read_vantveer
import read_tcga_brca
import read_tcga_laml
import read_tcga_laml_geneexpression
from rat import *

if __name__ == '__main__':
    def _f(learner_type, regularizer_index = None):
        print("%s" %(learner_type), file=sys.stderr)
        rat = Rat(learner_count = 1,
                  learner_type = learner_type,
                  regularizer_index = regularizer_index,
                  n_jobs = 1)
        scores = rat_cross_val_score(
            rat, tmpX, y,
            cv=cvs,
            scoring = 'roc_auc',
            n_jobs=cpu_count,
            verbose=1,
            max_learner_count = max_learner_count)

        scores = np.array(scores)
        commulative_score = dict()
        for i in range(max_learner_count):
            commulative_score[('N',i + 1)] = scores[:,i,0]

        this_method = 'RatBoost'
        rat_scores[this_method] = dict()
        rat_scores[this_method][('learner_type', learner_type)] = dict()
        rat_scores[this_method][('learner_type', learner_type)]\
          [('regularizer_index', regularizer_index)] = commulative_score
        return(scores)

    def _f_alltypes():
        _f("logistic regression")
        print_log(all_scores, rat_scores)
        _f("linear svc")
        print_log(all_scores, rat_scores)
        _f("nu svc")
        print_log(all_scores, rat_scores)

    print('hi', file=sys.stderr);

    working_dir = ''
    #working_dir = '/scratch/TL/pool0/ajalali/ratboost/data_3/TCGA-LAML/vital_status/'
    method = ''
    #method = 'ratboost_linear_svc'
    cv_index = -1
    #cv_index = 10
    regularizer_index = None
    for i in range(len(sys.argv)):
        print(sys.argv[i], file=sys.stderr)
        if (sys.argv[i] == '--working-dir'):
            working_dir = sys.argv[i + 1]
        if (sys.argv[i] == '--method'):
            method = sys.argv[i + 1]
        if (sys.argv[i] == '--cv-index'):
            cv_index = int(sys.argv[i + 1]) - 1
        if (sys.argv[i] == '--regularizer-index'):
            regularizer_index = int(sys.argv[i + 1])

    print(working_dir, method, cv_index, regularizer_index, file=sys.stderr)

    print("loading data...", file=sys.stderr)

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
    
    cpu_count = 1
    max_learner_count = 3
    rat_scores = dict()
    all_scores = defaultdict(list)

    if (method == 'others'):

        machine = svm.NuSVC(nu=0.25,
                            kernel='linear',
                            verbose=False,
                            probability=False)
        scores = cv.cross_val_score(
            machine, tmpX, y,
            cv = cvs,
            scoring = 'roc_auc',
            n_jobs = cpu_count,
            verbose=1)
        this_method = 'SVM, linear kernel'
        all_scores[this_method] = dict()
        all_scores[this_method][('nu', 0.25)] = scores

        machine = svm.NuSVC(nu=0.25,
                            kernel='rbf',
                            verbose=False,
                            probability=False)
        scores = cv.cross_val_score(
            machine, tmpX, y,
            cv = cvs,
            scoring = 'roc_auc',
            n_jobs = cpu_count,
            verbose=1)
        this_method = 'SVM, RBF kernel'
        all_scores[this_method] = dict()
        all_scores[this_method][('nu', 0.25)] = scores

        machine = svm.NuSVC(nu=0.25,
                            kernel='linear',
                            verbose=False,
                            probability=False)
        scores = cv.cross_val_score(
            machine, X_prime, y,
            cv = cvs,
            scoring = 'roc_auc',
            n_jobs = cpu_count,
            verbose=1)
        this_method = 'SVM, linear kernel, transformed'
        all_scores[this_method] = dict()
        all_scores[this_method][('nu', 0.25)] = scores

        machine = sklearn.ensemble.GradientBoostingClassifier(max_features = 5,
                                                              max_depth = 2,
                                                              n_estimators = 200)
        scores = cv.cross_val_score(
            machine, tmpX, y,
            cv = cvs,
            scoring = 'roc_auc',
            n_jobs = cpu_count,
            verbose=1)
        this_method = 'Gradient Boosting Classifier'
        all_scores[this_method] = dict()
        all_scores[this_method][('N', 200)] = scores

        machine = sklearn.ensemble.AdaBoostClassifier(
            sklearn.tree.DecisionTreeClassifier(max_depth=1),
            algorithm = "SAMME",
            n_estimators = 100)
        scores = cv.cross_val_score(
            machine, tmpX, y,
            cv = cvs,
            scoring = 'roc_auc',
            n_jobs = cpu_count,
            verbose=1)
        this_method = 'Adaboost'
        all_scores[this_method] = dict()
        all_scores[this_method][('N', 100)] = scores

        print_log(all_scores, rat_scores)

        dump_scores(working_dir + '/results/%s-%d-others-%s.dmp' % \
                    (method, cv_index, str(uuid.uuid1())),
                    all_scores)
        
    elif (method == 'ratboost_logistic_regression'):
        _f("logistic regression", regularizer_index)
        print_log(all_scores, rat_scores)
        dump_scores(working_dir + '/results/%s-%d-rat-%d-%s.dmp' % \
                    (method, cv_index, regularizer_index, str(uuid.uuid1())),
                    rat_scores)
    elif (method == 'ratboost_linear_svc'):
        _f("linear svc", regularizer_index)
        print_log(all_scores, rat_scores)
        dump_scores(working_dir + '/results/%s-%d-rat-%d-%s.dmp' % \
                    (method, cv_index, regularizer_index, str(uuid.uuid1())),
                    rat_scores)
    elif (method == 'ratboost_nu_svc'):
        _f("nu svc", regularizer_index)
        print_log(all_scores, rat_scores)
        dump_scores(working_dir + '/results/%s-%d-rat-%d-%s.dmp' % \
                    (method, cv_index, regularizer_index, str(uuid.uuid1())),
                    rat_scores)

    
    print('bye', file=sys.stderr)

