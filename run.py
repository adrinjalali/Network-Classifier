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

if __name__ == '__main__':
    def _f(learner_type, regularizer_index = None):
        print("%s" %(learner_type), file=sys.stderr)
        rat = Rat(learner_count = 1,
                  learner_type = learner_type,
                  regularizer_index = regularizer_index,
                  n_jobs = 1)
        result = rat_cross_val_score(
            rat, tmpX, y,
            cv=cvs,
            scoring = 'roc_auc',
            n_jobs=cpu_count,
            verbose=1,
            max_learner_count = max_learner_count)

        if (len(result) == 1):
            models = result[0][1]
        else:
            models = [m for s, m in result]
        
        scores = [s for s, m in result]
        scores = np.array(scores)
        commulative_score = dict()
        for i in range(max_learner_count):
            commulative_score[('N',i + 1)] = scores[:,i,0]

        this_method = 'RatBoost'
        rat_scores[this_method] = dict()
        rat_scores[this_method][('learner_type', learner_type)] = dict()
        rat_scores[this_method][('learner_type', learner_type)]\
          [('regularizer_index', regularizer_index)] = commulative_score
        return((scores, models))

    def _f_alltypes():
        _f("logistic regression")
        print_log(all_scores, rat_scores)
        _f("linear svc")
        print_log(all_scores, rat_scores)
        _f("nu svc")
        print_log(all_scores, rat_scores)

    def add_to_scores(params):
        current = all_scores
        for item in params:
            if not (item in current):
                current[item] = dict()
            current = current[item]


    print('hi', file=sys.stderr);

    working_dir = ''
    #working_dir = '/scratch/TL/pool0/ajalali/ratboost/data_8/TCGA-BRCA/T'
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
    max_learner_count = 30
    rat_scores = dict()
    all_scores = defaultdict(list)

    if (method == 'others'):

        for nu in np.arange(7) * 0.1 + 0.05:
            try:
                machine = svm.NuSVC(nu=nu,
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
                add_to_scores([this_method, ('nu', nu)])
                all_scores[this_method][('nu', nu)] = scores
            except ValueError as e:
                print(nu, e)

            try:
                machine = svm.NuSVC(nu=nu,
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
                add_to_scores([this_method, ('nu', nu)])
                all_scores[this_method][('nu', nu)] = scores
            except ValueError as e:
                print(nu, e)

            try:
                machine = svm.NuSVC(nu=nu,
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
                add_to_scores([this_method, ('nu', nu)])
                all_scores[this_method][('nu', nu)] = scores
            except ValueError as e:
                print(nu, e)
                
        for mf in np.arange(3) * 5 + 5:
            for md in np.arange(3) + 1:
                for ne in [5, 20, 50, 100, 200]:
                    machine = sklearn.ensemble.GradientBoostingClassifier(
                        max_features = mf,
                        max_depth = md,
                        n_estimators = ne)
                    scores = cv.cross_val_score(
                        machine, tmpX, y,
                        cv = cvs,
                        scoring = 'roc_auc',
                        n_jobs = cpu_count,
                        verbose=1)
                    this_method = 'Gradient Boosting Classifier'
                    add_to_scores([this_method, ('max_features', mf),
                           ('max_depth', md), ('N', ne)])
                    all_scores[this_method][('max_features', mf)] \
                        [('max_depth', md)][('N', ne)] = scores

        for md in np.arange(3) + 1:
            for ne in [5, 20, 50, 100, 200]:
                machine = sklearn.ensemble.AdaBoostClassifier(
                    sklearn.tree.DecisionTreeClassifier(max_depth=2),
                    algorithm = "SAMME.R",
                    n_estimators = ne)
                scores = cv.cross_val_score(
                    machine, tmpX, y,
                    cv = cvs,
                    scoring = 'roc_auc',
                    n_jobs = cpu_count,
                    verbose=1)
                this_method = 'Adaboost'
                add_to_scores([this_method, ('max_depth', md),
                               ('N', ne)])
                all_scores[this_method][('max_depth', md)][('N', ne)] = scores

        print_log(all_scores, rat_scores)

        dump_scores(working_dir + '/results/%s-%d-others-%s.dmp' % \
                    (method, cv_index, str(uuid.uuid1())),
                    all_scores)
        
    elif (method.startswith('ratboost')):
        '''
        method should be one of:
        ratboost_logistic_regression
        ratboost_linear_svc
        ratboost_nu_svc
        ''' 
        rat_method = ' '.join(re.split('_', method)[1:])
        scores, model = _f(rat_method, regularizer_index)
        print_log(all_scores, rat_scores)
        dump_scores(working_dir + '/results/%s-%d-rat-%d-%s.dmp' % \
                    (method, cv_index, regularizer_index, str(uuid.uuid1())),
                    rat_scores)
        model_structure = [{f : (l.getClassifierFeatureWeights()[f],
                                 l._FCEs[f].getFeatures())
                                 for f in l.getClassifierFeatures()}
                                 for l in model.learners]
        pickle.dump(model_structure,
                    open(working_dir + '/models/%s-%d-rat-%d-%s.dmp' % \
                    (method, cv_index, regularizer_index, str(uuid.uuid1())), 'wb'))

    
    print('bye', file=sys.stderr)

