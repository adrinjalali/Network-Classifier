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
import sklearn.metrics
from sklearn.grid_search import GridSearchCV
import sklearn.ensemble
import sklearn.tree
from collections import defaultdict
import time
from joblib import Parallel, delayed, logger
import pickle
import uuid
import re
import Raccoon.core.raccoon


from constants import *;
from misc import *
from rat import *

def add_to_scores(params):
    current = all_scores
    for item in params:
        if not (item in current):
            current[item] = dict()
        current = current[item]

def log(msg=''):
    d = list(filter(None, working_dir.split('/')))
    print('%s\t%s\tcv:%d\t%s' % (d[-2], d[-1], cv_index, msg), file=sys.stderr, flush=True)
    
if __name__ == '__main__':
    print('hi', file=sys.stderr);

    working_dir = ''
    #working_dir = '/scratch/TL/pool0/ajalali/ratboost/data_2015_07_05/Data/TCGA-LGG/tumor_status'
    #input_dir = '/scratch/TL/pool0/ajalali/ratboost/shared/Data/TCGA-LGG/tumor_status'
    method = ''
    #method = 'ratboost_linear_svc'
    cv_index = -1
    #cv_index = 5
    cpu_count = 40
    regularizer_index = None
    batch_based_cv = False
    for i in range(len(sys.argv)):
        print(sys.argv[i], file=sys.stderr)
        if (sys.argv[i] == '--input-dir'):
            input_dir = sys.argv[i + 1]
        if (sys.argv[i] == '--working-dir'):
            working_dir = sys.argv[i + 1]
        if (sys.argv[i] == '--method'):
            method = sys.argv[i + 1]
        if (sys.argv[i] == '--cv-index'):
            cv_index = int(sys.argv[i + 1]) - 1
        if (sys.argv[i] == '--regularizer-index'):
            regularizer_index = int(sys.argv[i + 1])
        if (sys.argv[i] == '--batch-based'):
            batch_based_cv = True
        if (sys.argv[i] == '--cpu-count'):
            cpu_count = int(sys.argv[i + 1])
    

    print(working_dir, method, cv_index, regularizer_index, cpu_count
          , file=sys.stderr)
    try:
        os.makedirs(working_dir + '/results', mode=0o750, exist_ok=True)
    except Exception as e:
        print(e, file=sys.stderr)

    try:
        os.makedirs(working_dir + '/models', mode=0o750, exist_ok=True)
    except Exception as e:
        print(e, file=sys.stderr)

    log("loading data...")

    log("trying an old input format...")
    data_loaded = False
    try:
        data_file = np.load(input_dir + '/npdata.npz')
        X = data_file['tmpX']
        X_prime = data_file['X_prime']
        y = data_file['y']
        sample_annotation = data_file['sample_annotation']
        feature_annotation = data_file['feature_annotation']
        g = gt.load_graph(input_dir + '/graph.xml.gz')
        cvs = pickle.load(open(input_dir + '/cvs.dmp', 'rb'))
        data_loaded = True
    except Exception as e:
        log(e)

    if (not data_loaded):
        log("trying another input format...")
        try:
            data_file = np.load(input_dir + '/data.npz');
            X = data_file['X']
            X_prime = data_file['X_prime']
            y = data_file['y']
            sample_annotation = data_file['patient_annot']
            data_file = np.load(input_dir + '/../genes.npz')
            feature_annotation = data_file['genes']
            g = gt.load_graph(input_dir + '/../graph.xml.gz')
            if (batch_based_cv):
                cvs = pickle.load(open(input_dir + '/batch_cvs.dmp', 'rb'))
            else:
                cvs = pickle.load(open(input_dir + '/normal_cvs.dmp', 'rb'))
            data_loaded = True
        except Exception as e:
            log(e)

    if (cv_index > len(cvs) - 1):
        log("requested cv (%d) doesn't exist (len(cvs) = %d)" % (cv_index,
                                                                   len(cvs)))
        sys.exit(1)
            
    #choosing only one cross-validation fold
    tmp = list()
    tmp.append((cvs[cv_index]))
    cvs = tmp

    Xtrain = X[cvs[0][0],]
    X_prime_train = X_prime[cvs[0][0],]
    ytrain = y[cvs[0][0],]
    Xtest = X[cvs[0][1],]
    X_prime_test = X_prime[cvs[0][1],]
    ytest = y[cvs[0][1],]

    #if (np.unique(ytest).shape[0] < 2):
    #    log('ytest has only one value:%s' % (ytest))
    #    log('exiting')
    #    sys.exit(1)
    
    max_learner_count = 25
    rat_scores = dict()
    all_scores = defaultdict(list)
    score_dump_file = working_dir + '/results/%s-%d-%s.dmp' % \
                (method, cv_index, str(uuid.uuid1()))

    if (method == 'all' or method == 'others'):

        log('svms')
        for nu in np.arange(7) * 0.1 + 0.05:
            try:
                machine = svm.NuSVC(nu=nu,
                                kernel='linear',
                                verbose=False,
                                probability=False)
                machine.fit(Xtrain, ytrain)
                scores = sklearn.metrics.average_precision_score(ytest, machine.decision_function(Xtest))
                #scores = roc_auc_score(ytest, machine.decision_function(Xtest))
                log('lsvm\tnu:%g\t%s' % (nu, scores))
                this_method = 'SVM, linear kernel'
                add_to_scores([this_method, ('nu', nu)])
                all_scores[this_method][('nu', nu)] = [scores]
            except ValueError as e:
                log('nu:%g\t%s' % (nu, e))
            

            try:
                machine = svm.NuSVC(nu=nu,
                                kernel='rbf',
                                verbose=False,
                                probability=False) 
                machine.fit(Xtrain, ytrain)
                scores = sklearn.metrics.average_precision_score(ytest, machine.decision_function(Xtest))
                #scores = roc_auc_score(ytest, machine.decision_function(Xtest))
                log('gsvm\tnu:%g\t%s' % (nu, scores))
                this_method = 'SVM, RBF kernel'
                add_to_scores([this_method, ('nu', nu)])
                all_scores[this_method][('nu', nu)] = [scores]
            except ValueError as e:
                log('nu:%g\t%s' % (nu, e))
            
        for nu in np.arange(7) * 0.1 + 0.05:
            try:
                machine = svm.NuSVC(nu=nu,
                                kernel='linear',
                                verbose=False,
                                probability=False)
                machine.fit(Xtrain, ytrain)
                scores = sklearn.metrics.average_precision_score(ytest, machine.decision_function(Xtest))
                #scores = roc_auc_score(ytest, machine.decision_function(Xtest))
                log('lsvm`\tnu:%g\t%s' % (nu, scores))
                this_method = 'SVM, linear kernel'
                add_to_scores([this_method, ('nu', nu)])
                all_scores[this_method][('nu', nu)] = [scores]
            except ValueError as e:
                log('nu:%g\t%s' % (nu, e))
            
                
        log('gbc')
        for mf in np.arange(3) * 5 + 5:
            for md in np.arange(3) + 1:
                for ne in [5, 20, 50, 100, 200]:
                    machine = sklearn.ensemble.GradientBoostingClassifier(
                        max_features = mf,
                        max_depth = md,
                        n_estimators = ne)
                    machine.fit(Xtrain, ytrain)
                    scores = sklearn.metrics.average_precision_score(ytest, machine.decision_function(Xtest))
                    #scores = roc_auc_score(ytest, machine.decision_function(Xtest))
                    log('gbc\tmf:%d\tmd:%d\tne:%d\t%s' % (mf, md, ne, scores))
                    this_method = 'Gradient Boosting Classifier'
                    add_to_scores([this_method, ('max_features', mf),
                           ('max_depth', md), ('N', ne)])
                    all_scores[this_method][('max_features', mf)] \
                        [('max_depth', md)][('N', ne)] = [scores]

        log('adaboost')
        for md in np.arange(3) + 1:
            for ne in [5, 20, 50, 100, 200]:
                machine = sklearn.ensemble.AdaBoostClassifier(
                    sklearn.tree.DecisionTreeClassifier(max_depth=md),
                    algorithm = "SAMME.R",
                    n_estimators = ne)
                machine.fit(Xtrain, ytrain)
                scores = sklearn.metrics.average_precision_score(ytest, machine.decision_function(Xtest))
                #scores = roc_auc_score(ytest, machine.decision_function(Xtest))
                log('adb\tmd:%d\tne:%d\t%s' % (md, ne, scores))
                this_method = 'Adaboost'
                add_to_scores([this_method, ('max_depth', md),
                               ('N', ne)])
                all_scores[this_method][('max_depth', md)][('N', ne)] = [scores]

        log()
        print_scores(all_scores)

        dump_scores(score_dump_file, all_scores)
        
    if (method == 'all' or method == 'rat'):
        log('ratboost')
        max_learner_count = 15

        name_params = [('RatBoost No First Layer Confidence',
                        {'firstLayerConfidence':False,
                         'secondLayerConfidence':True}),
                       ('RatBoost No Second Layer Confidence',
                        {'firstLayerConfidence':True,
                         'secondLayerConfidence':False}),
                       ('RatBoost No Confidence',
                        {'firstLayerConfidence':False,
                         'secondLayerConfidence':False}),
                       ('RatBoost Both Confidences',
                        {'firstLayerConfidence':True,
                         'secondLayerConfidence':True})]
                        
        for ri in np.hstack((1, np.array(list(range(10))) * 2)):
            log('------------ ri:%g' % (ri))
            rat = Rat(learner_count = max_learner_count,
                learner_type = 'linear svc',
                regularizer_index = ri,
                n_jobs = cpu_count)

            rat.fit(Xtrain, ytrain)
            
            log('scores')
            for item in name_params:
                this_method = item[0]
                conf_params = item[1]
                add_to_scores([this_method, ('regularizer_index', ri)])
                
                #rat_models.append(rat)
                test_decision_values = rat.decision_function(Xtest,
                                                             return_iterative = True,
                                                             **conf_params)
                train_decision_values = rat.decision_function(Xtrain,
                                                              return_iterative = True,
                                                              **conf_params)
                for i in range(len(test_decision_values)):
                    scores = sklearn.metrics.average_precision_score(ytest, test_decision_values[i])
                    #scores = roc_auc_score(ytest, test_decision_values[i])
                    log('trn:%g' % (roc_auc_score(ytrain, train_decision_values[i])))
                    log('tst:\t%g' % (scores))
    
                    all_scores[this_method][('regularizer_index', ri)]\
                        [('N', i)] = [scores]
            
            log()
            print_scores(all_scores)
            
            dump_scores(score_dump_file, all_scores)
    
            model_structure = [{f : (l.getClassifierFeatureWeights()[f],
                    l._FCEs[f].getFeatures())
                    for f in l.getClassifierFeatures()}
                    for l in rat.learners]
            model_dump_file = working_dir + '/models/%s-%d-rat-%d-%s.dmp' % \
                (method, cv_index, ri, str(uuid.uuid1()))
            pickle.dump(model_structure,
                open(model_dump_file, 'wb'))

    if (method == 'all' or method == 'raccoon'):
        log('raccoon')
        import sklearn.svm
        model = Raccoon.core.raccoon.Raccoon(verbose=3, logger=log)
        model.fit(Xtrain, ytrain)
        predictor = sklearn.svm.SVC()
        param_dist = {'C': pow(2.0, np.arange(-10, 11)), 'gamma': pow(2.0, np.arange(-10, 11)),
                      'kernel': ['linear', 'rbf']}
        #tmpt = model.predict(Xtrain, model=predictor, param_dist=param_dist)
        test_results = model.predict(Xtest, model=predictor, param_dist=param_dist)
        scores = sklearn.metrics.average_precision_score(ytest, [k['decision_function'][0] for k in test_results])
        
        log('raccoon\t%s' % (scores))
        this_method = 'Raccoon'
        all_scores[this_method] = [scores]

        log()
        print_scores(all_scores)

        dump_scores(score_dump_file, all_scores)
        
    print('bye', file=sys.stderr)

