import __future__

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



from constants import *;
from misc import *
import read_nordlund1 
import read_nordlund2
import read_vantveer
import read_tcga_brca
import read_tcga_laml
from rat import *

if __name__ == '__main__':
    def reload_rat():
        with open("./rat.py") as f:
            code = compile(f.read(), "rat.py", 'exec')
            exec(code)

    def print_log(all_scores, rat_scores):
        print('=========')
        def statstr(v):
            return("%.3lg +/- %.3lg" % (np.mean(v), 2 * np.std(v)))
        for key, value in all_scores.items():
            print("test auc %s: " % (key), statstr(value))
        for key, value in rat_scores.items():
            print("test auc %s:" % (key))
            for key2 in sorted(value.keys()):
                print("\t%s: " % (key2), statstr(value[key2]))

    def dump_scores(file_name, scores):
        import pickle
        pickle.dump(scores, open(file_name, "wb"))

    def _fit_and_score(estimator, X, y, scorer, train, test, verbose, parameters,
                       fit_params, max_learner_count, return_train_score=False,
                       return_parameters=False):
        if verbose > 1:
            if parameters is None:
                msg = "no parameters to be set"
            else:
                msg = '%s' % (', '.join('%s=%s' % (k, v)
                                        for k, v in parameters.items()))
            print("[CV] %s %s" % (msg, (64 - len(msg)) * '.'))
        
        # Adjust lenght of sample weights
        n_samples = len(X)
        fit_params = fit_params if fit_params is not None else {}
        fit_params = dict([(k, np.asarray(v)[train]
                            if hasattr(v, '__len__') and len(v) == n_samples else v)
                           for k, v in fit_params.items()])
        
        if parameters is not None:
            estimator.set_params(**parameters)
            
        X_train, y_train = sklearn.cross_validation._safe_split(
            estimator, X, y, train)
        X_test, y_test = sklearn.cross_validation._safe_split(
            estimator, X, y, test, train)
        result = list()
        from_scratch = True
        for i in range(max_learner_count):
            start_time = time.time()
        
            estimator.fit(X_train, y_train, from_scratch = from_scratch)
            test_score = sklearn.cross_validation._score(
                estimator, X_test, y_test, scorer)
            if return_train_score:
                train_score = _score(estimator, X_train, y_train, scorer)
            ret = [train_score] if return_train_score else []

            scoring_time = time.time() - start_time

            ret.extend([test_score, len(X_test), scoring_time])
            if return_parameters:
                ret.append(parameters)
            result.append(ret)
            from_scratch = False
            
            
            if verbose > 2:
                msg += ", score=%f" % test_score
            if verbose > 1:
                end_msg = "%s -%s" % (msg, logger.short_format_time(scoring_time))
                print("[CV] %s %s" % ((64 - len(end_msg)) * '.', end_msg))
            
        return result
        
    def rat_cross_val_score(estimator, X, y=None, scoring=None, cv=None, n_jobs=1,
                        verbose=0, fit_params=None, score_func=None,
                        pre_dispatch='2*n_jobs', max_learner_count = 2):
        X, y = sklearn.utils.check_arrays(X, y, sparse_format='csr', allow_lists=True)
        cv = sklearn.cross_validation._check_cv(cv,
                                                X, y,
                                                classifier=sklearn.base.is_classifier(estimator))
        scorer = sklearn.cross_validation.check_scoring(
            estimator, score_func=score_func, scoring=scoring)
        # We clone the estimator to make sure that all the folds are
        # independent, and that it is pickle-able.

        jobs = list(dict())

        fit_params = fit_params if fit_params is not None else {}
        parallel = Parallel(n_jobs=n_jobs, verbose=verbose,
                            pre_dispatch=pre_dispatch)

        fit_params['from_scratch'] = True
        collected_scores = dict()
        scorer = sklearn.metrics.scorer.get_scorer(scoring)
        scorer = sklearn.metrics.scorer.get_scorer(scoring)
        scores = parallel(
            delayed(_fit_and_score)(
                estimator,
                X, y, scorer,
                train, test,
                verbose, None, fit_params,
                max_learner_count = max_learner_count)
            for train, test in cv)
            
        return (scores)

    def _f(learner_type):
        print("%s" %(learner_type))
        rat = Rat(learner_count = 1,
                  learner_type = learner_type,
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
            commulative_score[i + 1] = scores[:,i,0]

        rat_scores[learner_type] = commulative_score
        return(scores)

    def _f_alltypes():
        _f("logistic regression")
        print_log(all_scores, rat_scores)
        _f("linear svc")
        print_log(all_scores, rat_scores)
        _f("nu svc")
        print_log(all_scores, rat_scores)
        
    print('hi');

    ''' load nordlund T-ALL vs BCP-ALL '''
    #(tmpX, y, g, sample_annotation, feature_annotation) = read_nordlund1.load_data()
    ''' load  nordlund subtypes A vs subtypes B '''
    #(tmpX, y, g,
    # sample_annotation,
    # feature_annotation) = read_nordlund2.load_data('HeH', 't(12;21)')
    ''' load vantveer data poor vs good prognosis '''
    #(tmpX, y, g, sample_annotation, feature_annotation) = read_vantveer.load_data()
    ''' load TCGA BRCA data '''
    (tmpX, y, g,
     sample_annotation,
     feature_annotation) = read_tcga_brca.load_data('stage')
    ''' load TCGA LAML data '''
    #(tmpX, y, g,
    # sample_annotation,
    # feature_annotation) = read_tcga_laml.load_data('vital_status')

    print("calculating L and transformation of the data...")
    B = gt.spectral.laplacian(g)
    M = np.identity(B.shape[0]) + Globals.beta * B
    M_inv = np.linalg.inv(M)
    L = np.linalg.cholesky(M_inv)
    X_prime = tmpX.dot(L)

    print("cross-validation...")

    cpu_count = 30
    max_learner_count = 40
    fold_count = 100
    rat_scores = dict()
    all_scores = defaultdict(list)
    cvs = cv.StratifiedShuffleSplit(y, n_iter = fold_count, test_size = 0.2)
    
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
    all_scores['original, nuSVM(0.25), linear'].append(scores)

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
    all_scores['original, nuSVM(0.25), rbf'].append(scores)

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
    all_scores['transformed, nuSVM(0.25), linear'].append(scores)

    machine = sklearn.ensemble.GradientBoostingClassifier(max_features = 5,
                                                          max_depth = 2,
                                                          n_estimators = 200)
    scores = cv.cross_val_score(
        machine, tmpX, y,
        cv = cvs,
        scoring = 'roc_auc',
        n_jobs = cpu_count,
        verbose=1)
    all_scores['gradientboostingclassifier'].append(scores)

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
    all_scores['adaboost'].append(scores)
    
    print_log(all_scores, rat_scores)    

    _f_alltypes()
    print('bye')


'''
exec(open("./rat.py").read())
a = LogisticRegressionClassifier(n_jobs = 30,
                                 excluded_features=list([346,715,785]),
                                 feature_confidence_estimator=PredictBasedFCE(),
                                 second_layer_feature_count = 5,
                                 C = 0.2)
a.fit(tmpX, y)



with open("./rat.py") as f:
    code = compile(f.read(), "rat.py", 'exec')
    exec(code)
a = Rat(learner_count = 2,
        learner_type = 'linear svc',
        C = 0.3,
        n_jobs = 30)
a.fit(tmpX[:60,], y[:60])
a.predict(tmpX[1,])
a.predict(tmpX[:3,])
a.score(tmpX, y)
scores = cv.cross_val_score(
    a, tmpX, y,
    cv=5,
scoring = 'roc_auc',
    n_jobs = 1,
    verbose=1)
print(np.average(scores))

machine = svm.NuSVC(nu=0.25,
    kernel='linear',
    verbose=False,
    probability=False)
machine.fit(tmpX[:60,], y[:60])
threshold = np.min(np.abs(machine.coef_)) + (np.max(np.abs(machine.coef_)) - np.min(np.abs(machine.coef_))) * 0.8
np.arange(machine.coef_.shape[1])[(abs(machine.coef_) > threshold).flatten()]


cs = sklearn.svm.l1_min_c(tmpX, y, loss='l2') * np.logspace(0,2)
start = datetime.now()
#clf = LogisticRegression(C=1.0, penalty='l1', tol=1e-6)
clf = sklearn.svm.LinearSVC(C = 1.0, penalty='l1', dual=False)
coefs_ = []
for c in cs:
    clf.set_params(C=c)
    clf.fit(tmpX, y)
    coefs_.append(clf.coef_.ravel().copy())
print("This took ", datetime.now() - start)

coefs_ = np.array(coefs_)
pl.plot(np.log10(cs), coefs_)
ymin, ymax = pl.ylim()
pl.xlabel('log(C)')
pl.ylabel('Coefficients')
pl.title('Logistic Regression Path')
pl.axis('tight')
pl.show()

'''


