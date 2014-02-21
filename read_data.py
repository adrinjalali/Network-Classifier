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



from constants import *;
from misc import *
import read_nordlund1 
import read_nordlund2
import read_vantveer
from rat import *

if __name__ == '__main__':
    print('hi');

    ''' load nordlund T-ALL vs BCP-ALL '''
    (tmpX, y, g, sample_annotation, feature_annotation) = read_nordlund1.load_data()
    ''' load  nordlund subtypes A vs subtypes B '''
    #(tmpX, y, g,
    # sample_annotation,
    # feature_annotation) = read_nordlund2.load_data('HeH', 't(12;21)')
    ''' load vantveer data poor vs good prognosis '''
    #(tmpX, y, g, sample_annotation, feature_annotation) = read_vantveer.load_data()

    print("calculating L and transformation of the data...")
    B = gt.spectral.laplacian(g)
    M = np.identity(B.shape[0]) + Globals.beta * B
    M_inv = np.linalg.inv(M)
    L = np.linalg.cholesky(M_inv)
    X_prime = tmpX.dot(L)

    print("cross-validation...")

    def f(l, c, s):
        rat = Rat(learner_count = l,
                  learner = LogisticRegressionClassifier(
                      C = c,
                      feature_confidence_estimator=PredictBasedFCE(
                          feature_count = s),
                      n_jobs = 30))
        scores = cv.cross_val_score(
            rat, tmpX, y,
            cv=cvs,
            scoring = 'roc_auc',
            n_jobs=1,
            verbose=1)
        return(scores)
        
    def log(all_scores):
        print('=========')
        def statstr(v):
            return("%.3lg +/- %.3lg" % (np.mean(v), 2 * np.std(v)))
        for key, value in all_scores.items():
            print("test  auc %s: " % (key), statstr(value))

    all_scores = defaultdict(list)
    for i in range(10):
        print("%%%%%% i : ", i)
        cvs = cv.StratifiedKFold(y, 5)

        machine = svm.NuSVC(nu=0.25,
                            kernel='linear',
                            verbose=False,
                            probability=False)
        scores = cv.cross_val_score(
            machine, tmpX, y,
            cv = cvs,
            scoring = 'roc_auc',
            n_jobs = 30,
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
            n_jobs = 30,
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
            n_jobs = 30,
            verbose=1)
        all_scores['transformed, nuSVM(0.25), linear'].append(scores)

        machine = sklearn.ensemble.GradientBoostingClassifier(max_features = 5,
                                                              max_depth = 2,
                                                              n_estimators = 200)
        scores = cv.cross_val_score(
            machine, tmpX, y,
            cv = cvs,
            scoring = 'roc_auc',
            n_jobs = 30,
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
            n_jobs = 30,
            verbose=1)
        all_scores['adaboost'].append(scores)


        log(all_scores)

        scores = f(1, 0.3, 5)
        all_scores['1,0.3,5'].append(scores)
        log(all_scores)

        scores = f(1, 0.4, 5)
        all_scores['1,0.4,5'].append(scores)
        log(all_scores)

        scores = f(1, 0.3, 10)
        all_scores['1,0.3,10'].append(scores)
        log(all_scores)

        scores = f(1, 0.4, 10)
        all_scores['1,0.4,10'].append(scores)
        log(all_scores)

        scores = f(5, 0.3, 5)
        all_scores['5,0.3,5'].append(scores)
        log(all_scores)

        scores = f(5, 0.3, 10)
        all_scores['5,0.3,10'].append(scores)
        log(all_scores)
        
        scores = f(10, 0.3, 5)
        all_scores['10,0.3,5'].append(scores)
        log(all_scores)
        
        scores = f(10, 0.3, 10)
        all_scores['10,0.3,10'].append(scores)
        log(all_scores)
        
        scores = f(5, 0.4, 5)
        all_scores['5,0.4,5'].append(scores)
        log(all_scores)
        
        scores = f(5, 0.4, 10)
        all_scores['5,0.4,10'].append(scores)
        log(all_scores)
        
        scores = f(10, 0.4, 5)
        all_scores['10,0.4,5'].append(scores)
        log(all_scores)
        
        scores = f(10, 0.4, 10)
        all_scores['10,0.4,10'].append(scores)
        log(all_scores)
        '''
        rat = Rat(learner_count = 10,
                  learner = LogisticRegressionClassifier(
                      C = 0.3,
                      feature_confidence_estimator=PredictBasedFCE(
                          feature_count = 5),
                      n_jobs = 1))
        learners = list()
        for c in [0.3, 0.4]:
            for s in [5, 10]:
                learners.append(LogisticRegressionClassifier(
                    C = c,
                    feature_confidence_estimator=PredictBasedFCE(feature_count=s),
                    n_jobs = 1))

        param_grid = {'learner_count': [5,10], 'learner': learners,
                      'overlapping_features':[False]}
        model = GridSearchCV(rat, param_grid = param_grid, scoring = 'roc_auc',
                             n_jobs = 30, refit = True, cv=5, verbose=1)
        scores = cv.cross_val_score(
            model, tmpX, y,
            cv=cvs,
            scoring = 'roc_auc',
            n_jobs=1,
            verbose=1)
        all_scores['gridsearchcv'].append(scores)
        log(all_scores)
        '''
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
a = Rat(learner_count = 10,
        learner = LogisticRegressionClassifier(
            C = 0.4,
            feature_confidence_estimator=PredictBasedFCE(
                feature_count = 10),
            n_jobs = 30))
a.fit(tmpX, y)
a.score(tmpX, y)
'''