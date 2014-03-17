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
import read_tcga_brca
from rat import *

if __name__ == '__main__':
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
     feature_annotation) = read_tcga_brca.load_data('N')

    print("calculating L and transformation of the data...")
    B = gt.spectral.laplacian(g)
    M = np.identity(B.shape[0]) + Globals.beta * B
    M_inv = np.linalg.inv(M)
    L = np.linalg.cholesky(M_inv)
    X_prime = tmpX.dot(L)

    print("cross-validation...")

    def _f(l, c, learner_type):
        print("%d %g %s" %(l,c, learner_type))
        rat = Rat(learner_count = l,
                  learner_type = learner_type,
                  C = c,
                  n_jobs = 1)
        scores = cv.cross_val_score(
            rat, tmpX, y,
            cv=cvs,
            scoring = 'roc_auc',
            n_jobs=30,
            verbose=1)
        all_scores["%d, %g, %s" % (l, c, learner_type)] = scores
        return(scores)

    def _f_alltypes(l, c):
        _f(l, c, "logistic regression")
        _f(l, c, "linear svc")
        _f(l, c, "nu svc")
        
    def log(all_scores):
        print('=========')
        def statstr(v):
            return("%.3lg +/- %.3lg" % (np.mean(v), 2 * np.std(v)))
        for key, value in all_scores.items():
            print("test  auc %s: " % (key), statstr(value))

    def dump_scores(file_name, scores):
        import pickle
        pickle.dump(scores, open(file_name, "wb"))

    all_scores = defaultdict(list)
    cvs = cv.StratifiedShuffleSplit(y, n_iter = 100, test_size = 0.2)
    
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
    
    _f_alltypes(1, 0.3)
    log(all_scores)
    
    _f_alltypes(3, 0.3)
    log(all_scores)
    
    _f_alltypes(5, 0.3)
    log(all_scores)
    
    _f_alltypes(7, 0.3)
    log(all_scores)
    
    _f_alltypes(10, 0.3)
    log(all_scores)
    
    _f_alltypes(15, 0.3)
    log(all_scores)
    
    _f_alltypes(20, 0.3)
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

boz = sklearn.svm.NuSVC(nu = 0.25, kernel='linear', probability=True)
boz.fit(tmpX, y)
boz.decision_function(tmpX[0,])
boz.coef_[boz.coef_ != 0]
boz.coef_[abs(boz.coef_) > 0.90 * np.max(abs(boz.coef_))]

cs = sklearn.svm.l1_min_c(tmpX, y, loss='l2') * np.logspace(0,2)

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
        C = 0.2,
        n_jobs = 30)
a.fit(tmpX[:200,], y[:200])
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


