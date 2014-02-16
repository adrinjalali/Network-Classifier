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
    #(tmpX, y, g, sample_annotation, feature_annotation) = read_nordlund1.load_data()
    ''' load  nordlund subtypes A vs subtypes B '''
    #(tmpX, y, g,
    # sample_annotation,
    # feature_annotation) = read_nordlund2.load_data('HeH', 't(12;21)')
    ''' load vantveer data poor vs good prognosis '''
    (tmpX, y, g, sample_annotation, feature_annotation) = read_vantveer.load_data()

    print("calculating L and transformation of the data...")
    B = gt.spectral.laplacian(g)
    M = np.identity(B.shape[0]) + Globals.beta * B
    M_inv = np.linalg.inv(M)
    L = np.linalg.cholesky(M_inv)
    X_prime = tmpX.dot(L)

    print("cross-validation...")
    cfolds = cv.StratifiedShuffleSplit(y, n_iter=Globals.cfold_count, test_size=0.20,
                                       random_state=0)

    def mini_job(experiment_type, machine, train_data, train_labels,
                 test_data, test_labels):
        print(experiment_type)
        machine.fit(train_data, train_labels)
        out = machine.predict(train_data)
        out_test = machine.predict(test_data)
        train_auc[experiment_type].append(roc_auc_score(train_labels, out))
        test_auc[experiment_type].append(roc_auc_score(test_labels, out_test))
        #print('train ', train_auc[experiment_type])
        #print('test ', test_auc[experiment_type])
        
    
    train_auc = defaultdict(list)
    test_auc = defaultdict(list)

    i = 0;
    for train_index, test_index in cfolds:
        machine = svm.NuSVC(nu=Globals.nu,
                            kernel='linear',
                            verbose=False,
                            probability=False)

        print(i)
        i = i + 1

        print("NICK transformed data")
        train_data = X_prime[train_index,:]
        train_labels = y[train_index]
        test_data = X_prime[test_index,:]
        test_labels = y[test_index]

        experiment_type = 'NICK svm'
        mini_job(experiment_type, machine, train_data, train_labels,
                 test_data, test_labels)


        print("normal data")
        train_data = tmpX[train_index,:]
        train_labels = y[train_index]
        test_data = tmpX[test_index,:]
        test_labels = y[test_index]


        experiment_type = 'normal svm'
        mini_job(experiment_type, machine, train_data, train_labels,
                 test_data, test_labels)


        experiment_type = 'Gradient Boosting Classifier'
        gbc_model = sklearn.ensemble.GradientBoostingClassifier(max_features = 5,
                                                                max_depth = 2,
                                                                n_estimators = 200)
        mini_job(experiment_type, gbc_model, train_data, train_labels,
                 test_data, test_labels)
        

        experiment_type = 'Adaboost Classifier'
        abc_model = sklearn.ensemble.AdaBoostClassifier(
            sklearn.tree.DecisionTreeClassifier(max_depth=1),
            algorithm = "SAMME",
            n_estimators = 100)
        mini_job(experiment_type, abc_model, train_data, train_labels,
                 test_data, test_labels)


        
        experiment_type = 'Rat 1 model'
        print(experiment_type)

        weak_learner = LogisticRegressionClassifier(C = 0.3)

        rat_model_single = Rat(learner_count = 1,
                               learner = weak_learner, n_jobs = 30)
        rat_model_single.fit(train_data, train_labels)
        out = rat_model_single.predict(train_data)
        out_test = rat_model_single.predict(test_data)
        train_auc[experiment_type].append(roc_auc_score(train_labels, out))
        test_auc[experiment_type].append(roc_auc_score(test_labels, out_test))
        #print('train ', train_auc[experiment_type])
        #print('test ', test_auc[experiment_type])

        

        experiment_type = 'Rat 5 model'
        print(experiment_type)

        weak_learner = LogisticRegressionClassifier(C = 0.3)
        
        rat_model = Rat(learner_count = 5, learner = weak_learner, n_jobs = 30)
        rat_model.fit(train_data, train_labels)
        out = rat_model.predict(train_data)
        out_test = rat_model.predict(test_data)
        train_auc[experiment_type].append(roc_auc_score(train_labels, out))
        test_auc[experiment_type].append(roc_auc_score(test_labels, out_test))
        #print('train ', train_auc[experiment_type])
        #print('test ', test_auc[experiment_type])

        experiment_type = 'Rat 10 model'
        print(experiment_type)

        weak_learner = LogisticRegressionClassifier(C = 0.3)
        
        rat_model = Rat(learner_count = 10, learner = weak_learner, n_jobs = 30)
        rat_model.fit(train_data, train_labels)
        out = rat_model.predict(train_data)
        out_test = rat_model.predict(test_data)
        train_auc[experiment_type].append(roc_auc_score(train_labels, out))
        test_auc[experiment_type].append(roc_auc_score(test_labels, out_test))
        #print('train ', train_auc[experiment_type])
        #print('test ', test_auc[experiment_type])


        def statstr(v):
            return("%.3lg +/- %.3lg" % (np.mean(v), 2 * np.std(v)))
        for key, value in test_auc.items():
            print("test  auc %s: " % (key), statstr(value))
        for key, value in train_auc.items():
            print("train auc %s: " % (key), statstr(value))

    '''
    writer = csv.writer(open("results.csv", "w"))
    writer.writerow(['test_auc', 'train_auc', 'test_tr_auc', 'train_tr_auc'])
    for i in range(len(test_auc)):
        writer.writerow([test_auc[i], train_auc[i],
                        test_tr_auc[i], train_tr_auc[i]])
    '''

    print('bye')


exec(open("./rat.py").read())
a = LogisticRegressionClassifier(n_jobs = 30,
                                 excluded_features=list([346,715,785]),
                                 feature_confidence_estimator=PredictBasedFCE(),
                                 second_layer_feature_count = 5,
                                 C = 0.2)
a.fit(tmpX, y)


with open("/home/ajalali/.local/lib/python3.3/site-packages/sklearn/grid_search.py") as f:
    code = compile(f.read(), "rat.py", 'exec')
    exec(code)


exec(open("./rat.py").read())

with open("./rat.py") as f:
    code = compile(f.read(), "rat.py", 'exec')
    exec(code)
a = Rat(learner_count = 5, learner = LogisticRegressionClassifier(C = 0.3,
                                                                    n_jobs = 30))
a.fit(tmpX, y)
a.score(tmpX, y)


from sklearn.grid_search import GridSearchCV
cvs = cv.StratifiedKFold(y, 5)

rat = Rat(learner_count = 10,
          overlapping_features = True,
          learner = LogisticRegressionClassifier(C = 0.3,
                                                 second_layer_feature_count = 5,
                                                 n_jobs = 30))
learners = list()
for c in [0.3, 0.4]:
    for s in [5]:
        learners.append(LogisticRegressionClassifier(C = c,
                                                     second_layer_feature_count = s,
                                                     n_jobs = 30))

param_grid = {'learner_count': [5], 'learner': learners,
              'overlapping_features':[False]}
model = GridSearchCV(rat, param_grid = param_grid, scoring = 'roc_auc',
                     n_jobs = 1, refit = True)
scores = cv.cross_val_score(
    model, tmpX, y,
    cv=cvs,
    scoring = 'roc_auc',
    n_jobs=1)
