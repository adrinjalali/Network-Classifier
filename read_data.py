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



from constants import *;
from misc import *
from rat import *
import read_nordlund1 
import read_nordlund2
import read_vantveer

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
    cfolds = cv.StratifiedShuffleSplit(y, n_iter=Globals.cfold_count, test_size=0.30,
                                       random_state=0)
    train_auc = list()
    test_auc = list()
    train_tr_auc = list()
    test_tr_auc = list()
    train_rat_auc = list()
    test_rat_auc = list()
    train_rat_single_auc = list()
    test_rat_single_auc = list()

    i = 0;
    for train_index, test_index in cfolds:
        machine = svm.NuSVC(nu=Globals.nu,
                            kernel='linear',
                            verbose=False,
                            probability=False)

        print(i)
        i = i + 1
        print('normal')
        train_data = tmpX[train_index,:]
        train_labels = y[train_index]
        test_data = tmpX[test_index,:]
        test_labels = y[test_index]
        machine.fit(train_data, train_labels)
        out = machine.predict(train_data)
        out_test = machine.predict(test_data)
        train_auc.append(roc_auc_score(train_labels, out))
        test_auc.append(roc_auc_score(test_labels, out_test))
        print('train ', train_auc)
        print('test ', test_auc)


        print('transformed')
        train_data = X_prime[train_index,:]
        train_labels = y[train_index]
        test_data = X_prime[test_index,:]
        test_labels = y[test_index]
        machine.fit(train_data, train_labels)
        out = machine.predict(train_data)
        out_test = machine.predict(test_data)
        train_tr_auc.append(roc_auc_score(train_labels, out))
        test_tr_auc.append(roc_auc_score(test_labels, out_test))
        print('train ', train_tr_auc)
        print('test ', test_tr_auc)

        print("Rat 1 model")
        train_data = tmpX[train_index,:]
        train_labels = y[train_index]
        test_data = tmpX[test_index,:]
        test_labels = y[test_index]

        weak_learner = LogisticRegressionClassifier(C = 0.3)

        rat_model_single = Rat(train_data, train_labels, learner_count = 1,
                               learner = weak_learner)
        rat_model_single.train()
        out = rat_model_single.predict(train_data)
        out_test = rat_model_single.predict(test_data)
        train_rat_single_auc.append(roc_auc_score(train_labels, out))
        test_rat_single_auc.append(roc_auc_score(test_labels, out_test))
        print('train ', train_rat_single_auc)
        print('test ', test_rat_single_auc)

        print("Rat 5 model")
        train_data = tmpX[train_index,:]
        train_labels = y[train_index]
        test_data = tmpX[test_index,:]
        test_labels = y[test_index]

        weak_learner = LogisticRegressionClassifier(C = 0.3)
        
        rat_model = Rat(train_data, train_labels, learner_count = 10,
                        learner = weak_learner)
        rat_model.train()
        out = rat_model.predict(train_data)
        out_test = rat_model.predict(test_data)
        train_rat_auc.append(roc_auc_score(train_labels, out))
        test_rat_auc.append(roc_auc_score(test_labels, out_test))
        print('train ', train_rat_auc)
        print('test ', test_rat_auc)


    def statstr(v):
        return("%.3lg +/- %.3lg" % (np.mean(v), 2 * np.std(v)))
    print("test auc: ", statstr(test_auc))
    print("test transformed auc: ", statstr(test_tr_auc))
    print("test rat single auc: ", statstr(test_rat_single_auc))
    print("test rat auc: ", statstr(test_rat_auc))
    print("train auc: ", statstr(train_auc))
    print("train transformed auc: ", statstr(train_tr_auc))
    print("train rat single auc: ", statstr(train_rat_single_auc))
    print("train rat auc: ", statstr(train_rat_auc))

    '''
    writer = csv.writer(open("results.csv", "w"))
    writer.writerow(['test_auc', 'train_auc', 'test_tr_auc', 'train_tr_auc'])
    for i in range(len(test_auc)):
        writer.writerow([test_auc[i], train_auc[i],
                        test_tr_auc[i], train_tr_auc[i]])
    '''

    print('bye')
