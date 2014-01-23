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
import bidict;


from constants import *;
from misc import *
from rat import *
import read_nordlund1 
import read_nordlund2

def get_gene_expression_indices(entrezid, expressions_colnames, probe2gene_array):
    probes = probe2gene_array[np.where(probe2gene_array[:,1] ==
                                       entrezid),0].reshape(-1)
    probes = [x for x in probes if x in expressions_colnames]
    indices = [i for i in range(expressions_colnames.shape[0])
               if expressions_colnames[i] in probes]
    return indices;

def get_genename_entrezids(genename, genename2entrez_array):
    entrezids = genename2entrez_array[np.where(genename2entrez_array[:,0] ==
                                               genename),1].reshape(-1)
    return entrezids;

if __name__ == '__main__':
    print('hi');

    (tmpX, y, g, sample_annotation) = read_nordlund1.load_data()

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
        
        rat_model = Rat(train_data, train_labels, learner_count = 5,
                        learner = weak_learner)
        rat_model.train()
        out = rat_model.predict(train_data)
        out_test = rat_model.predict(test_data)
        train_rat_auc.append(roc_auc_score(train_labels, out))
        test_rat_auc.append(roc_auc_score(test_labels, out_test))
        print('train ', train_rat_auc)
        print('test ', test_rat_auc)


    print("test auc: ", np.mean(test_auc))
    print("test transformed auc: ", np.mean(test_tr_auc))
    print("test rat single auc: ", np.mean(test_rat_single_auc))
    print("test rat auc: ", np.mean(test_rat_auc))
    print("train auc: ", np.mean(train_auc))
    print("train transformed auc: ", np.mean(train_tr_auc))
    print("train rat single auc: ", np.mean(train_rat_single_auc))
    print("train rat auc: ", np.mean(train_rat_auc))

    '''
    writer = csv.writer(open("results.csv", "w"))
    writer.writerow(['test_auc', 'train_auc', 'test_tr_auc', 'train_tr_auc'])
    for i in range(len(test_auc)):
        writer.writerow([test_auc[i], train_auc[i],
                        test_tr_auc[i], train_tr_auc[i]])
    '''

    print('bye')
