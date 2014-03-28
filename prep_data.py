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


from constants import *;
from misc import *
import read_nordlund1 
import read_nordlund2
import read_vantveer
import read_tcga_brca
import read_tcga_laml
from rat import *

if __name__ == '__main__':
    print('hi');

    data_name = ''
    target_name = ''
    dump_dir = ''
    for i in range(len(sys.argv)):
        print(sys.argv[i])
        if (sys.argv[i] == '--data'):
            data_name = sys.argv[i + 1]
        if (sys.argv[i] == '--target'):
            target_name = sys.argv[i + 1]
        if (sys.argv[i] == '--dump-dir'):
            dump_dir = sys.argv[i + 1]

    print(data_name, target_name, dump_dir)
        

    if (data_name == 'nordlund'):
        if (target_name == 'TvsB'):
            ''' load nordlund T-ALL vs BCP-ALL '''
            (tmpX, y, g,
             sample_annotation, feature_annotation) = read_nordlund1.load_data()
        elif (target_name == 'HeHvst1221'):
            ''' load  nordlund subtypes A vs subtypes B '''
            (tmpX, y, g,
             sample_annotation,
             feature_annotation) = read_nordlund2.load_data('HeH', 't(12;21)')
    elif (data_name == 'vantveer'):
        ''' load vantveer data poor vs good prognosis '''
        (tmpX, y, g,
         sample_annotation, feature_annotation) = read_vantveer.load_data()
    elif (data_name == 'TCGA-BRCA'):
        if (target_name in {'ER', 'T', 'N', 'stage'}):
            ''' load TCGA BRCA data '''
            (tmpX, y, g,
             sample_annotation,
             feature_annotation) = read_tcga_brca.load_data(target_name)
    elif (data_name == 'TCGA-LAML'):
        ''' load TCGA LAML data '''
        if (target_name in {'vital_status', 'risk_group'}):
            (tmpX, y, g,
             sample_annotation,
             feature_annotation) = read_tcga_laml.load_data(target_name)

    print("calculating L and transformation of the data...")
    B = gt.spectral.laplacian(g)
    M = np.identity(B.shape[0]) + Globals.beta * B
    M_inv = np.linalg.inv(M)
    L = np.linalg.cholesky(M_inv)
    X_prime = tmpX.dot(L)

    print("saving data...")

    fold_count = 100
    cvs = list(cv.StratifiedShuffleSplit(y, n_iter = fold_count, test_size = 0.2))

    np.savez(open(dump_dir + '/npdata.npz', 'wb'),
             tmpX = tmpX, X_prime = X_prime, y = y,
             sample_annotation = sample_annotation,
             feature_annotation = feature_annotation)
    g.save(dump_dir + '/graph.xml.gz')
    pickle.dump(cvs, open(dump_dir + '/cvs.dmp', 'wb'))

    print('bye')
    '''
    cpu_count = 1
    max_learner_count = 40
    rat_scores = dict()
    all_scores = defaultdict(list)

    
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

    print_log(all_scores, rat_scores)
    '''