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
import shutil


from constants import *;
from misc import *
import read_nordlund1 
import read_nordlund2
import read_vantveer
import read_tcga_brca
import read_tcga_laml
import read_tcga_laml_geneexpression
import read_tcga
from rat import *

if __name__ == '__main__':
    print('hi');

    data_name = ''
    target_name = ''
    dump_dir = ''
    batch_based_cv = False
    for i in range(len(sys.argv)):
        print(sys.argv[i])
        if (sys.argv[i] == '--data'):
            data_name = sys.argv[i + 1]
        if (sys.argv[i] == '--target'):
            target_name = sys.argv[i + 1]
        if (sys.argv[i] == '--dump-dir'):
            dump_dir = sys.argv[i + 1]
        if (sys.argv[i] == '--batch-based-cv'):
            batch_based_cv = True

    print(data_name, target_name, dump_dir)
        

    if (data_name == 'nordlund'):
        print("nordlund didn't give us enough information about the dataset,\
        and we decided not to work on the dataset anymore.\
        continuing without any output...")
        sys.exit(0)
        
    elif (data_name == 'vantveer'):
        ''' load vantveer data poor vs good prognosis '''
        (tmpX, y, g,
         sample_annotation, feature_annotation) = read_vantveer.load_data()
        print("calculating L and transformation of the data...")
        B = gt.spectral.laplacian(g)
        M = np.identity(B.shape[0]) + Globals.beta * B
        M_inv = np.linalg.inv(M)
        L = np.linalg.cholesky(M_inv)
        X_prime = tmpX.dot(L)

        print("saving data...")
    
        if (batch_based_cv == False):
            fold_count = 100
            cvs = list(cv.StratifiedShuffleSplit(
                y, n_iter = fold_count, test_size = 0.2))

        np.savez(open(dump_dir + '/npdata.npz', 'wb'),
                tmpX = tmpX, X_prime = X_prime, y = y,
                sample_annotation = sample_annotation,
                feature_annotation = feature_annotation)
        g.save(dump_dir + '/graph.xml.gz')
        pickle.dump(cvs, open(dump_dir + '/cvs.dmp', 'wb'))
        sys.exit(0)
    elif (data_name == 'TCGA-LAML-GeneExpression'):
        ''' load TCGA LAML data '''
        if (target_name in {'vital_status', 'risk_group'}):
            (tmpX, y, g,
             sample_annotation,
             feature_annotation) = read_tcga_laml_geneexpression.load_data(
                 target_name)
        print("calculating L and transformation of the data...")
        B = gt.spectral.laplacian(g)
        M = np.identity(B.shape[0]) + Globals.beta * B
        M_inv = np.linalg.inv(M)
        L = np.linalg.cholesky(M_inv)
        X_prime = tmpX.dot(L)

        print("saving data...")
    
        if (batch_based_cv == False):
            fold_count = 100
            cvs = list(cv.StratifiedShuffleSplit(
                y, n_iter = fold_count, test_size = 0.2))

        np.savez(open(dump_dir + '/npdata.npz', 'wb'),
                tmpX = tmpX, X_prime = X_prime, y = y,
                sample_annotation = sample_annotation,
                feature_annotation = feature_annotation)
        g.save(dump_dir + '/graph.xml.gz')
        pickle.dump(cvs, open(dump_dir + '/cvs.dmp', 'wb'))
        sys.exit(0)
        
    elif (data_name == 'TCGA-BRCA'):
        input_dir = "/TL/stat_learn/work/ajalali/Data/TCGA-BRCA/"
        sample_type = '01A'
        target_labels = {'er_status_by_ihc': {-1: 'Negative', 1: 'Positive'},
                 'ajcc_pathologic_tumor_stage': {-1: ['Stage I','Stage IA','Stage IB',
                           'Stage II','Stage IIA', 'Stage IIB'],
                           1: ['Stage III', 'Stage IIIA', 'Stage IIIB']},
                 'ajcc_tumor_pathologic_pt': {-1: ['T1', 'T2'], 1: ['T3', 'T4']},
                 'ajcc_nodes_pathologic_pn': {-1: 'N0', 1: ['N1', 'N2', 'N3']}}
            
    elif (data_name == 'TCGA-LAML'):
        input_dir = "/TL/stat_learn/work/ajalali/Data/TCGA-LAML/"
        sample_type = '03A'
        target_labels = {'cyto_risk_group': {-1: 'Favorable',
                                        1: ['Intermediate/Normal', 'Poor']},
                         'vital_status': {-1: 'Alive', 1: 'Dead'}}
            
    elif (data_name == 'TCGA-UCEC'):
        input_dir = '/TL/stat_learn/work/ajalali/Data/TCGA-UCEC/'
        sample_type = '01A'
        target_labels = {'vital_status': {-1:'Dead', 1:'Alive'},
                        'tumor_status': {-1: 'TUMOR FREE', 1: 'WITH TUMOR'},
                        'retrospective_collection': {-1: 'NO', 1: 'YES'}}
    elif (data_name == 'TCGA-THCA'):
        input_dir = '/TL/stat_learn/work/ajalali/Data/TCGA-THCA'
        sample_type = '01A'
        target_labels = {'tumor_focality': {1: 'Multifocal',
                                            -1: 'Unifocal'},
                         'ajcc_pathologic_tumor_stage': {1: ['Stage I', 'Stage II'],
                                                         -1: ['Stage III',
                                                              'Stage IV',
                                                              'Stage IVA',
                                                              'Stage IVC']}}
    elif (data_name == 'TCGA-SARC'):
        input_dir = '/TL/stat_learn/work/ajalali/Data/TCGA-SARC/'
        sample_type = '01A'
        target_labels = {'vital_status': {-1:'Dead', 1:'Alive'},
                        'tumor_status': {-1: 'TUMOR FREE', 1: 'WITH TUMOR'},
                        'residual_tumor': {-1: 'R0', 1: 'R1'}}
    elif (data_name == 'TCGA-LGG'):
        input_dir = '/TL/stat_learn/work/ajalali/Data/TCGA-LGG/'
        sample_type = '01A'
        target_labels = {'vital_status': {-1:'Dead', 1:'Alive'},
                        'tumor_status': {-1: 'TUMOR FREE', 1: 'WITH TUMOR'},
                        'histologic_diagnosis': {-1: 'Astrocytoma',
                                                 1: 'Oligodendroglioma'},
                        'tumor_grade': {-1: 'G2', 1:'G3'}}
    
    data = read_tcga.load_data(input_dir = input_dir,
                sample_type = sample_type,
                target_labels = target_labels)

    print('copy from %s/processed \nto %s/%s' % (input_dir, dump_dir, data_name))
    shutil.copytree(input_dir + '/processed', dump_dir + '/' + data_name)
    
    print('bye')
