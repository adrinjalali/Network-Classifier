import gc
import numpy as np
import pandas
import sklearn
import sklearn.svm
import sklearn.metrics
import sklearn.ensemble
import sklearn.tree
import itertools
import os
import sys

import Raccoon.core.raccoon

def add_to_scores(params):
    current = all_scores
    for item in params:
        if not (item in current):
            current[item] = dict()
        current = current[item]

def log(msg=''):
    if 'working_dir' in globals():
        d = list(filter(None, working_dir.split('/')))
    else:
        d = ['','','']
    if 'cv_index' not in globals():
        cv_index = -1
    print('%s\t%s\tcv:%d\t%s' % (d[-2], d[-1], cv_index, msg), file=sys.stderr, flush=True)

if __name__ == "__main__":
    raw_dir = "/TL/stat_learn/work/ajalali/Data/deep_aging/"
    proc_data_file = '%s/proc_data.npz' % raw_dir
    annot_file = "/TL/stat_learn/work/ajalali/Data/met_annot.csv"
    cpu_count = 40

    if (os.path.isfile(proc_data_file)):
        tmp = np.load(proc_data_file)
        Xtrain = tmp['Xtrain']
        Xtest = tmp['Xtest']
        Ytrain = tmp['Ytrain']
        Ytest = tmp['Ytest']
        del tmp
    else:
        annot = pandas.read_csv(annot_file)
    
        tmpX = pandas.read_csv(raw_dir + "training_set.csv")
        Xtrain = np.array(tmpX)
        marker_names = Xtrain[:, 0]
        Xtrain = Xtrain[:, 1:]
        Xtrain = Xtrain.transpose()
        Xtrain = Xtrain.astype('f')
    
        tmpY = pandas.read_csv(raw_dir + "training_phenotype_info.csv")
        Ytrain = np.array(tmpY['age'])
        Ytrain = Ytrain.astype('f')
    
        tmpX = pandas.read_csv(raw_dir + "test_set.csv")
        Xtest = np.array(tmpX)
        marker_names_test = Xtest[:, 0]
        Xtest = Xtest[:, 1:]
        Xtest = Xtest.transpose()
        Xtest = Xtest.astype('f')
        
        tmpY = pandas.read_csv(raw_dir + "test_phenotype_info.csv")
        Ytest = np.array(tmpY['age'])
        Ytest = Ytest.astype('f')

        del tmpX
        del tmpY
        gc.collect()

        means = np.nanmean(Xtrain, axis=0)
        for i in range(means.shape[0]):
            Xtrain[:, i][np.isnan(Xtrain[:, i])] = means[i]
            Xtest [:, i][np.isnan(Xtest [:, i])] = means[i]

        tmp = np.array(annot['UCSC_REFGENE_NAME']).astype(str)
        tmp = tmp[tmp != 'nan']
        tmp = [x.split(';') for x in tmp]
        genes = list(set(list(itertools.chain(*tmp))))

        targetids = np.array(annot['TargetID']).astype(str)
        target_genes = np.array(annot['UCSC_REFGENE_NAME']).astype(str)
        targetids2genes = {targetids[i]: target_genes[i] for i in range(len(targetids))}
    
        genes2targetid = {g:[] for g in genes}
        for i in range(marker_names.shape[0]):
            if marker_names[i] not in targetids2genes:
                continue
            if targetids2genes[marker_names[i]] == 'nan':
                continue
            gs = targetids2genes[marker_names[i]].split(';')
            for g in gs:
                genes2targetid[g].append(i)

        counter = 0
        Xtr = np.ndarray(shape=(Xtrain.shape[0], len(genes2targetid)))
        Xtr[:,:] = 0
        Xte = np.ndarray(shape=(Xtest.shape[0], len(genes2targetid)))
        Xte[:,:] = 0
        for g, idx in genes2targetid.items():
            if len(idx) == 0:
                continue
            Xtr[:,counter] = np.median(Xtrain[:,idx], axis=1)
            Xte[:,counter] = np.median(Xtest[:,idx], axis=1)
            counter = counter + 1

        Xtrain = Xtr
        Xtest = Xte
        means = np.nanmean(Xtrain, axis=0)
        stds = np.nanstd(Xtrain, axis=0) + 0.000001
        Xtrain = (Xtrain - means) / stds
        Xtest = (Xtest - means) / stds
        np.savez_compressed(proc_data_file, Xtrain=Xtrain, Xtest=Xtest,
                            Ytrain=Ytrain, Ytest=Ytest)


    print('svms')
    #model = sklearn.svm.SVR(kernel='linear', C=.003)
    model.fit(Xtrain, Ytrain)
    #sklearn.metrics.r2_score(Ytest, model.predict(Xtest))
    
    model = sklearn.svm.SVR()
    params = {'C': pow(2.0, np.arange(-10, 11)),
              'gamma': pow(2.0, np.arange(-10, 11)),
              'kernel': ['linear', 'rbf']}
    machine = sklearn.grid_search.RandomizedSearchCV(model,
                                                     param_distributions=params,
                                                     scoring = 'r2',
                                                     n_iter=100, n_jobs=cpu_count,
                                                     cv=10,
                                                     verbose=3)
    machine.fit(Xtrain, Ytrain)
    sklearn.metrics.r2_score(Ytest, machine.predict(Xtest))
    # returns 0.81601599268590863

    print('l1-svm')
    model = sklearn.svm.LinearSVR(loss='epsilon_insensitive', epsilon=0)
    params = {'C': pow(2.0, np.arange(-10, 11))}
    machine = sklearn.grid_search.GridSearchCV(model,
                                               param_grid=params,
                                               scoring = 'r2',
                                               n_jobs=cpu_count,
                                               cv=10,
                                               verbose=3)
    machine.fit(Xtrain, Ytrain)
    sklearn.metrics.r2_score(Ytest, machine.predict(Xtest))
    # returns -9.6597408915601903
    

    print('raccoon')
    model = Raccoon.core.raccoon.Raccoon(verbose=1, logger=log, n_jobs=cpu_count)
    model.fit(Xtrain, Ytrain)
    predictor = sklearn.svm.SVR()
    params = {'C': pow(2.0, np.arange(-10, 11)),
              'gamma': pow(2.0, np.arange(-10, 11)),
              'kernel': ['linear', 'rbf']}


        
    model = Raccoon.core.raccoon.Raccoon(verbose=1, logger=log, n_jobs=cpu_count,
                                         FCE_type='PredictBasedFCE')
    model.fit(Xtrain, Ytrain)
    predictor = sklearn.svm.SVR()
    params = {'C': pow(2.0, np.arange(-10, 11)),
              'gamma': pow(2.0, np.arange(-10, 11)),
              'kernel': ['linear', 'rbf']}
    test_results = model.predict(Xtest, model=predictor, param_dist=params)
    scores = sklearn.metrics.average_precision_score(ytest, [k['decision_function'][0] for k in test_results])


    import cProfile, pstats, io
    import common.rdc
    f = 3721
    pr = cProfile.Profile()
    pr.enable()
    garbage = [common.rdc.rdc(Xtrain[:,f], Xtrain[:,i]) for i in range(200)]
    pr.disable()
    
    s = io.StringIO()
    sortby = 'cumulative'
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats()
    print(s.getvalue())

    rdcs = [common.rdc.rdc(Xtrain[:,f], Xtrain[:,i]) for i in range(Xtrain.shape[1])]
    from joblib import Parallel, delayed
    rdcs2 = Parallel(n_jobs=cpu_count, backend="threading")(
        delayed(common.rdc.rdc)(Xtrain[:,f], Xtrain[:,i])
        for i in range(Xtrain.shape[1]))
