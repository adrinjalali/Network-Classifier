import sys
sys.path.insert(0, '..')
import pickle
import numpy as np
import core.FCE
import core.raccoon
import graph_tool as gt

if __name__ == '__main__':
    input_dir = '../data/TCGA-SARC/vital_status'
    data_file = np.load(input_dir + '/data.npz')
    X = data_file['X']
    X_prime = data_file['X_prime']
    y = data_file['y']
    sample_annotation = data_file['patient_annot']
    data_file = np.load(input_dir + '/../genes.npz')
    feature_annotation = data_file['genes']
    g = gt.load_graph(input_dir + '/../graph.xml.gz')
    cvs = pickle.load(open(input_dir + '/batch_cvs.dmp', 'rb'))

    # choosing only one cross-validation fold
    cvs_results = list()
    for cv_index in range(len(cvs)):
    #for cv_index in [3]:
        tmp = list()
        tmp.append((cvs[cv_index]))
        tcvs = tmp

        Xtrain = X[tcvs[0][0], ]
        ytrain = y[tcvs[0][0], ]
        Xtest = X[tcvs[0][1], ]
        ytest = y[tcvs[0][1], ]

        model = core.raccoon.Raccoon(verbose=2)
        model.fit(Xtrain, ytrain)

        import sklearn.lda
        import sklearn.neighbors
        import sklearn.qda
        import sklearn.svm
        import sklearn.metrics
        import scipy.stats
        predictor = sklearn.svm.SVC()
        param_dist = {'C': scipy.stats.expon(scale=.1), 'gamma': scipy.stats.expon(scale=0.1),
                      'kernel': ['linear', 'rbf']}
        param_dist = {'C': pow(2.0, np.arange(-10, 11)), 'gamma': pow(2.0, np.arange(-10, 11)),
                      'kernel': ['linear', 'rbf']}
        tmpt = model.predict(Xtrain, model=predictor, param_dist=param_dist)
        tmp = model.predict(Xtest, model=predictor, param_dist=param_dist)
        cvs_results.append({'train_result': tmpt, 'test_result': tmp,
                            'Xtrain': Xtrain, 'ytrain': ytrain,
                            'Xtest': Xtest, 'ytest': ytest})

    train_perfs = list()
    test_perfs = list()
    #for i in range(len(cvs)):
    for i in range(len(cvs_results)):
        train_result = cvs_results[i]['train_result']
        test_result = cvs_results[i]['test_result']
        Xtrain = cvs_results[i]['Xtrain']
        ytrain = cvs_results[i]['ytrain']
        Xtest = cvs_results[i]['Xtest']
        ytest = cvs_results[i]['ytest']
        print("CV", i)
        print("    ", "train")
        #print("    ", [k['decision_function'][0] for k in train_result])
        #print("    ", [k['prediction'][0] for k in train_result])
        #print("    ", ytrain)
        try:
            #perf = sklearn.metrics.roc_auc_score(ytrain, [k['decision_function'][0] for k in train_result])
            #precision, recall, thresholds = sklearn.metrics.precision_recall_curve(ytrain, [k['decision_function'][0] for k in train_result])
            #perf = sklearn.metrics.auc(recall, precision)
            perf = sklearn.metrics.average_precision_score(ytrain, [k['decision_function'][0] for k in train_result])
            print("    ", perf)
            train_perfs.append(perf)
        except:
            pass

        print("    ", "test")
        #print("    ", [k['decision_function'][0] for k in test_result])
        #print("    ", [k['prediction'][0] for k in test_result])
        #print("    ", ytest)
        try:
            #perf = sklearn.metrics.roc_auc_score(ytest, [k['decision_function'][0] for k in test_result])
            #precision, recall, thresholds = sklearn.metrics.precision_recall_curve(ytest, [k['decision_function'][0] for k in test_result])
            #perf = sklearn.metrics.auc(recall, precision)
            perf = sklearn.metrics.average_precision_score(ytest, [k['decision_function'][0] for k in test_result])
            print("    ", perf)
            test_perfs.append(perf)
        except:
            pass
