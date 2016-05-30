import numpy as np
import scipy.stats.stats
import sklearn.svm
import sklearn.base
import sklearn.preprocessing
import sklearn.grid_search
from sklearn import cross_validation as cv

from common import FCE


class Raccoon:
    def __init__(self, verbose=0, logger=None, n_jobs=1, dynamic_features=True):
        self.features = None
        self.FCEs = dict()
        self.verbose = verbose
        self.n_jobs = n_jobs
        self.Xtrain = None
        self.ytrain = None
        self.normalizer = None
        self.dynamic_features = dynamic_features
        if logger is None:
            self.logger = print
        else:
            self.logger = logger

    def fit(self, X, y):
        normalizer = sklearn.preprocessing.Normalizer().fit(X)
        self.normalizer = normalizer
        X = self.normalizer.transform(X)
        self.Xtrain = X
        self.ytrain = y

        if self.verbose > 0:
            self.logger("Selecting candidate features for feature pool")

        # calculate pearson correlation and respective p-values
        # select the ones with a p-value < 0.05
        inner_cv = cv.StratifiedKFold(y, n_folds=5)
        i = 0
        features = set(np.arange(X.shape[1]))
        for train, test in inner_cv:
            inner_xtrain = X[train, :]
            inner_ytrain = y[train]
            tmp = [scipy.stats.stats.pearsonr(inner_ytrain, inner_xtrain[:, i])
                   for i in range(inner_xtrain.shape[1])]
            tmp = np.array(tmp)
            threshold_1000 = np.sort(tmp[:, 1])[1000]
            threshold = min(threshold_1000, 0.05)
            if self.verbose > 0:
                self.logger("threshold 1000, threshold: %g %g" % (threshold_1000, threshold))
                
            features = features.intersection(set(np.arange(tmp.shape[0])[abs(tmp[:, 1]) < threshold]))
            if self.verbose > 0:
                self.logger("new features length: %d" % len(features))

        self.features = np.array(list(features))

        if self.verbose > 0:
            self.logger("%d features selected. Fitting feature confidence estimators" % (len(self.features)))

        if self.dynamic_features == False:
            self.logger("Done.")
            return
        
        self.FCEs = dict()
        i = 0
        for f in self.features:
            i += 1
            if self.verbose > 0:
                self.logger("%d / %d fitting FCE for feature %d" % (i, self.features.shape[0], f))
            fce = FCE.RidgeBasedFCE(self.logger, n_jobs=self.n_jobs, verbose=self.verbose)
            fce.fit(X, f)
            self.FCEs[f] = fce

        if self.verbose > 0:
            self.logger("Done.")

    def predict(self, X, model=sklearn.svm.SVC(), param_dist=None):
        X = X.view(np.ndarray)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        X = self.normalizer.transform(X)

        if self.dynamic_features == False:
            if self.verbose > 0:
                self.logger("training the model")
                
            random_search = sklearn.grid_search.RandomizedSearchCV(model, param_distributions=param_dist,
                                                                   n_iter=100, n_jobs=self.n_jobs, cv=10,
                                                                   verbose=0)
            random_search.fit(self.Xtrain[:, self.features], self.ytrain)
            results = list()
            for i in range(X.shape[0]):
                results.append({'model': random_search, 'confidences': None,
                            'selected_features': None,
                            'prediction': random_search.predict(X[i, self.features]),
                            'decision_function': random_search.decision_function(X[i, self.features])})

            if self.verbose > 0:
                self.logger("predict done.")
            return results
            
        results = list()
        for i in range(X.shape[0]):
            if self.verbose > 0:
                self.logger("Sample %d / %d" % (i+1, X.shape[0]))

            if self.verbose > 0:
                self.logger("Selecting high confidence features")

            confidences = {f: self.FCEs[f].getConfidence(X[i, ]) for f in self.features}

            if self.verbose > 2:
                self.logger(confidences)

            max_confidence = max(confidences.values())
            min_confidence = min(confidences.values())

            if self.verbose > 1:
                self.logger("Max and min confidences: %f, %f" % (max_confidence, min_confidence))

            best_threshold = None
            best_score = 0
            for threshold in [0.2, 0.4, 0.6, 0.8]:
                selected_features = [key for (key, value) in confidences.items()
                                     if value > min_confidence + (max_confidence - min_confidence) * threshold]

                if self.verbose > 2:
                    self.logger("Selected features and their confidences:")
                    self.logger([(key, confidences[key]) for key in selected_features])

                random_search = sklearn.grid_search.RandomizedSearchCV(model, param_distributions=param_dist,
                                                                       n_iter=100, n_jobs=self.n_jobs, cv=10,
                                                                       verbose=0)
                random_search.fit(self.Xtrain[:, selected_features], self.ytrain)
                if random_search.best_score_ > best_score:
                    best_score = random_search.best_score_
                    best_threshold = threshold

                if self.verbose > 0:
                    self.logger("score, threshold: %f, %g" % (random_search.best_score_, threshold))

            if self.verbose > 1:
                self.logger("Selected threshold: %g" % (best_threshold))

            selected_features = [key for (key, value) in confidences.items()
                                 if value > min_confidence + (max_confidence - min_confidence) * best_threshold]

            if self.verbose > 2:
                self.logger("Selected features and their confidences:")
                self.logger([(key, confidences[key]) for key in selected_features])

            random_search = sklearn.grid_search.RandomizedSearchCV(model, param_distributions=param_dist,
                                                                   n_iter=100, n_jobs=self.n_jobs, cv=10,
                                                                   verbose=0)
            random_search.fit(self.Xtrain[:, selected_features], self.ytrain)

            results.append({'model': random_search, 'confidences': confidences,
                            'selected_features': selected_features,
                            'prediction': random_search.predict(X[i, selected_features]),
                            'decision_function': random_search.decision_function(X[i, selected_features])})

        return results

