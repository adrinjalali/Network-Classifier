from common.misc import Misc
from common import FCE
import sklearn.svm
import sklearn.grid_search
import numpy as np
import scipy.special
import math

class WeakLearner:
    def __init__(self, excluded_features=[], learner=None, n_jobs=1, verbose=0, logger=None):
        self.n_jobs = n_jobs
        if learner is None:
            predictor = sklearn.svm.LinearSVC(penalty='l1', dual=False, class_weight='auto')
            param_dist = {'C': pow(2.0, np.arange(-10, 11))}
            learner = sklearn.grid_search.GridSearchCV(estimator=predictor,
                                                       param_grid=param_dist,
                                                       n_jobs=self.n_jobs, cv=5)
        self.learner = learner
        self.excluded_features = np.copy(excluded_features)
        self._X_colcount = 0
        self.FCEs = dict()
        self.verbose = verbose
        if logger is None:
            self.logger = print
        else:
            self.logger = logger

    def transform(self, X):
        return Misc.exclude_cols(X, self.excluded_features)

    def fit(self, X, y):
        self._X_colcount = X.shape[1]
        my_X = self.transform(X)

        self.learner.fit(my_X, y)

        selected_features = self.get_features()
        i = 0
        for f in selected_features:
            i += 1
            if self.verbose > 1:
                self.logger("%d / %d fitting FCE for feature %d" % (i, selected_features.shape[0], f))
            fce = FCE.RidgeBasedFCE(self.logger, n_jobs=self.n_jobs)
            fce.fit(X, f)
            self.FCEs[f] = fce

    def get_features(self):
        if hasattr(self.learner, 'best_estimator_'):
            learner = self.learner.best_estimator
        else:
            learner = self.learner
        scores = learner.coef_.flatten()
        threshold = np.min(abs(scores)) + (
            np.max(abs(scores)) - np.min(abs(scores))) * 0.0
        local_cols = np.arange(scores.shape[0])[abs(scores) > threshold]
        return np.delete(np.arange(self._X_colcount),
                         self.excluded_features)[local_cols]

    def get_feature_weights(self):
        if hasattr(self.learner, 'best_estimator_'):
            learner = self.learner.best_estimator
        else:
            learner = self.learner
        scores = learner.coef_.flatten()
        threshold = np.min(abs(scores)) + (
            np.max(abs(scores)) - np.min(abs(scores))) * 0.0
        local_cols = np.arange(scores.shape[0])[abs(scores) > threshold]
        scores = scores[local_cols, ]
        features = self.get_features()
        return dict([(features[i], scores[i]) for i in range(len(scores))])

    def decision_function(self, X):
        return self.learner.decision_function(self.transform(X))

    def confidence(self, X):
        def phi(x):
            return 0.5 + 0.5 * scipy.special.erf(x / math.sqrt(2))

        def my_score(x):
            return abs(phi(x) - phi(-x))

        X = X.view(np.ndarray)

        if X.ndim == 1:
            X = X.reshape(1, -1)

        result = np.ones(len(X))
        feature_weights = self.get_feature_weights()
        weight_sum = sum(np.abs(list(feature_weights.values())))
        for key, fc in self.FCEs.items():
            tmp = fc.getConfidence(X) * abs(feature_weights[key])
            result += tmp

        return result / weight_sum
