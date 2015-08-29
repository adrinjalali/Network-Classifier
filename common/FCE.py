# FCE: feature confidence extimator
import math

import sklearn
import sklearn.kernel_ridge
import sklearn.svm
import numpy as np
import sklearn.cross_validation as cv
from sklearn.base import BaseEstimator
import scipy.special

from common.misc import Misc


class RidgeBasedFCE(BaseEstimator):
    """ This class uses Ridge Regression as the regression
    algorithm. It then calculates the confidence from the difference
    between observed and predicted value, and expected variance which
    is calculated from training data.
    """

    def __init__(self, logger=None, n_jobs=1, verbose=0):
        #self._learner = sklearn.kernel_ridge.KernelRidge(alpha=10,
        #                                                 kernel='linear',
        #                                                 gamma=None,
        #                                                 degree=3,
        #                                                 coef0=1,
        #                                                 kernel_params=None)
        #param_dist = {'C': pow(2.0, np.arange(-10, 11)), 'gamma': pow(2.0, np.arange(-10, 11)),
        #              'kernel': ['linear', 'rbf']}
        model = sklearn.svm.SVR(C=0.1, kernel='linear')
        param_dist = {'C': pow(2.0, np.arange(-10, 11))}
        self._learner = sklearn.grid_search.RandomizedSearchCV(model, param_distributions=param_dist,
                                                               n_iter=40, n_jobs=n_jobs, cv=5,
                                                               verbose=verbose)
        self.feature = None
        self.error_mean = None
        self.error_std = None
        self.input_col_count = None
        if logger is None:
            self.logger = print
        else:
            self.logger = logger

    def fit(self, X, feature):
        try:
            feature = int(feature)
        except Exception:
            self.logger("feature should be int")
            raise TypeError("feature should be int")

        X = X.view(np.ndarray)
        self.input_col_count = X.shape[1]
        self.feature = feature
        my_X = Misc.exclude_cols(X, self.feature)

        cvs = cv.KFold(len(X), 10, shuffle=True)
        output_errors = np.empty(0)
        for train, test in cvs:
            tmp_l = sklearn.clone(self._learner)
            tmp_l.fit(my_X[train, :], X[train, self.feature])
            output_errors = np.hstack((output_errors, tmp_l.predict(my_X[test, :]) - X[test, self.feature]))

        self.error_std = np.std(output_errors)
        self.error_mean = np.mean(output_errors)

        self._learner.fit(my_X, X[:, self.feature])

        return self

    def predict(self, X):
        X = X.view(np.ndarray)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        return self._learner.predict(Misc.exclude_cols(X, self.feature))

    def getConfidence(self, X):
        def phi(x):
            return 0.5 + 0.5 * scipy.special.erf(x / math.sqrt(2))

        def my_score(x):
            return 1 - abs(phi(x) - phi(-x))

        X = X.view(np.ndarray)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        observed_diff = self._learner.predict(Misc.exclude_cols(X, self.feature)) - X[:, self.feature]

        return my_score((observed_diff - self.error_mean) / self.error_std)

    def getFeatures(self):
        if hasattr(self._learner, "coef_"):
            local_cols = np.arange(self._learner.coef_.shape[0])[self._learner.coef_ != 0]
            return np.delete(np.arange(self.input_col_count), self.feature)[local_cols]
        else:
            return []

