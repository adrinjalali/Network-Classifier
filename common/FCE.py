# FCE: feature confidence extimator
import math

import sklearn
import sklearn.kernel_ridge
import sklearn.svm
import numpy as np
import sklearn.cross_validation as cv
import sklearn.gaussian_process as gp
from sklearn.base import BaseEstimator
import scipy.special
from joblib import Parallel, delayed

from common.misc import Misc
from common.rdc import rdc


class RidgeBasedFCE(BaseEstimator):
    """ This class uses Ridge Regression as the regression
    algorithm. It then calculates the confidence from the difference
    between observed and predicted value, and expected variance which
    is calculated from training data.
    """

    def __init__(self, logger=None, n_jobs=1, verbose=0):
        model = sklearn.svm.SVR(C=0.1, kernel='linear')
        param_dist = {'C': pow(2.0, np.arange(-10, 11))}
        self._learner = sklearn.grid_search.GridSearchCV(model, param_grid=param_dist,
                                                         n_jobs=n_jobs, cv=5,
                                                         verbose=0)
        self._learner = sklearn.svm.SVR(C=0.1, kernel='linear')

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
        my_y = X[:, self.feature]
        y_mean = np.mean(my_y)
        y_std = np.std(my_y)

        # ref: http://www.sciencedirect.com/science/article/pii/S0893608004002102
        self._learner.C = max(abs(y_mean + 3 * y_std), abs(y_mean - 3 * y_std))

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

class PredictBasedFCE(BaseEstimator):
    ''' This class uses Gaussian Processes as the regression
    algorithm. It uses Mutual Information to select features
    to give to the GP, and at the end uses GP's output, compared
    to the observed value, and the predicted_MSE of the GP, to
    calculate the confidence.
    '''
    def __init__(self, feature_count=10, n_jobs=1,
                 logger=None, verbose=0):
        self._learner = gp.GaussianProcess(nugget=1e-1, optimizer='Welch',
                                           random_start = 10)
        self.feature_count = feature_count
        self.n_jobs = n_jobs
        if logger is None:
            self.logger = print
        else:
            self.logger = logger
        self.verbose = verbose

    def fit(self, X, feature):
        try:
            feature = int(feature)
        except Exception:
            if self.verbose > 0:
                self.logger("feature should be int")
            raise TypeError("feature should be int")

        X = X.view(np.ndarray)
        self.input_col_count = X.shape[1]
        self.feature = feature
        my_X = Misc.exclude_cols(X, self.feature)
        my_y = X[:, self.feature]

        if self.n_jobs > 1:
            scores = Parallel(n_jobs=self.n_jobs, backend="multiprocessing")(
                delayed(rdc)(X[:,self.feature], my_X[:,i])
                for i in range(my_X.shape[1]))
        else:
            scores = [rdc(my_y, my_X[:,i])
                      for i in range(my_X.shape[1])]

        scores = np.array(scores)
        scores[np.isnan(scores)] = 0
        
        self._selected_features = self._selectFeatures(scores = scores,
                                                       k = self.feature_count)
        #try:
        self._learner.fit(my_X[:,self._selected_features], my_y)
        #except:
        '''    print('gp failed.')
        print('feature:', feature)
        print('excluded_feature:', self.excluded_features)
        print('selected cols:', self._selected_features)
        print('X.shape, my_X.shape:', X.shape, my_X.shape)
        raise(RuntimeError('gp failed.'))'''
        self._trained = True
        return(self)
        
    def _selectFeatures(self, scores, k = 10):
        ''' computes mutual information of all features with
        the target feature. Note that excluded_features and the
        target feature are excluded from self.X in initialization.
        Then returns k features of corresponding to most MICs.
        The precision can be improved by increasing alpha of the
        MINE object, but it drastically increases the computation
        time, and the ordering of the features doesn't change much.
        '''
        res = (np.arange(len(scores))[scores >
                                           np.max(scores) * 0.90])
        if (res.shape[0] < 5):
            res = (np.array([t[0] for t in heapq.nlargest(5,
                                                          enumerate(scores),
                                                          lambda t:t[1])]))
        return(res)

    def predict(self, X):
        X = X.view(np.ndarray)
        if (X.ndim == 1):
            X = X.reshape(1, -1)
        
        my_X = X[:,self.getFeatures()]
        return(self._learner.predict(my_X))
        
    def getConfidence(self, X):
        def phi(x): return(0.5 + 0.5 * scipy.special.erf(x / math.sqrt(2)))
        def my_score(x): return(1 - abs(phi(x) - phi(-x)))
                
        X = X.view(np.ndarray)
        if (X.ndim == 1):
            X = X.reshape(1, -1)
        
        my_X = X[:,self.getFeatures()]
        y_pred, sigma2_pred = self._learner.predict(my_X, eval_MSE=True)
        res = []
        for i in range(len(y_pred)):
            standardized_x = (X[i, self.feature] -
                              y_pred[i]) / math.sqrt(sigma2_pred[i])
            res.append(my_score(standardized_x))
        return(np.array(res))
        #return(sigma2_pred / (abs(y_pred - X[:,self.feature]) + sigma2_pred))
        #return(1 / (abs(y_pred - sample[:,self.feature]) + sigma2_pred))

    def getFeatures(self):
        local_cols = self._selected_features
        return(np.delete(np.arange(self._X_colcount),
                         self.excluded_features)[local_cols])

