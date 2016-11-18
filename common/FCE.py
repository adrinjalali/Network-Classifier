# FCE: feature confidence extimator
import math
import heapq

import sklearn
import sklearn.kernel_ridge
import sklearn.svm
import numpy as np
import sklearn.cross_validation as cv
import sklearn.gaussian_process as gp
from sklearn.base import BaseEstimator
import scipy.special
from joblib import Parallel, delayed
import GPy

from common.misc import Misc
from common.rdc import R_rdc


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
        #self._learner = gp.GaussianProcessRegressor(alpha=1e-2, n_restarts_optimizer=5)
        self.feature_count = feature_count
        self.n_jobs = n_jobs
        #self.n_jobs = 1 # no gain was observed with multithreading
        if logger is None:
            self.logger = print
        else:
            self.logger = logger
        self.verbose = verbose
        self._selected_features = None

    def fit(self, X, feature, fit_rdcs = True, fit_gp = True):
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

        if fit_rdcs:
            scores = R_rdc(my_X, my_y)
            """
            if self.n_jobs > 1:
                scores = Parallel(n_jobs=self.n_jobs, backend="multiprocessing")(
                    delayed(rdc)(X[:,self.feature], my_X[:,i])
                    for i in range(my_X.shape[1]))
            else:
                scores = [rdc(my_y, my_X[:,i])
                        for i in range(my_X.shape[1])]
            """

            if self.verbose > 0:
                self.logger("rdc scores calculated")

            scores = np.array(scores)
            scores[np.isnan(scores)] = 0
        
            self._selected_features = self._selectFeatures(scores = scores,
                                                        k = self.feature_count)


        if fit_gp:
            if self._selected_features == None:
                if self.verbose > 0:
                    self.logger("you need to fit the rdcs first")
                raise RuntimeError("you need to fit the rdcs first")
        
            cols = len(self._selected_features)
        
            if self.verbose > 0:
                self.logger("training GP with %d input features" % cols)
            
            kernel = GPy.kern.Linear(input_dim = cols) + GPy.kern.White(input_dim = cols)
            self._learner = GPy.models.GPRegression(my_X[:, self._selected_features],
                                                    my_y.reshape(-1,1), kernel)
            self._learner.optimize()
        
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
        #res = (np.arange(len(scores))[scores >
        #                                   np.max(scores) * 0.90])
        #if (res.shape[0] < 5):
        
        res = (np.array([t[0] for t in heapq.nlargest(k,
                                                      enumerate(scores),
                                                      lambda t:t[1])]))
        return(res)

    def predict(self, X):
        X = X.view(np.ndarray)
        if (X.ndim == 1):
            X = X.reshape(1, -1)
        
        my_X = Misc.exclude_cols(X, self.feature)
        mean, _ = self._learner.predict(my_X[:, self._selected_features], full_cov=False, include_likelihood=True)
        return mean.reshape(-1)
        
    def getConfidence(self, X):
        def phi(x): return(0.5 + 0.5 * scipy.special.erf(x / math.sqrt(2)))
        def my_score(x): return(1 - abs(phi(x) - phi(-x)))
                
        X = X.view(np.ndarray)
        if (X.ndim == 1):
            X = X.reshape(1, -1)
        
        my_X = Misc.exclude_cols(X, self.feature)
        mean, var = self._learner.predict(my_X[:, self._selected_features], full_cov=False, include_likelihood=True)
        y_obs = X[:, self.feature]
        normalized_y = ((y_obs - mean) / np.sqrt(var)).reshape(-1)
        yscore = np.array([my_score(iy) for iy in normalized_y])
        return yscore

    def getFeatures(self):
        local_cols = self._selected_features
        return(np.delete(np.arange(self._X_colcount),
                         self.excluded_features)[local_cols])

