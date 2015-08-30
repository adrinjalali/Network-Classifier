import numpy as np
import sklearn.preprocessing
from .learner import WeakLearner


class RatBoost:
    def __init__(self, learner=None, max_learners=15, n_jobs=1, logger=None, verbose=0):
        self.n_jobs = n_jobs
        if logger is None:
            self.logger = print
        else:
            self.logger = logger
        self.learner = learner
        self.learners = []
        self.verbose = verbose
        self.max_learners = max_learners
        self.normalizer = None

    def fit(self, X, y):
        normalizer = sklearn.preprocessing.Normalizer().fit(X)
        self.normalizer = normalizer
        X = self.normalizer.transform(X)

        excluded_features = np.empty(0, dtype=int)
        for i in range(self.max_learners):
            if self.verbose > 0:
                self.logger('fitting learner %d / %d' % (i + 1, self.max_learners))
            wlearner = WeakLearner(excluded_features=excluded_features, learner=self.learner,
                                   n_jobs=self.n_jobs, verbose=self.verbose, logger=self.logger)
            wlearner.fit(X, y)
            excluded_features = np.union1d(excluded_features, wlearner.get_features())
            self.learners.append(wlearner)
            if excluded_features.shape[0] > (X.shape[1] / 10):
                if self.verbose > 0:
                    self.logger('Aborting learner fit due to too many selected features: %d / %d' %
                                (excluded_features.shape[0], X.shape[1]))
                break

    def decision_function(self, X,
                          return_iterative=False,
                          return_details=False):
        X = X.view(np.ndarray)

        if X.ndim == 1:
            X = X.reshape(1, -1)

        X = self.normalizer.transform(X)

        predictions = np.empty((X.shape[0], 0), dtype=float)
        confidences = np.empty((X.shape[0], 0), dtype=float)
        if return_iterative:
            iterative_result = list()
        for l in self.learners:
            predictions = np.hstack((predictions,
                                     l.decision_function(X).reshape(-1, 1)))
            confidences = np.hstack((confidences,
                                     l.confidence(X,).reshape(-1, 1)))

            self.logger("in ratboost")
            self.logger('X shape %s' % str(X.shape))
            self.logger('predictions.shape %s' % str(predictions.shape))
            self.logger('confidences.shape %s' % str(confidences.shape))
            self.logger(predictions)
            self.logger(confidences)
            if return_iterative:
                if len(confidences) > 0:
                    result = np.average(predictions, weights=confidences, axis=1)
                else:
                    result = predictions
                if return_details:
                    result = (result, predictions, confidences)
                self.logger('iterative result %s' % str(result))
                iterative_result.append(result)

        self.logger('total iterative result %s' % str(iterative_result))
        if return_iterative:
            return iterative_result

        if len(self.learners) > 1:
            result = np.average(predictions, weights=confidences, axis=1)
        else:
            result = predictions
        if return_details:
            return result, predictions, confidences
        else:
            return result
