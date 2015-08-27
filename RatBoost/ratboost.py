import numpy as np
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

    def fit(self, X, y):
        excluded_features = np.empty(0, dtype=int)
        for i in range(self.max_learners):
            wlearner = WeakLearner(excluded_features=excluded_features, learner=self.learner,
                                   n_jobs=self.n_jobs, verbose=self.verbose, logger=self.logger)
            wlearner.fit(X, y)
            excluded_features = np.union1d(excluded_features, wlearner.get_features())
            self.learners.append(wlearner)
            if excluded_features.shape[0] > (X.shape[1] / 5):
                    break

    def decision_function(self, X,
                          return_iterative=False,
                          return_details=False):
        X = X.view(np.ndarray)

        if X.ndim == 1:
            X = X.reshape(1, -1)

        predictions = np.empty((X.shape[0], 0), dtype=float)
        confidences = np.empty((X.shape[0], 0), dtype=float)
        if return_iterative:
            iterative_result = list()
        for l in self.learners:
            predictions = np.hstack((predictions,
                                     l.decision_function(X).reshape(-1, 1)))
            confidences = np.hstack((confidences,
                                     l.confidence(X,).reshape(-1, 1)))

            if return_iterative:
                if len(iterative_result) > 0:
                    result = np.average(predictions, weights=confidences, axis=1)
                else:
                    result = predictions
                if return_details:
                    result = (result, predictions, confidences)
                iterative_result.append(result)

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
