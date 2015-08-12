import numpy as np
import scipy.stats.stats
import sklearn.svm
import sklearn.base
import sklearn.preprocessing
import sklearn.grid_search
from . import FCE


class Raccoon:
    def __init__(self, verbose=0):
        self.features = None
        self.FCEs = dict()
        self.verbose = verbose
        self.Xtrain = None
        self.ytrain = None
        self.normalizer = None

    def fit(self, X, y):
        normalizer = sklearn.preprocessing.Normalizer().fit(X)
        self.normalizer = normalizer
        X = self.normalizer.transform(X)
        self.Xtrain = X
        self.ytrain = y

        if self.verbose > 0:
            print("Selecting candidate features for feature pool")

        # calculate pearson correlation and respective p-values
        # select the ones with a p-value < 0.05
        tmp = [scipy.stats.stats.pearsonr(y, X[:, i]) for i in range(X.shape[1])]
        tmp = np.array(tmp)
        self.features = np.arange(tmp.shape[0])[abs(tmp[:, 1]) < 0.01]

        if self.verbose > 0:
            print(len(self.features), "features selected.\nFitting feature confidence estimators")

        self.FCEs = dict()
        i = 0
        for f in self.features:
            i += 1
            if self.verbose > 1:
                print(i, "/", self.features.shape[0], "fitting FCE for feature", f)
            fce = FCE.RidgeBasedFCE()
            fce.fit(X, f)
            self.FCEs[f] = fce

        if self.verbose > 0:
            print("Done.")

    def predict(self, X, model=sklearn.svm.SVC(), param_dist=None):
        X = X.view(np.ndarray)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        X = self.normalizer.transform(X)

        results = list()
        for i in range(X.shape[0]):
            if self.verbose > 0:
                print("Sample", i+1, "/", X.shape[0])

            if self.verbose > 0:
                print("Selecting high confidence features")

            confidences = {f: self.FCEs[f].getConfidence(X[i, ]) for f in self.features}

            if self.verbose > 2:
                print(confidences)

            max_confidence = max(confidences.values())
            min_confidence = min(confidences.values())

            if self.verbose > 1:
                print("Max and min confidences:", max_confidence, min_confidence)

            best_threshold = None
            best_score = 0
            for threshold in [0.2, 0.4, 0.6, 0.8]:
                selected_features = [key for (key, value) in confidences.items()
                                     if value > min_confidence + (max_confidence - min_confidence) * threshold]

                if self.verbose > 2:
                    print("Selected features and their confidences:")
                    print([(key, confidences[key]) for key in selected_features])

                random_search = sklearn.grid_search.RandomizedSearchCV(model, param_distributions=param_dist,
                                                                       n_iter=100, n_jobs=-1, cv=10)
                random_search.fit(self.Xtrain[:, selected_features], self.ytrain)
                if random_search.best_score_ > best_score:
                    best_score = random_search.best_score_
                    best_threshold = threshold

                if self.verbose > 0:
                    print("score, threshold:", random_search.best_score_, threshold)

            if self.verbose > 1:
                print("Selected threshold:", best_threshold)

            selected_features = [key for (key, value) in confidences.items()
                                 if value > min_confidence + (max_confidence - min_confidence) * best_threshold]

            if self.verbose > 2:
                print("Selected features and their confidences:")
                print([(key, confidences[key]) for key in selected_features])

            random_search = sklearn.grid_search.RandomizedSearchCV(model, param_distributions=param_dist,
                                                                   n_iter=100, n_jobs=-1, cv=10)
            random_search.fit(self.Xtrain[:, selected_features], self.ytrain)

            results.append({'model': random_search, 'confidences': confidences,
                            'selected_features': selected_features,
                            'prediction': random_search.predict(X[i, selected_features]),
                            'decision_function': random_search.decision_function(X[i, selected_features])})

        return results

