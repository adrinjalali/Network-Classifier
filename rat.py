''' This is the bootstrap - network existance assumed - basian inspired
    classifier thing.
    License: GPLv3.
    Adrin Jalali.
    Jan 2014, Saarbruecken, Germany.
'''

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model.base import LinearClassifierMixin
import sklearn.base
from sklearn.linear_model import LogisticRegression
import sklearn.cross_validation as cv
import sklearn.gaussian_process as gp
from minepy import MINE
import heapq
import numpy as np
from joblib import Parallel, delayed
import copy

class utilities:
    def exclude_cols(X, cols):
        ''' exludes indices in cols, from columns of X '''
        return(X[:,~np.in1d(np.arange(X.shape[1]), cols)])

    def check_1d_array(a):
        return(isinstance(a, np.ndarray) and (a.ndim == 1 or
                                              a.shape[0] == 1 or
                                              a.shape[1] == 1))


def _evaluate_single(data, target_feature):
    mine = MINE(alpha=0.3, c=15)
    MICs = list()
    for i in range(data.shape[1]):
        mine.compute_score(target_feature,data[:,i])
        MICs.append(mine.mic())
    return(MICs)
class SecondLayerFeatureEvaluator:
        
    def evaluate(self, data, target_features, n_jobs = 1):
            
        result = np.zeros((target_features.shape[1], data.shape[1]))
        if (n_jobs == 1):
            for i in range(target_features.shape[1]):
                result[i,:] = _evaluate_single(data, target_features[:,i])
                
        elif (n_jobs > 1):
            result = Parallel(n_jobs = n_jobs)(
                delayed(_evaluate_single)
                (data, target_features[:,i]) for i in range(target_features.shape[1]))
            
        return(result)

class PredictBasedFCE(BaseEstimator):
    ''' This class uses Gaussian Processes as the regression
    algorithm. It uses Mutual Information to select features
    to give to the GP, and at the end uses GP's output, compared
    to the observed value, and the predicted_MSE of the GP, to
    calculate the confidence.
    '''
    def __init__(self, feature_count=5):
        self._learner = gp.GaussianProcess()
        #self._learner = gp.GaussianProcess(theta0=1e-2, thetaL=1e-4,
        #                                   thetaU=1e-1, optimizer='Welch')
        self.feature_count = feature_count

    def get_params(self, deep=True):
        return({'feature_count' : self.feature_count})

    def set_params(self, **parameters):
        self.__init__(**parameters)
        return(self)
        
    def fit(self, X, feature, scores, excluded_features):
        try:
            feature = int(feature)
        except Exception:
            raise TypeError("feature should be int")

        if (isinstance(excluded_features, set) or
            isinstance(excluded_features, list)):
            excluded_features = np.array(list(excluded_features), dtype=int)
            
        if not utilities.check_1d_array(excluded_features):
            raise TypeError("excluded features should be 1d ndarray")

        X = X.view(np.ndarray)
        self._X_colcount = X.shape[1]
        self.feature = feature
        self.excluded_features = np.union1d(excluded_features, [feature])
        my_X = utilities.exclude_cols(X, self.excluded_features)

        self._selected_features = self._selectFeatures(scores = scores,
                                                       k = self.feature_count)
        #try:
        self._learner.fit(my_X[:,self._selected_features],
                          X[:,self.feature])
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
        return(np.array([t[0] for t in heapq.nlargest(k,
                                                      enumerate(scores),
                                                      lambda t:t[1])]))

    def predict(self, X):
        X = X.view(np.ndarray)
        if (X.ndim == 1):
            X = X.reshape(1, -1)
        
        my_X = X[:,self.getFeatures()]
        return(self._learner.predict(my_X))
        
    def getConfidence(self, X):
        X = X.view(np.ndarray)
        if (X.ndim == 1):
            X = X.reshape(1, -1)
        
        my_X = X[:,self.getFeatures()]
        y_pred, sigma2_pred = self._learner.predict(my_X, eval_MSE=True)
        return(sigma2_pred / (abs(y_pred - X[:,self.feature]) + sigma2_pred))
        #return(1 / (abs(y_pred - sample[:,self.feature]) + sigma2_pred))

    def getFeatures(self):
        local_cols = self._selected_features
        return(np.delete(np.arange(self._X_colcount),
                         self.excluded_features)[local_cols])
        

class BaseWeakClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self):
        pass
        
    def __init__(self, learner, n_jobs = 1,
                 excluded_features=None, feature_confidence_estimator=None):
        self.learner = learner
        self.n_jobs = n_jobs
        self.feature_confidence_estimator = feature_confidence_estimator

        if (isinstance(excluded_features, set) or
            isinstance(excluded_features, list)):
            excluded_features = np.array(list(excluded_features), dtype=int)
        
        if (excluded_features == None):
            self.excluded_features = np.empty(0, dtype=int)
        else:
            self.excluded_features = np.copy(excluded_features)

        self._FCEs = dict()
        self._second_layer_features = np.empty(0, dtype=int)

    def get_params(self, deep=True):
        return( {
            'learner':self.learner,
            'excluded_features':self.excluded_features,
            'feature_confidence_estimator':self.feature_confidence_estimator,
            'n_jobs':self.n_jobs})

    def set_params(self, **parameters):
        self.__init__(**parameters)
        return(self)

    def _transform(self, X):
        return(utilities.exclude_cols(X, self.excluded_features))
    
    def fit(self, X, y):
        self._X_colcount = X.shape[1]
        self.learner.fit(self._transform(X), y)
        classifier_features = self.getClassifierFeatures()

        fe = SecondLayerFeatureEvaluator()
        local_excluded_features = np.union1d(self.excluded_features,
                                             classifier_features)
        local_X = utilities.exclude_cols(X,
                                         local_excluded_features)
        scores = fe.evaluate(local_X, X[:,classifier_features],
                             n_jobs = self.n_jobs)
        i = 0
        for feature in classifier_features:
            fc = sklearn.base.clone(
                self.feature_confidence_estimator).set_params(
                    **self.feature_confidence_estimator.get_params())
            fc.fit(X, feature,
                   scores[i],
                   local_excluded_features)
            self.setFeatureConfidenceEstimator(feature, fc)
            self._second_layer_features = np.union1d(
                self._second_layer_features, fc.getFeatures())
            i += 1
        return(self)
                    
    def predict(self, X):
        return(self.learner.predict(self._transform(X)))
        
    def predict_proba(self, X):
        return(self.learner.predict_proba(self._transform(X)))

    def decision_function(self, X):
        return(self.learner.decision_function(self._transform(X)))
        
    def setFeatureConfidenceEstimator(self, feature, fc):
        self._FCEs[feature] = fc
        return(self)

    def getConfidence(self, X):
        result = 1.0
        feature_weights = self.getClassifierFeatureWeights()
        weight_sum = 0.0
        for key, fc in self._FCEs.items():
            result += fc.getConfidence(X) * abs(feature_weights[key])
            weight_sum += abs(feature_weights[key])
        return((result / weight_sum) * np.max(self.predict_proba(X), axis=1))
        
    def getAllFeatures(self):
        features = self.getClassifierFeatures()
        return(np.union1d(features, self._second_layer_features))

    def getClassifierFeatures(self):
        raise NotImplementedError()

    def getClassifierFeatureWeights(self):
        raise NotImplementedError()


class LogisticRegressionClassifier(BaseWeakClassifier):
    def __init__(self, n_jobs = 1,
                 excluded_features=None,
                 feature_confidence_estimator=PredictBasedFCE(),
                 C = 0.2):
        learner = LogisticRegression(penalty = 'l1',
                                     dual = False,
                                     C = C,
                                     fit_intercept = True)
        super(LogisticRegressionClassifier, self).__init__(
            learner, n_jobs, excluded_features, feature_confidence_estimator)

    def get_params(self, deep=True):
        return( {
            'C':self.learner.get_params()['C'],
            'excluded_features':self.excluded_features,
            'feature_confidence_estimator':self.feature_confidence_estimator,
            'n_jobs':self.n_jobs})

    def set_params(self, **parameters):
        self.__init__(**parameters)
        return(self)

    def getClassifierFeatures(self):
        scores = self.learner.coef_.flatten()
        local_cols = np.arange(scores.shape[0])[(scores != 0),]
        #print('lc', local_cols)
        #print('ef', self.excluded_features)
        #print('xc', self._X_colcount)
        return(np.delete(np.arange(self._X_colcount),
                         self.excluded_features)[local_cols])

    def getClassifierFeatureWeights(self):
        scores = self.learner.coef_.flatten()
        scores = scores[(scores != 0),]
        features = self.getClassifierFeatures()
        return(dict([(features[i],scores[i]) for i in range(len(scores))]))

class Rat(BaseEstimator, LinearClassifierMixin):
    def __init__(self):
        pass
        
    def __init__(self, 
                 learner_count=10,
                 learner=None,
                 overlapping_features=False):
        
        try:
            learner_count = int(learner_count)
        except:
            raise TypeError("learner_count should be an int")

        if (not isinstance(overlapping_features, bool)):
            raise TypeError("overlapping_features should be a Boolean.")

        self.learner_count = learner_count
        if (learner == None):
            self.learner = LogisticRegressionClassifier(n_jobs = 1)
        else:
            self.learner = learner
        self.overlapping_features = overlapping_features
        self.learners = []
        self.excluded_features = np.empty(0, dtype=int)

    def get_params(self, deep=True):
        return( {
            'learner_count':self.learner_count,
            'learner':self.learner,
            'overlapping_features':self.overlapping_features})

    def set_params(self, **parameters):
        self.__init__(**parameters)
        return(self)

    def getSingleLearner(self, X, y):
        for i in range(5):
            rs = cv.ShuffleSplit(n=X.shape[0], n_iter=1,
                                 train_size=0.9)
            l = sklearn.clone(self.learner)
            
            for train_index, test_index in rs:
                if (self.overlapping_features):
                    l.excluded_features = np.empty(0, dtype=int)
                else:
                    l.excluded_features = np.copy(self.excluded_features)
            l.fit(X[train_index,], y[train_index,])
            if (len(list(l.getClassifierFeatures())) > 0):
                return (l)
        raise(RuntimeError("Tried 5 times to fit a learner, all chose no features."))
        
    def fit(self, X, y):
        #print(self)
        self.X = X.view(np.ndarray)
        self.y = y.view(np.ndarray).squeeze()

        self.classes_, y = sklearn.utils.fixes.unique(y, return_inverse=True)

        self.learners = []
        self.excluded_features = np.empty(0, dtype=int)
        for i in range(self.learner_count):
            tmp = self.getSingleLearner(X, y)
            self.learners.append(tmp)
            #print(tmp.getFeatures())
            self.excluded_features = np.union1d(
                self.excluded_features, tmp.getAllFeatures())
            if (self.excluded_features.shape[0] > (X.shape[1] / 5)):
                break
        return(self)
    '''            
    def predict(self, X, return_details = False):
        D = self.decision_function(X)
        return self.classes_[np.argmax(D, axis=1)]
    '''
    def decision_function(self, X, return_details=False):

        predictions = np.empty((X.shape[0],0), dtype=float)
        confidences = np.empty((X.shape[0],0), dtype=float)
        for l in self.learners:
            predictions = np.hstack((predictions,
                                     l.decision_function(X).reshape(-1,1)))
            confidences = np.hstack((confidences,
                                     l.getConfidence(X).reshape(-1,1)))

        if (len(self.learners) > 1):
            #result = predictions[max(enumerate(confidences),key=lambda x: x[1])[0]]
            result = np.average(predictions, weights=confidences, axis=1)
        else:
            result = predictions
        if (return_details):
            return((result, predictions, confidences))
        else:
            return(result)
        
'''            
predictions = np.empty((97,0), dtype=float)
confidences = np.empty((97,0), dtype=float)
for l in rat.learners:
    predictions = np.hstack((predictions, l.decision_function(tmpX).reshape(-1,1)))
    confidences = np.hstack((confidences, l.getConfidence(tmpX).reshape(-1,1)))

if (len(self.learners) > 1):
    #result = predictions[max(enumerate(confidences),key=lambda x: x[1])[0]]
    result = np.average(predictions, weights=confidences)
else:
    result = predictions
if (return_details):
    return((result, predictions, confidences))
else:
    return(result)


l = rat.learners[0]
result = 1.0
feature_weights = l.getClassifierFeatureWeights()
weight_sum = 0.0
for key, fc in l._FCEs.items():
    result += fc.getConfidence(tmpX) * abs(feature_weights[key])
    weight_sum += abs(feature_weights[key])
res = ((result / weight_sum) * np.max(l.predict_proba(tmpX), axis=1))
'''

    
def rat_parameter_search(X, y, weak_learner_count = [5, 10, 15],
                         second_leayer_feature_count = [5, 10],
                         regularization_parameter = [0.2, 0.3, 0.4]):

    train_auc = list()
    test_auc = list()
    for wlc in weak_learner_count:
        for slfc in second_layer_feature_count:
            for rp in regularization_parameter:
                weak_learner = LogisticRegressionClassifier(
                    C = rp,
                    second_layer_feature_count = slfc)
                rat_model = Rat(learner_count = wlc,
                                learner = weak_learner,
                                n_jobs = 30)
                scores = cv.cross_val_score(
                    rat_model, tmpX, y,
                    cv=5, scoring = 'roc_auc', n_jobs=5)
                rat_model.fit(train_data, train_labels)
                out = rat_model.predict(train_data)
                out_test = rat_model.predict(test_data)
                train_auc.append((wlc, slfc, rp,
                                  roc_auc_score(train_labels, out)))
                test_auc.append((wlc, slfc, rp,
                                 roc_auc_score(test_labels, out_test)))
