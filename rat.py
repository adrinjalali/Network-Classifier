''' This is the bootstrap - network existance assumed - basian inspired
    classifier thing.
    License: GPLv3.
    Adrin Jalali.
    Jan 2014, Saarbruecken, Germany.
'''
from __future__ import print_function

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model.base import LinearClassifierMixin
import sklearn.base
from sklearn.linear_model import LogisticRegression
import sklearn.cross_validation as cv
import sklearn.gaussian_process as gp
import sklearn.svm
from minepy import MINE
import heapq
import numpy as np
from joblib import Parallel, delayed
import copy
import sys
import math


class utilities:
    def exclude_cols(X, cols):
        ''' exludes indices in cols, from columns of X '''
        return(X[:,~np.in1d(np.arange(X.shape[1]), cols)])

    def check_1d_array(a):
        return(isinstance(a, np.ndarray) and (a.ndim == 1 or
                                              (a.ndim == 2 and
                                              (a.shape[0] == 1 or
                                              a.shape[1] == 1))))

def _evaluate_single(data, target_feature):
    mine = MINE(alpha=0.4, c=15)
    MICs = list()
    for i in range(data.shape[1]):
        mine.compute_score(target_feature,data[:,i])
        MICs.append(mine.mic())
    return(MICs)
class SecondLayerFeatureEvaluator:
        
    def evaluate(self, data, target_features, n_jobs = 1):
            
        result = np.zeros((target_features.shape[1], data.shape[1]))
        print(n_jobs, file=sys.stderr)
        if (n_jobs == 1):
            for i in range(target_features.shape[1]):
                result[i,:] = _evaluate_single(data, target_features[:,i])
                
        elif (n_jobs > 1):
            result = Parallel(n_jobs = n_jobs, backend='threading',
                              max_nbytes=None, pre_dispatch='3*n_jobs')(
                delayed(_evaluate_single)
                (data, target_features[:,i]) for i in range(target_features.shape[1]))
        print('done SLFE', file=sys.stderr)
        return(result)

class PredictBasedFCE(BaseEstimator):
    ''' This class uses Gaussian Processes as the regression
    algorithm. It uses Mutual Information to select features
    to give to the GP, and at the end uses GP's output, compared
    to the observed value, and the predicted_MSE of the GP, to
    calculate the confidence.
    '''
    def __init__(self, feature_count=5):
        self._learner = gp.GaussianProcess(nugget=1e-2)
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
            print("feature should be int", file=sys.stderr)
            raise TypeError("feature should be int")

        if (isinstance(excluded_features, set) or
            isinstance(excluded_features, list)):
            excluded_features = np.array(list(excluded_features), dtype=int)
            
        if not utilities.check_1d_array(excluded_features):
            print("excluded features should be 1d ndarray", file=sys.stderr)
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
        res = (np.arange(len(scores))[scores >
                                           np.max(scores) * 0.90])
        if (res.shape[0] < 5):
            res = (np.array([t[0] for t in heapq.nlargest(5,
                                                          enumerate(scores),
                                                          lambda t:t[1])]))
        #print(res)
        return(res)

    def predict(self, X):
        X = X.view(np.ndarray)
        if (X.ndim == 1):
            X = X.reshape(1, -1)
        
        my_X = X[:,self.getFeatures()]
        return(self._learner.predict(my_X))
        
    def getConfidence(self, X):
        def phi(x): return(0.5 + 0.5 * math.erf(x / math.sqrt(2)))
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
        
        if (excluded_features is None):
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
    
    def fit(self, X, y, scores = None):
        self._X_colcount = X.shape[1]
        #self.learner.fit(self._transform(X), y)
        learner = self.get_learner(X, y)
        if (not hasattr(learner, 'predict_proba')) and \
            hasattr(learner, 'decision_function'):
            decision_values = self.decision_function(X)
            if (decision_values.ndim > 1):
                raise("don't know what to do")
            self.df_var = np.var(decision_values)
            self.df_mean = np.mean(decision_values)
            print("df_var, df_mean: %g, %g" % (self.df_var, self.df_mean), file=sys.stderr)

        # no GP
        # return(self)

        classifier_features = self.getClassifierFeatures()

        fe = SecondLayerFeatureEvaluator()
        local_excluded_features = np.union1d(self.excluded_features,
                                             classifier_features)
        local_X = utilities.exclude_cols(X,
                                         local_excluded_features)
        if (scores is None):
            scores = fe.evaluate(local_X, X[:,classifier_features],
                             n_jobs = self.n_jobs)
        else:
            scores = [scores[i][~np.in1d(np.arange(len(scores)), local_excluded_features)] for i in range(len(scores))]

        print('fitting GPs', file=sys.stderr)
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
        if (hasattr(self.learner, 'predict_proba')):
            return(self.learner.predict_proba(self._transform(X)))

    def predict_log_proba(self, X):
        if (hasattr(self.learner, 'predict_log_proba')):
            return(self.learner.predict_log_proba(self._transform(X)))

    def decision_function(self, X):
        res = self.learner.decision_function(self._transform(X))
        if (utilities.check_1d_array(res)):
            res = res.reshape(-1)
        return(res)
        
    def setFeatureConfidenceEstimator(self, feature, fc):
        self._FCEs[feature] = fc
        return(self)

    def getConfidence(self, X, per_feature = False):
        def phi(x): return(0.5 + 0.5 * math.erf(x / math.sqrt(2)))
        def my_score(x): return(abs(phi(x) - phi(-x)))

        # no GP
        # return 1
            
        X = X.view(np.ndarray)

        if (X.ndim == 1):
            X = X.reshape(1,-1)
            
        #res = []
        #for i in range(len(X)):
            #result = 1.0
        result = np.ones(len(X))
        per_feature_conf = dict()
        feature_weights = self.getClassifierFeatureWeights()
        weight_sum = sum(np.abs(list(feature_weights.values())))
        for key, fc in self._FCEs.items():
            tmp = fc.getConfidence(X) * abs(feature_weights[key])
            per_feature_conf[key] = tmp
            result += tmp
        if (not per_feature):
            return (result / weight_sum)
        else:
            return dict([(key,value/weight_sum) for key, value in per_feature_conf.items()])
        #res.append(result/weight_sum)
        '''
        if (hasattr(self.learner, 'predict_proba')):
            res.append((result / weight_sum) * \
                my_score(np.max(self.predict_proba(X[[i],]))))
        else:
            decision_value_normalized = \
                  (self.decision_function(X[[i],]) - self.df_mean) / \
                  math.sqrt(self.df_var)
            res.append((result / weight_sum) \
                           * my_score(decision_value_normalized))
        '''
        #return(np.array(res).reshape(-1))
        
    def getAllFeatures(self):
        features = self.getClassifierFeatures()
        return(features)
        #return(np.union1d(features, self._second_layer_features))

    def getClassifierFeatures(self):
        print("getClassifierFeatures(self): not implemented")
        raise NotImplementedError()

    def getClassifierFeatureWeights(self):
        print("getClassifierFeatureWeights(self): not implemented")
        raise NotImplementedError()

    def set_params(self, **parameters):
        self.__init__(**parameters)
        return(self)

class LogisticRegressionClassifier(BaseWeakClassifier):
    def __init__(self, n_jobs = 1,
                 excluded_features=None,
                 feature_confidence_estimator=PredictBasedFCE(),
                 regularizer_index = 10):
        learner = LogisticRegression(penalty = 'l1',
                                     dual = False,
                                     fit_intercept = True)
        self.regularizer_index = regularizer_index
        super(LogisticRegressionClassifier, self).__init__(
            learner, n_jobs, excluded_features, feature_confidence_estimator)

    def get_params(self, deep=True):
        return( {
            'regularizer_index':self.regularizer_index,
            'excluded_features':self.excluded_features,
            'feature_confidence_estimator':self.feature_confidence_estimator,
            'n_jobs':self.n_jobs})

    def get_learner(self, X, y):
        local_X = self._transform(X)
        index = self.regularizer_index
        cs = sklearn.svm.l1_min_c(local_X, y, loss='log') * np.logspace(0,5)
        while (index < len(cs)):
            self.learner.set_params(C = cs[index])
            self.learner.fit(local_X, y)
            if (len(self.getClassifierFeatures()) > 0):
                return(self.learner)
            index += 5
            print("index: %d" % (index), file=sys.stderr)
        return(self.learner)
    
    def getClassifierFeatures(self):
        scores = self.learner.coef_.flatten()
        local_cols = np.arange(scores.shape[0])[(scores != 0),]
        return(np.delete(np.arange(self._X_colcount),
                         self.excluded_features)[local_cols])

    def getClassifierFeatureWeights(self):
        scores = self.learner.coef_.flatten()
        scores = scores[(scores != 0),]
        features = self.getClassifierFeatures()
        return(dict([(features[i],scores[i]) for i in range(len(scores))]))
    '''
    def getClassifierFeatures(self):
        scores = self.learner.coef_.flatten()
        threshold = (np.max(abs(scores)) - np.min(abs(scores))) * 0.80
        local_cols = np.arange(scores.shape[0])[abs(scores) > threshold]
        return(np.delete(np.arange(self._X_colcount),
                         self.excluded_features)[local_cols])

    def getClassifierFeatureWeights(self):
        scores = self.learner.coef_.flatten()
        threshold = (np.max(abs(scores)) - np.min(abs(scores))) * 0.80
        local_cols = np.arange(scores.shape[0])[abs(scores) > threshold]
        scores = scores[local_cols,]
        features = self.getClassifierFeatures()
        return(dict([(features[i],scores[i]) for i in range(len(scores))]))
    '''

class LinearSVCClassifier(BaseWeakClassifier):
    def __init__(self, n_jobs = 1,
                 excluded_features=None,
                 feature_confidence_estimator=PredictBasedFCE(),
                 regularizer_index = 10):
        learner = sklearn.svm.LinearSVC(penalty = 'l1',
                                        dual = False)
        self.regularizer_index = regularizer_index
        self.min_C = None
        super(LinearSVCClassifier, self).__init__(
            learner, n_jobs, excluded_features, feature_confidence_estimator)

    def get_params(self, deep=True):
        return( {
            'regularizer_index':self.regularizer_index,
            'excluded_features':self.excluded_features,
            'feature_confidence_estimator':self.feature_confidence_estimator,
            'n_jobs':self.n_jobs})
        
    def get_min_C(self, X, y):
        high = 1/1000
        low = 0
        while (True):
            learner = sklearn.svm.LinearSVC(penalty = 'l1',
                dual = False, C = high, verbose=False)
            learner.fit(X, y)
            t_fc = np.sum(learner.coef_ != 0)
            if (t_fc > 1):
                high = (low + high) / 2
            elif (t_fc < 1):
                low = high
                high = high * 2
            else:
                break
        print('selected C: %g' % (high), file=sys.stderr)
        return high
    
    def get_learner(self, X, y):
        local_X = self._transform(X)
        index = self.regularizer_index
        if (self.min_C is None):
            self.min_C = self.get_min_C(local_X, y)
        cs = self.min_C * np.logspace(0,1)
        while (index < len(cs)):
            self.learner.set_params(C = cs[index])
            self.learner.fit(local_X, y)
            if (len(self.getClassifierFeatures()) > 0):
                print('training nusvm', file=sys.stderr)
                print(self.getClassifierFeatures(), file=sys.stderr)
                local_cols = abs(self.learner.coef_.flatten()) > 0
                #print(local_cols[0:50])
                #print(sum(local_cols))
                to_zero = np.arange(local_X.shape[1])[local_cols == False]
                #print(to_zero[0:50])
                tmp_X = np.copy(local_X)
                tmp_X[:,to_zero] = 0

                for nu in np.hstack((np.arange(.2, .0, -.05), np.array([.4,.3,.2,.1,.01,.001]))):
                    try:
                        self.predictor = sklearn.svm.NuSVC(nu=nu,
                                        kernel='linear',
                                        verbose=False,
                                        probability=True)
                        self.predictor.fit(tmp_X, y)
                        break
                    except Exception as e:
                        #print(nu, e, file=sys.stderr)
                        pass
                    
                print('nu: ', nu, file=sys.stderr, flush=True)
                        
                self.learner = self.predictor
                #print(self.getClassifierFeatures())
                #return(self.learner)
                return(self.predictor)
            index += 1
            print(index, cs, file=sys.stderr)
        return(self.learner)

    def getClassifierFeatures(self):
        scores = self.learner.coef_.flatten()
        threshold = np.min(abs(scores)) + (
            np.max(abs(scores)) - np.min(abs(scores))) * 0.0
        local_cols = np.arange(scores.shape[0])[abs(scores) > threshold]
        return(np.delete(np.arange(self._X_colcount),
                         self.excluded_features)[local_cols])

    def getClassifierFeatureWeights(self):
        scores = self.learner.coef_.flatten()
        threshold = np.min(abs(scores)) + (
            np.max(abs(scores)) - np.min(abs(scores))) * 0.0
        local_cols = np.arange(scores.shape[0])[abs(scores) > threshold]
        scores = scores[local_cols,]
        features = self.getClassifierFeatures()
        return(dict([(features[i],scores[i]) for i in range(len(scores))]))

class NuSVCClassifier(BaseWeakClassifier):
    def __init__(self, n_jobs = 1,
                 excluded_features=None,
                 feature_confidence_estimator=PredictBasedFCE(feature_count=10)):
        learner = sklearn.svm.NuSVC(nu = 0.25,
                                    kernel = 'linear',
                                    probability = True)
        super(NuSVCClassifier, self).__init__(
            learner, n_jobs, excluded_features, feature_confidence_estimator)

    def get_params(self, deep=True):
        return( {
            'excluded_features':self.excluded_features,
            'feature_confidence_estimator':self.feature_confidence_estimator,
            'n_jobs':self.n_jobs})

    def get_learner(self, X, y):
        local_X = self._transform(X)
        self.learner.fit(local_X, y)
        return(self.learner)

    def getClassifierFeatures(self):
        scores = self.learner.coef_.flatten()
        local_cols = np.arange(scores.shape[0])[
            abs(scores) > 0.80 * np.max(abs(scores))]        
        return(np.delete(np.arange(self._X_colcount),
                         self.excluded_features)[local_cols])

    def getClassifierFeatureWeights(self):
        scores = self.learner.coef_.flatten()
        local_cols = np.arange(scores.shape[0])[
            abs(scores) > 0.80 * np.max(abs(scores))]        
        scores = scores[local_cols,]
        features = self.getClassifierFeatures()
        return(dict([(features[i],scores[i]) for i in range(len(scores))]))

class Rat(BaseEstimator, LinearClassifierMixin):
    def __init__(self):
        pass
        
    def __init__(self, 
                 learner_count=10,
                 learner_type='linear svc',
                 overlapping_features=False,
                 regularizer_index = None,
                 feature_cluster_data = None,
                 n_jobs = 1):
        
        self.learner_count = learner_count
        self.learner_type = learner_type
        self.overlapping_features = overlapping_features
        self.regularizer_index = regularizer_index
        self.feature_cluster_data = feature_cluster_data
        self.n_jobs = n_jobs

    def get_params(self, deep=True):
        return( {
            'learner_count':self.learner_count,
            'learner_type':self.learner_type,
            'overlapping_features':self.overlapping_features,
            'regularizer_index':self.regularizer_index,
            'feature_cluster_data':self.feature_cluster_data,
            'n_jobs':self.n_jobs})
            
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
            if (self.feature_cluster_data is not None):
                scores = [(self.feature_cluster_data == self.feature_cluster_data[i]).astype('int') for i in
                          range(len(self.feature_cluster_data))]
            else:
                scores = None
            l.fit(X, y, scores)
            if (len(list(l.getClassifierFeatures())) > 0):
                return (l)
        print("Tried 5 times to fit a learner, all chose no features.\
              len(learners): %d" % (len(self.learners)), file=sys.stderr)
        raise(RuntimeError("Tried 5 times to fit a learner, all chose no features."))
        
    def fit(self, X, y, from_scratch = True):
        X = X.view(np.ndarray)
        y = y.view(np.ndarray).squeeze()

        if (from_scratch == True):
            try:
                learner_count = int(self.learner_count)
            except:
                print("learner_count should be an int", file=sys.stderr)
                raise TypeError("learner_count should be an int")

            if (not isinstance(self.overlapping_features, bool)):
                print("overlapping_features should be a Boolean.", file=sys.stderr)
                raise TypeError("overlapping_features should be a Boolean.")

            if (self.learner_type == 'logistic regression'):
                if (self.regularizer_index == None):
                    self.regularizer_index = 15
                self.learner = LogisticRegressionClassifier(
                    regularizer_index = self.regularizer_index,
                    n_jobs = self.n_jobs)
            elif (self.learner_type == 'linear svc'):
                if (self.regularizer_index == None):
                    self.regularizer_index = 15
                self.learner = LinearSVCClassifier(
                    regularizer_index = self.regularizer_index,
                    n_jobs = self.n_jobs)
            elif (self.learner_type == 'nu svc'):
                self.learner = NuSVCClassifier(n_jobs = self.n_jobs)
            else:
                print("learner_type must be in ('logistic regression',\
                                   'linear svc', 'nu svc')", file=sys.stderr)
                raise RuntimeError("learner_type must be in ('logistic regression',\
                                   'linear svc', 'nu svc')")

            self.classes_, y = np.unique(y, return_inverse=True)

            self.learners = []
            self.excluded_features = np.empty(0, dtype=int)
            for i in range(self.learner_count):
                self.add_learner(X, y)
                if (self.excluded_features.shape[0] > (X.shape[1] / 5)):
                    break
        else:
            self.add_learner(X, y)
            
        return(self)

    def add_learner(self, X, y):
        print("adding learner %d" %(len(self.learners)), file=sys.stderr)
        if (hasattr(self, 'single_learner_failed')):
            print("I'm not stupid, won't try again, learner_count: \
                  %d" % (self.learner_count), file=sys.stderr)
            return(self)
            
        X = X.view(np.ndarray)
        y = y.view(np.ndarray).squeeze()
        try:
            tmp = self.getSingleLearner(X, y)
        except:
            print(sys.exc_info(), file=sys.stderr)
            self.single_learner_failed = True
            return(self)
        self.learners.append(tmp)
        self.excluded_features = np.union1d(
            self.excluded_features, tmp.getAllFeatures())
        self.learner_count = len(self.learners)
        return(self)

    '''            
    def predict(self, X, return_details = False):
        D = self.decision_function(X)
        return self.classes_[np.argmax(D, axis=1)]
    '''
    def decision_function(self, X, return_details=False, return_iterative=False):
        #global predictions, confidences
        X = X.view(np.ndarray)

        if (X.ndim == 1):
            X = X.reshape(1,-1)

        predictions = np.empty((len(self.learners), X.shape[0],2), dtype=float)
        confidences = np.empty((len(self.learners), X.shape[0]), dtype=float)
        if (return_iterative):
            iterative_result = list()
           
        i = 0;
        for l in self.learners:
            predictions[i, ] = l.predict_log_proba(X)
            confidences[i, ] = l.getConfidence(X)
            i = i + 1;
            
            if(return_iterative):
                if (len(iterative_result) > 0):
                    result = np.sum(np.multiply(predictions[0:i,:,1], confidences[0:i,]), axis=0) - \
                        np.sum(np.multiply(predictions[0:i,:,0], confidences[0:i,]), axis=0)
                    #result = np.average(predictions[0:i,:,1], weights=confidences[0:i,], axis=0) - \
                    #    np.average(predictions[0:i,:,0], weights=confidences[0:i,], axis=0)
                else:
                    result = (predictions[0:i,:,1] - predictions[0:i, :, 0]).reshape(-1)
                if (return_details):
                    result = (result, predictions, confidences)
                iterative_result.append(result)


        if (return_iterative):
            return iterative_result

        if (len(self.learners) > 1):
            #result = predictions[max(enumerate(confidences),key=lambda x: x[1])[0]]
            #result = np.mean(predictions * confidences, axis=1)
            result = np.average(predictions[:,:,1], weights=confidences, axis=0) - \
                np.average(predictions[:,:,0], weights=confidences, axis=0)
        else:
            result = predictions
        if (return_details):
            return((result, predictions, confidences))
        else:
            return(result)
        
    def feature_importances(self, sample = None):
        result = dict()
        for l in self.learners:
            if (sample is None):
                tmp = l.getClassifierFeatureWeights()
            else:
                tmp = l.getConfidence(sample, per_feature = True)
                
            for key, value in tmp.items():
                if key in result:
                    print('%d already in result set, this is a BUG' % (key), file=sys.stderr)
                    print(l, file=sys.stderr)
                    print(result, file=sys.stderr)
                    return None
                else:
                    result[key] = value
        return result
