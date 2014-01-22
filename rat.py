''' This is the bootstrap - network existance assumed - basian inspired
    classifier thing.
    License: GPLv3.
    Adrin Jalali.
    Jan 2014, Saarbruecken, Germany.
'''
 
import sklearn.base
from sklearn.linear_model import LogisticRegression
import sklearn.cross_validation as cv
import sklearn.gaussian_process as gp
from minepy import MINE
import heapq
import numpy as np

class utilities:
    def exclude_cols(X, cols):
        ''' exludes indices in cols, from columns of X '''
        return(X[:,~np.in1d(np.arange(X.shape[1]), cols)])

    def check_1d_array(a):
        return(isinstance(a, np.ndarray) and (a.ndim == 1 or
                                              a.shape[0] == 1 or
                                              a.shape[1] == 1))

class AbstractClassException(Exception):
    pass

class AbstractClass:
    def __init__(self):
        raise AbstractClassException()

class BaseFeatureConfidenceEstimator(AbstractClass):
    '''Evaluate validity of a single feture
    Every Confidence Estimator should inherit this class.'''
    def initialize(self, X, feature, excluded_features):
        if (not isinstance(X, np.ndarray)):
            raise TypeError("X should be ndarray")
            
        try:
            feature = int(feature)
        except Exception:
            raise TypeError("feature should be int")

        if (isinstance(excluded_features, set) or
            isinstance(excluded_features, list)):
            excluded_features = np.array(list(excluded_features), dtype=int)
            
        if not utilities.check_1d_array(excluded_features):
            raise TypeError("excluded features should be 1d ndarray")

        self.X = X.view(np.ndarray)
        self.feature = feature
        self.excluded_features = np.copy(np.hstack((excluded_features, feature)))
        self.my_X = utilities.exclude_cols(self.X, self.excluded_features)
        self._initialized = True
        self._trained = False

    def _checkTrained(self):
        if (not self._trained):
            raise Exception("The model is not trained.")

    def getConfidence(self, sample):
        raise NotImplementedError()

    def train(self):
        raise NotImplementedError()

    def getFeatures(self):
        raise NotImplementedError()
        
    def clone_init(self):
        raise NotImplementedError()

class PredictBasedFCE(BaseFeatureConfidenceEstimator):
    ''' This class uses Gaussian Processes as the regression
    algorithm. It uses Mutual Information to select features
    to give to the GP, and at the end uses GP's output, compared
    to the observed value, and the predicted_MSE of the GP, to
    calculate the confidence.
    '''
    def __init__(self):
        self._learner = gp.GaussianProcess(theta0=1e-2, thetaL=1e-4, thetaU=1e-1)
        
    def train(self, feature_count = 10):
        self._selected_features = self._selectFeatures(k = feature_count)
        self._learner.fit(self.my_X[:,self._selected_features],
                          self.X[:,self.feature])
        self._trained = True
        
    def _selectFeatures(self, k = 10):
        ''' computes mutual information of all features with
        the target feature. Note that excluded_features and the
        target feature are excluded from self.X in initialization.
        Then returns k features of corresponding to most MICs.
        The precision can be improved by increasing alpha of the
        MINE object, but it drastically increases the computation
        time, and the ordering of the features doesn't change much.
        '''
        mine = MINE(alpha=0.3, c=15)
        MICs = list()
        for i in range(self.my_X.shape[1]):
            mine.compute_score(self.X[:,self.feature],self.my_X[:,i])
            MICs.append(mine.mic())
        return([t[0] for t in heapq.nlargest(k,
                                             enumerate(MICs),
                                             lambda t:t[1])])

    def predict(self, sample):
        self._checkTrained()
        
        #if (not utilities.check_1d_array(sample)):
        #    raise TypeError("sample should be 1d ndarray")

        sample = sample.view(np.ndarray)
        if (sample.ndim == 1):
            sample = sample.reshape(1, -1)
        
        my_sample = sample[:,self.getFeatures()]
        return(self._learner.predict(my_sample))
        
    def getConfidence(self, sample):
        self._checkTrained()
        #if (not utilities.check_1d_array(sample)):
        #    raise TypeError("sample should be 1d ndarray")

        sample = sample.view(np.ndarray)
        if (sample.ndim == 1):
            sample = sample.reshape(1, -1)
        
        my_sample = sample[:,self.getFeatures()]
        y_pred, sigma2_pred = self._learner.predict(my_sample, eval_MSE=True)
        return(sigma2_pred / (abs(y_pred - sample[:,self.feature]) + sigma2_pred))

    '''
    Using Gaussian Processes, there is no need for this function.
    def calculateGeneralConfidence(self):
        cols = self.getContributingFeatures()
        local_X = self.X[:,cols]
        rs = cv.ShuffleSplit(self.my_X.shape[0],
                             n_iter = self.cross_validation_count,
                             test_size = 0.2, indices = True)
        test_aucs = list()
        for train_idx, test_idx in rs:
            self._learner.fit(local_X[train_idx,],
                              self.X[:,self.feature][train_idx,:])
            test_out = self._learner.predict(local_X[test_idx,])
            test_aucs.append(roc_auc_score(self.X[:,feature][test_idx,:], test_out))

        self.general_confidence = np.mean(test_aucs)
    '''

    def getFeatures(self):
        self._checkTrained()
        local_cols = self._selected_features
        return(np.delete(np.arange(self.X.shape[1]),
                         [self.excluded_features])[local_cols])

    def clone_init(self):
        other = self.__class__()
        other._learner = sklearn.base.clone(self._learner)
        if (hasattr(self, '_initialized')):
            other.initialize(self.X, self.feature, self.excluded_features)
        return(other)

    def __str__(self):
        result = "### %s\n" % (self.__class__)

        if (hasattr(self, '_initialized')):
            result += "X (shape: %s):\n" % (self.X.shape.__str__())
            result += self.X.__str__() + "\n"

            result += "excluded_features(%s):\n%s\n" %(
                self.excluded_features.shape.__str__(),
                self.excluded_features.__str__())

            result += "feature: %d\n" % (self.feature)
            
            result += "my_X (shape: %s):\n" % (self.my_X.shape.__str__())
            result += self.my_X.__str__() + "\n"

            result += self._learner.__str__()

        return(result)
        

class WeakClassifier(AbstractClass):
    def _train(self, X, y):
        self._learner.fit(X, y)

    def _predict(self, sample):
        return(self._learner.predict(sample))
    
    def initialize(self, X, y, sample_names=None, feature_names=None,
                   excluded_features=None):
        self.X = X.view(np.ndarray)
        self.y = y.view(np.ndarray).squeeze()
        self.sample_names = sample_names
        self.feature_names = feature_names

        if (isinstance(excluded_features, set) or
            isinstance(excluded_features, list)):
            excluded_features = np.array(list(excluded_features), dtype=int)
        
        if (excluded_features == None):
            self.excluded_features = np.empty(0, dtype=int)
        else:
            self.excluded_features = np.copy(excluded_features)

        self.my_X = utilities.exclude_cols(self.X, self.excluded_features)
        self._FCEs = dict()
        self._initialized = True
        self._trained = False

    def _checkTrained(self):
        if (not self._trained):
            raise Exception("The model is not trained.")
        
    def train(self):
        self._train(self.my_X, self.y)
        self._trained = True

    def predict(self, sample):
        self._checkTrained()
        #if (not utilities.check_1d_array(sample)):
        #    raise TypeError("sample should be 1d ndarray")

        sample = sample.view(np.ndarray)
        if (sample.ndim == 1):
            sample = sample.reshape(1, -1)
        
        my_sample = np.delete(sample, self.excluded_features)
        return(self._predict(my_sample)[0])
        
    def setFeatureConfidenceEstimator(self, feature, fc):
        self._FCEs[feature] = fc

    def getConfidence(self, sample):
        #if (not utilities.check_1d_array(sample)):
        #    raise TypeError("sample should be 1d ndarray")

        sample = sample.view(np.ndarray)
        if (sample.ndim == 1):
            sample = sample.reshape(1, -1)
        
        result = 1.0
        for key, fc in self._FCEs.items():
            result *= fc.getConfidence(sample)
        return(result[0])
        
    def clone_init(self):
        other = self.__class__()
        other._learner = sklearn.base.clone(self._learner)
        if (hasattr(self, '_initialized')):
            other.initialize(self.X, self.y, self.sample_names,
                             self.feature_names, self.excluded_features)
        return(other)

    def getFeatures(self):
        raise NotImplementedError()

    def __str__(self):
        result = "### %s\n" % (self.__class__);

        if (hasattr(self, '_initialized')):
            result += "X (%s)\n" % (self.X.shape.__str__())
            result += self.X.__str__() + "\n"

            result += "y (%s)\n" % (self.y.shape.__str__())
            result += self.y.__str__() + "\n"
        
            if (self.sample_names != None):
                result += "sample names (%s)\n%s\n" % (
                    self.sample_names.shape.__str__(),
                    self.sample_names.__str__())

            if (self.feature_names != None):
                result += "feature names (%s)\n%s\n" % (
                    self.feature_names.shape.__str__(),
                    self.feature_names.__str__())
        
            result += "excluded_features (%s):\n%s\n" % (
                self.excluded_features.shape.__str__(),
                self.excluded_features.__str__())

            result += "my_X (%s)\n%s\n" % (
                self.my_X.shape.__str__(),
                self.my_X.__str__())
        
            result += "FCEs (%s)\n" % (len(self._FCEs))
            for v in self._FCEs.values():
                result += v.__str__() + "\n"

        return(result)

class LogisticRegressionClassifier(WeakClassifier):
    def __init__(self, C = 0.2):
        self._learner = LogisticRegression(penalty = 'l1',
                                           dual = False,
                                           C = C,
                                           fit_intercept = True)
        
        '''self._learner = RandomizedLogisticRegression(C = 1,
                                                     sample_fraction = 0.8,
                                                     n_resampling = 1,
                                                     selection_threshold = 0.25,
                                                     fit_intercept = True,
                                                     verbose = True,
                                                     normalize = False,
                                                     n_jobs = 1)'''
        
    def getFeatures(self):
        self._checkTrained()
        scores = self._learner.coef_.flatten()
        local_cols = np.arange(scores.shape[0])[(scores != 0),]
        return(np.delete(np.arange(self.X.shape[1]),
                         [self.excluded_features])[local_cols])

    def __str__(self):
        result = super(LogisticRegressionClassifier, self).__str__() + "\n"
        result += self._learner.__str__()
        return(result)


class Rat:
    def __init__(self, X, y, feature_names=None, sample_names=None,
                 learner_count=10,
                 learner=LogisticRegressionClassifier(),
                 feature_confidence_estimator=PredictBasedFCE(),
                 overlapping_features=False):
        if ((not isinstance(X, np.ndarray)) or
            X.ndim != 2):
            raise TypeError("X should be 2d ndarray")
        if (not utilities.check_1d_array(y)):
            raise TypeError("Y should be 1d ndarray")
        try:
            learner_count = int(learner_count)
        except:
            raise TypeError("learner_count should be an int")
        
        if (not isinstance(learner, WeakClassifier)):
            raise TypeError("learner should be of WeakClassifier")

        if (not isinstance(feature_confidence_estimator,
                           BaseFeatureConfidenceEstimator)):
            raise TypeError("feature_confidence_estimator should be of\
                            BaseFeatureConfidenceEstimator")

        if (not isinstance(overlapping_features, bool)):
            raise TypeError("overlapping_features should be a Boolean.")

        if (feature_names == None):
            self.feature_names = np.arange(X.shape[1])
        else:
            self.feature_names = feature_names

        if (sample_names == None):
            self.sample_names = np.arange(X.shape[0])
        else:
            self.sample_names = sample_names

        self.X = X.view(np.ndarray)
        self.y = y.view(np.ndarray).squeeze()
        self.learner_count = learner_count
        self.learner = learner
        self.feature_confidence_estimator = feature_confidence_estimator
        self.overlapping_features = overlapping_features
        self.learners = []
        self.excluded_features = set()
        self._trained = False

    def _checkTrained(self):
        if (not self._trained):
            raise Exception("The model is not trained.")

    def getSingleLearner(self):
        rs = cv.ShuffleSplit(n=self.X.shape[0], n_iter=1,
                             train_size=0.8, indices=True)
        l = self.learner.clone_init()
            
        for train_index, test_index in rs:
            if (self.overlapping_features):
                l.initialize(self.X[train_index,:], self.y[train_index,],
                             sample_names=self.sample_names,
                             feature_names=self.feature_names)
            else:
                l.initialize(self.X[train_index,:], self.y[train_index,],
                             sample_names=self.sample_names,
                             feature_names=self.feature_names,
                             excluded_features=self.excluded_features)
            l.train()
            return(l)
        
    def train(self):
        if (self._trained):
            print("Warning: Rat is being retrained!")

        self.learners = []
        self.excluded_features = set()
        for i in range(self.learner_count):
            tmp = self.getSingleLearner()
            self.learners.append(tmp)
            print(tmp.getFeatures())
            self.excluded_features.update(list(tmp.getFeatures()))

        for l in self.learners:
            for feature in l.getFeatures():
                fc = self.feature_confidence_estimator.clone_init()
                fc.initialize(X, feature, self.excluded_features)
                fc.train()
                self.excluded_features.update(list(fc.getFeatures()))
                l.setFeatureConfidenceEstimator(feature, fc)

        self._trained = True

    def _predict(self, item, return_details = False):
        self._checkTrained()
        
        #if (not utilities.check_1d_array(item)):
        #    raise TypeError("item should be 1d ndarray")

        #item = item.reshape(1, -1)
        item = item.view(np.ndarray)
        if (item.ndim == 1):
            item = item.reshape(1, -1)

        predictions = []
        confidences = []
        for l in self.learners:
            predictions.append(l.predict(item))
            confidences.append(l.getConfidence(item))

        if (len(self.learners) > 1):
            result = np.average(predictions, weights=confidences)
        else:
            result = predictions[0]
        if (return_details):
            return((result, predictions, confidences))
        else:
            return(result)

    def predict(self, item, return_details = False):
        self._checkTrained()

        if (not isinstance(item, np.ndarray) or
            item.ndim > 2):
            raise TypeError("item should be 1d or 2d ndarray")

        item = item.view(np.ndarray)

        if (item.ndim == 1):
            return(self._predict(item, return_details))
        else:
            result = list()
            for i in range(item.shape[0]):
                result.append(self._predict(item[i,:], return_details))
            return(result)

