import scipy.io
import sklearn.ensemble
import sklearn.tree

from misc import *
from rat import *


def add_to_scores(params):
    current = all_scores
    for item in params:
        if not (item in current):
            current[item] = dict()
        current = current[item]


if __name__ == '__main__':
    bnet_count = -1
    feature_noise = -1
    result_dump_dir = '/TL/stat_learn/work/ajalali/Network-Classifier/synthesized_results'
    #data_dir = '/TL/stat_learn/work/ajalali/Network-Classifier/synthesized_results-1'
    data_dir = 'synthesized_results-1'
    # data_dir = '/TL/stat_learn/work/ajalali/bayesnet'
    for i in range(len(sys.argv)):
        print(sys.argv[i], file=sys.stderr)
        if sys.argv[i] == '--bnet_count':
            bnet_count = int(sys.argv[i + 1])
        if (sys.argv[i] == '--feature_noise'):
            feature_noise = float(sys.argv[i + 1])

    if feature_noise == -1:
        feature_noise = .3

    if bnet_count == -1:
        bnet_count = 10

    tmp = scipy.io.loadmat('%s/data-bnet_count-%d-feature_noise-%g.mat' % (data_dir, bnet_count, feature_noise))
    Xtrain = tmp['Xtrain']
    ytrain = tmp['ytrain'].reshape(-1)
    Xtest = tmp['Xtest']
    ytest = tmp['ytest'].reshape(-1)
    train_feature_noise = tmp['train_feature_noise']
    test_feature_noise = tmp['test_feature_noise']
    train_feature_bnet = tmp['train_feature_bnet']
    test_feature_bnet = tmp['test_feature_bnet']

    cpu_count = 50

    # cvs = list(cv.StratifiedShuffleSplit(y, n_iter = 100, test_size = 0.2))

    all_scores = defaultdict(list)

    print('svms')
    for nu in np.arange(7) * 0.1 + 0.05:
        try:
            machine = svm.NuSVC(nu=nu,
                                kernel='linear',
                                verbose=False,
                                probability=False)
            machine.fit(Xtrain, ytrain)
            scores = roc_auc_score(ytest, machine.decision_function(Xtest))
            print(scores)
            this_method = 'SVM, linear kernel'
            add_to_scores([this_method, ('nu', nu)])
            all_scores[this_method][('nu', nu)] = [scores]
        except ValueError as e:
            print(nu, e)

        try:
            machine = svm.NuSVC(nu=nu,
                                kernel='rbf',
                                verbose=False,
                                probability=False)
            machine.fit(Xtrain, ytrain)
            scores = roc_auc_score(ytest, machine.decision_function(Xtest))
            print(scores)
            this_method = 'SVM, RBF kernel'
            add_to_scores([this_method, ('nu', nu)])
            all_scores[this_method][('nu', nu)] = [scores]
        except ValueError as e:
            print(nu, e)

    print('gbc')
    for mf in np.arange(3) * 5 + 5:
        for md in np.arange(3) + 1:
            for ne in [5, 20, 50, 100, 200]:
                machine = sklearn.ensemble.GradientBoostingClassifier(
                    max_features=mf,
                    max_depth=md,
                    n_estimators=ne)
                machine.fit(Xtrain, ytrain)
                scores = roc_auc_score(ytest, machine.decision_function(Xtest))
                print(scores)
                this_method = 'Gradient Boosting Classifier'
                add_to_scores([this_method, ('max_features', mf),
                               ('max_depth', md), ('N', ne)])
                all_scores[this_method][('max_features', mf)] \
                    [('max_depth', md)][('N', ne)] = [scores]

    # print('adaboost')
    # for md in np.arange(3) + 1:
    #     for ne in [5, 20, 50, 100, 200]:
    #         machine = sklearn.ensemble.AdaBoostClassifier(
    #             sklearn.tree.DecisionTreeClassifier(max_depth=md),
    #             algorithm="SAMME.R",
    #             n_estimators=ne)
    #         machine.fit(Xtrain, ytrain)
    #         scores = roc_auc_score(ytest, machine.decision_function(Xtest))
    #         print(scores)
    #         this_method = 'Adaboost'
    #         add_to_scores([this_method, ('max_depth', md),
    #                        ('N', ne)])
    #         all_scores[this_method][('max_depth', md)][('N', ne)] = [scores]

    print('ratboost')
    max_learner_count = 25
    this_method = 'RatBoost'
    all_scores[this_method] = dict()
    rat_models = list()
    #for ri in np.hstack((1, np.array(list(range(10))) * 2)):
    for ri in [1]:
        print('------------ ri', ri)
        rat = Rat(learner_count=max_learner_count,
                  learner_type='gdb',
                  regularizer_index=ri,
                  n_jobs=cpu_count)
        all_scores[this_method][('regularizer_index', ri)] = dict()
        rat.fit(Xtrain, ytrain)
        rat_models.append(rat)
        print('scores')
        test_decision_values = rat.decision_function(Xtest, return_iterative=True)
        train_decision_values = rat.decision_function(Xtrain, return_iterative=True)
        for i in range(len(test_decision_values)):
            scores = roc_auc_score(ytest, test_decision_values[i])
            print(roc_auc_score(ytrain, train_decision_values[i]))
            print('test', scores)

            all_scores[this_method][('regularizer_index', ri)] \
                [('N', i)] = [scores]

    # pickle.dump(all_scores, open('%s/scores-bnet_count-%d-feature_noise-%g.pickled' %
    #                              (result_dump_dir, bnet_count, feature_noise), 'wb'))


def check_results(result_dump_dir):
    files = os.listdir(result_dump_dir)
    for f in files:
        print('\n^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^')
        print(f)
        data = pickle.load(open('%s/%s' % (result_dump_dir, f), 'rb'))
        print_scores(data)


def gradientboostingclassifier_get_feature_importances(model):
    tmp = np.vstack((np.arange(len(machine.feature_importances_))[machine.feature_importances_ != 0],
                     machine.feature_importances_[machine.feature_importances_ != 0]));
    tmp = tmp[:, np.argsort(tmp)[1,]];
    [print('%d, %g' % (tmp[0, i], tmp[1, i])) for i in range(tmp.shape[1])];


def ratboost_get_feature_importances(model, sample, feature_noise):
    tmp = model.feature_importances(sample);
    tmp = np.array([[key, value[0]] for key, value in tmp.items()])
    tmp = tmp[np.argsort(tmp, axis=0)[:, 1],];
    [print('%d, %d, %g' % (tmp[i, 0], feature_noise[tmp[i, 0]], tmp[i, 1])) for i in range(tmp.shape[0])];
