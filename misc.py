"""
    License: GPLv3.
    Adrin Jalali.
    Jan 2014, Saarbruecken, Germany.

miscellaneous functions used mostly in the data loading/preprocessing
phase.
"""
import csv
import sklearn.ensemble
import sklearn.tree
import time
from joblib import logger
import re
import scipy.stats

from rat import *

def read_csv(file_name, skip_header, delimiter = '\t'):
    data = csv.reader(open(file_name, 'r'), delimiter=delimiter);
    if (skip_header): next(data);
    table = [row for row in data];
    return table;

def get_column(table, col):
    res = list();
    for i in range(len(table)):
        res.append(table[i][col]);
    return res;

def dump_list(data, file_name):
    file = open(file_name, 'w');
    for item in data:
        print>>file, item;
    file.close();

def extract(d, keys):
    return dict((k, d[k]) for k in keys if k in d);

def print_stats(mine):
    print("MIC", mine.mic())
    print("MAS", mine.mas())
    print("MEV", mine.mev())
    print("MCN (eps=0)", mine.mcn(0))
    print("MCN (eps=1-MIC)", mine.mcn_general())
                        
def reload_rat():
    with open("./rat.py") as f:
        code = compile(f.read(), "rat.py", 'exec')
        exec(code)

def get_max_score(scores, prefix = ''):
    if isinstance(scores, dict):
        items = [get_max_score(v, prefix + ' %s' % (str(k))) for
                k, v in scores.items()]
        return(max(items, key=lambda x:x[1]))
    else:
        return(prefix + ' (count: %d)' % (len(list(scores))), np.median(scores),
               np.std(scores))
        
def print_scores(scores, prefix = '', output_stream=sys.stderr):
    if (len(scores) == 0):
        return
    #report_max = ['N', 'learner_type']
    report_max = ['N']
    ignore_params = ['learner_type']
    if isinstance(scores, dict):
        key0 = next(iter(scores.keys()))
        if isinstance(key0, tuple) and key0[0] in report_max:
            txt, m, v = get_max_score(scores, prefix)
            message = "%s: %.3lg +/- %.3lg" % (txt, m, v)
        else:
            for key in sorted(scores.keys()):
                value = scores[key]
                if (prefix != ''):
                    print_scores(value, "%s %s" %(prefix, str(key)))
                else:
                    print('\n', key, file=output_stream)
                    print_scores(value, "\t")
            return
    else:
        message = "%s (count: %d): %.3lg +/- %.3lg" % (prefix, len(scores),
                                           np.mean(scores),
                                           2 * np.std(scores))
    for s in ignore_params:
        message = re.sub(" *\('%s.*'\) *" % (s), ' ', message)
    print(message, file=output_stream)


def print_summary(scores, methods):
    method_ranks = {m: [] for m in methods}
    for problem in scores.keys():
        p_scores = scores[problem]
        has_methods = np.all([m in p_scores.keys() for m in methods])
        if not has_methods:
            continue

        result_counts = [len(p_scores[m]) for m in methods]
        if len(np.unique(result_counts)) > 1:
            continue

        means = {x: np.mean(y) for x, y in p_scores.items() if x in methods}
        order = sorted(means, key=means.get, reverse=True)

        print(scipy.stats.ttest_rel(p_scores[order[0]], p_scores[order[len(order) - 1]]))

        for i in range(len(order)):
            method_ranks[order[i]].append(i)

    print("%d problems have complete results" % len(method_ranks[methods[0]]))
    print("Method rank average (0 is best):")
    for m, r in method_ranks.items():
        print(m, np.mean(r))
                    
        
def print_log(all_scores, rat_scores = dict()):
    print('=========')
    print_scores(all_scores)
    print_scores(rat_scores)

def dump_scores(file_name, scores):
    import pickle
    pickle.dump(scores, open(file_name, "wb"))

def _fit_and_score(estimator, X, y, scorer, train, test, verbose, parameters,
                   fit_params, max_learner_count, return_train_score=False,
                   return_parameters=False):
    if verbose > 1:
        if parameters is None:
            msg = "no parameters to be set"
        else:
            msg = '%s' % (', '.join('%s=%s' % (k, v)
                                    for k, v in parameters.items()))
        print("[CV] %s %s" % (msg, (64 - len(msg)) * '.'))

    # Adjust lenght of sample weights
    n_samples = len(X)
    fit_params = fit_params if fit_params is not None else {}
    fit_params = dict([(k, np.asarray(v)[train]
                        if hasattr(v, '__len__') and len(v) == n_samples else v)
                       for k, v in fit_params.items()])

    if parameters is not None:
        estimator.set_params(**parameters)

    X_train, y_train = sklearn.cross_validation._safe_split(
        estimator, X, y, train)
    X_test, y_test = sklearn.cross_validation._safe_split(
        estimator, X, y, test, train)
    result = list()
    from_scratch = True
    for i in range(max_learner_count):
        start_time = time.time()

        estimator.fit(X_train, y_train, from_scratch = from_scratch)
        test_score = sklearn.cross_validation._score(
            estimator, X_test, y_test, scorer)
        if return_train_score:
            train_score = _score(estimator, X_train, y_train, scorer)
        ret = [train_score] if return_train_score else []

        scoring_time = time.time() - start_time

        ret.extend([test_score, len(X_test), scoring_time])
        if return_parameters:
            ret.append(parameters)
        result.append(ret)
        from_scratch = False


        if verbose > 2:
            msg += ", score=%f" % test_score
        if verbose > 1:
            end_msg = "%s -%s" % (msg, logger.short_format_time(scoring_time))
            print("[CV] %s %s" % ((64 - len(end_msg)) * '.', end_msg))

    return (result, estimator)
        
def rat_cross_val_score(estimator, X, y=None, scoring=None, cv=None, n_jobs=1,
                    verbose=0, fit_params=None, score_func=None,
                    pre_dispatch='2*n_jobs', max_learner_count = 2):
    X, y = sklearn.utils.check_arrays(X, y, sparse_format='csr', allow_lists=True)
    cv = sklearn.cross_validation._check_cv(cv,
                                            X, y,
                                            classifier=sklearn.base.is_classifier(estimator))
    scorer = sklearn.cross_validation.check_scoring(
        estimator, score_func=score_func, scoring=scoring)
    # We clone the estimator to make sure that all the folds are
    # independent, and that it is pickle-able.

    jobs = list(dict())

    fit_params = fit_params if fit_params is not None else {}

    fit_params['from_scratch'] = True
    collected_scores = dict()
    scorer = sklearn.metrics.scorer.get_scorer(scoring)
    if (n_jobs > 1):
        parallel = Parallel(n_jobs=n_jobs, verbose=verbose,
                        pre_dispatch=pre_dispatch)
        result = parallel(
            delayed(_fit_and_score)(
                estimator,
                X, y, scorer,
                train, test,
                verbose, None, fit_params,
                max_learner_count = max_learner_count)
            for train, test in cv)
    else:
        print("going for single job system")
        result = [_fit_and_score(
            estimator,
            X, y, scorer,
            train, test,
            verbose, None, fit_params,
            max_learner_count = max_learner_count)
            for train, test in cv]
    
    return (result)

