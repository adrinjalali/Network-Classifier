import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pickle
import sys
import os
import re

from misc import *

dataset_resolve = {'vantveer-prognosis': "Vant' Veer - Prognosis",
                   'TCGA-LAML-vital_status': 'TCGA-LAML - Vital Status',
                   'TCGA-LAML-risk_group': 'TCGA-LAML - Risk Group',
                   'TCGA-BRCA-ER': 'TCGA-BRCA - Estrogen Receptor',
                   'TCGA-BRCA-N': 'TCGA-BRCA - Lymph Node Status',
                   'TCGA-BRCA-T': 'TCGA-BRCA - Tumor Size',
                   'TCGA-BRCA-stage': 'TCGA-BRCA - Cancer Stage',
                   'TCGA-LAML-GeneExpression-risk_group':
                   'TCGA-LAML - Risk Group - Gene Expression',
                   'TCGA-LAML-GeneExpression-vital_status':
                   'TCGA-LAML - Vital Status - Gene Expression',
                   }
methods_order = ['SVM, linear kernel',
                 'SVM, RBF kernel',
                 'SVM, linear kernel, transformed',
                 'Adaboost',
                 'Gradient Boosting Classifier',
                 'RatBoost']

def append_score(scores, score):
    if (not isinstance(score, dict)):
        scores.append(score.flatten()[0])
    else:
        for key, value in score.items():
            if not key in scores:
                if isinstance(value, dict):
                    scores[key] = dict()
                else:
                    scores[key] = []
            append_score(scores[key], value)
        
    
def get_scores(root_dir):
    datas = os.listdir(root_dir)
    datas = [name for name in datas if os.path.isdir(
        root_dir + '/' + name)]

    all_scores = dict()

    for data in datas:
        print(data)
        targets = os.listdir(root_dir + '/' + data)

        for target in targets:
            print(target)
            files = os.listdir('%s/%s/%s/results/' % (root_dir,
                                                      data,
                                                      target))
            print(len(files))

            problem = '%s-%s' %(data, target)
            all_scores[problem] = dict()

            for f in files:
                #print('%s/%s/results/%s' % (
                #    data, target, f))
                scores = pickle.load(open('%s/%s/%s/results/%s' % (
                    root_dir, data, target, f), 'rb'))

                method, cv_index, major = re.split('[-\.]', f)[:3]
                cv_index = int(cv_index)

                append_score(all_scores[problem], scores)
    

    return(all_scores)

def add_text(prefix, parameter):
    ignore_parameters = ['learner_type', 'regularizer_index']
    if not isinstance(parameter, tuple):
        return prefix
    if (parameter[0] in ignore_parameters):
        return prefix
    if prefix.strip() == '':
        return prefix + "%s: %s" % (str(parameter[0]), str(parameter[1]))
    else:
        return prefix + ", %s: %s" % (str(parameter[0]), str(parameter[1]))
    return ('NA')
        

def flatten_scores_dict(scores):
    res = list()
    labels = list()
    for key in sorted(scores.keys()):
        value = scores[key]
        if isinstance(value, dict):
            _scores, _labels = flatten_scores_dict(value)
            for label in _labels:
                labels.append(add_text(label, key))
            for s in _scores:
                res.append(s)
        else:
            res.append(value)
            labels.append(add_text('', key))
    return(res, labels)

def draw_plot(all_scores, problem):
    colors = ['b', 'g', 'y', 'k', 'c', 'r', 'm', '0.5', '0.9']
    index = 0
    plot_colors = []
    plot_texts = []
    tmp = list()
    print_scores('', all_scores[problem])
    for method in methods_order:
        if not method in all_scores[problem]:
            continue
        _scores, _texts = flatten_scores_dict(all_scores[problem][method])
        for s in _scores:
            tmp.append(s)

        for t in _texts:
            plot_texts.append(t)

            plot_colors.append(colors[index])
        index += 1

    #plt.subplot(211)
    fig, ax = plt.subplots()
    pl = ax.boxplot(tmp, True)
    last_color = None
    idx = 0
    objs = []
    nms = []
    for i in range(len(plot_colors)):
        pl['boxes'][i].set_c(plot_colors[i])
        pl['boxes'][i].set_linewidth(2)
        if last_color != plot_colors[i]:
            objs.append(pl['boxes'][i])
            nms.append(methods_order[idx])
            idx += 1
        last_color = plot_colors[i]
    lgnd = plt.legend(objs, nms, fancybox=True)
    for l in lgnd.get_lines():
        l.set_linewidth(3)
    plt.suptitle(dataset_resolve[problem])
    
    texts = ax.set_xticklabels(plot_texts)
    for text in texts:
        text.set_rotation(270)
    plt.show()
    
def draw_plots(all_scores):
    for problem in sorted(all_scores.keys()):
        print(problem)
        draw_plot(all_scores, problem)
        
if __name__ == '__main__':
    root_dir = ''
    for i in range(len(sys.argv)):
        print(sys.argv[i])
        if (sys.argv[i] == '--root-dir'):
            root_dir = sys.argv[i + 1]

    if (root_dir == ''):
        root_dir = "/scratch/TL/pool0/ajalali/ratboost/data_6/"

    all_scores = get_scores(root_dir)

    print_log(all_scores)

    draw_plots(all_scores)

    draw_plot(all_scores, 'vantveer-prognosis')
    # stabilize results ( multiple runs )
    # make the plots
