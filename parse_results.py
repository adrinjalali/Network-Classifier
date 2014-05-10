import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pickle
import sys
import os
import re
import glob

from itertools import chain

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
                try:
                    scores = pickle.load(open('%s/%s/%s/results/%s' % (
                        root_dir, data, target, f), 'rb'))
                except EOFError as e:
                    print(e)
                    print('%s/%s/results/%s' % (
                        data, target, f))

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
    print_scores(all_scores[problem])
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


def load_models_info(root_dir, regularizer_index, learner_count):
    datas = os.listdir(root_dir)
    datas = [name for name in datas if os.path.isdir(
        root_dir + '/' + name)]

    tmp_g = gt.Graph(directed = False)
    vertex_map_name2index = dict()
    vertex_map_index2name = dict()
    

    for data in datas:
        print(data)
        targets = os.listdir(root_dir + '/' + data)

        for target in targets:
            print(target)
            files = glob.glob('%s/%s/%s/models/*-rat-%d-*' % (root_dir,
                                                      data,
                                                      target,
                                                      regularizer_index))
            structs = list()
            for f in files:
                struct = pickle.load(open(f, 'rb'))
                structs.append(struct)

            node_groups = [[list(m.keys()) for
                            m in s[:learner_count]]
                            for s in structs]
            nodes = sorted(set(chain.from_iterable(
                chain.from_iterable(node_groups))))

            adj = np.zeros(shape=(len(nodes), len(nodes)))
            for s in structs:
                for m in s[:learner_count]:
                    for i in m:
                        for j in m:
                            if i != j:
                                v1 = nodes.index(i)
                                v2 = nodes.index(j)
                                adj[v1,v2] = adj[v1,v2] + 1
                                #adj[v2,v1] = adj[v2,v1] + 1


        from scipy.stats import gaussian_kde
        import matplotlib.pyplot as plt
        density = gaussian_kde(adj[adj != 0])
        xs = np.linspace(0, max(adj[adj != 0]), 200)
        #density.covariance_factor = lambda : .25
        #density._compute_covariance()
        plt.plot(xs, density(xs))
        plt.show()


                                
        for b in range(10):
            tmp_g = gt.Graph(directed = False)
            tmp_g.add_vertex(len(nodes))
            edge_weights = tmp_g.new_edge_property('double')
            for i in range(len(nodes)):
                for j in range(len(nodes)):
                    if i > j and adj[i,j] > 30:
                        e = tmp_g.add_edge(i, j)
                        edge_weights[e] = 1 + 1/adj[i,j]

            gt.draw.graphviz_draw(tmp_g, layout='twopi',
                          size=(25,15),
                          #vcolor=vcolor,
                          #vcmap=plt.get_cmap('Blues'),
                          #vprops = {'label': vxlabel,
                          #          'shape': vshape,
                          #          'height': vheight,
                          #          'width': vwidth},
                          eprops = {'len': edge_weights})
            input()

if __name__ == '__main__':
    root_dir = ''
    for i in range(len(sys.argv)):
        print(sys.argv[i])
        if (sys.argv[i] == '--root-dir'):
            root_dir = sys.argv[i + 1]

    if (root_dir == ''):
        root_dir = "/scratch/TL/pool0/ajalali/ratboost/data_6/"

    all_scores = get_scores(root_dir)

    print_scores(all_scores)

    draw_plots(all_scores)
    
    draw_plot(all_scores, 'vantveer-prognosis')
    # stabilize results ( multiple runs )
    # make the plots
