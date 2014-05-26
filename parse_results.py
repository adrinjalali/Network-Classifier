import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pickle
import sys
import os
import re
import glob
from joblib import Parallel, delayed, logger

from itertools import chain
from scipy.stats import gaussian_kde

from plot_ratboost import generate_graph_plots
from plot_ratboost import get_feature_annotation
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
        


def ignore_key(key, filters):
    if filters == None:
        return False
    elif isinstance(filters, tuple):
        if filters[0] == key[0] and filters[1] != key[1]:
            return True
    else:
        for f in filters:
            if f[0] == key[0] and f[1] != key[1]:
                return True

    return False

def flatten_scores_dict(scores, filters = None):
    res = list()
    labels = list()
    for key in sorted(scores.keys()):
        #print(key)
        if ignore_key(key, filters):
            continue
        value = scores[key]
        if isinstance(value, dict):
            _scores, _labels = flatten_scores_dict(value, filters)
            #print(_labels)
            #print(labels)
            #print('$$$$$$$')
            for label in _labels:
                labels.append(add_text(label, key))
            for s in _scores:
                res.append(s)
        else:
            res.append(value)
            labels.append(add_text('', key))
    return(res, labels)

def draw_plot(all_scores, problem, filters = None, plt_ax = None, verbose=True):
    colors = ['b', 'g', 'y', 'k', 'c', 'r', 'm', '0.5', '0.9']
    index = 0
    plot_colors = []
    plot_texts = []
    tmp = list()
    if (verbose):
        print_scores(all_scores[problem])
    for method in methods_order:
        if not method in all_scores[problem]:
            continue
        _scores, _texts = flatten_scores_dict(all_scores[problem][method], filters)
        for s in _scores:
            tmp.append(s)

        for t in _texts:
            plot_texts.append(t)

            plot_colors.append(colors[index])
            #print(plot_colors)
        index += 1

    if (plt_ax is None):
        fig, ax = plt.subplots()
    else:
        ax = plt_ax
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
    #lgnd = plt.legend(objs, nms, fancybox=True)
    lgnd = plt.legend(objs, nms, bbox_to_anchor=[-1.6, -2.1, 2, 2],
                      loc='upper center',
              ncol=3,
              mode="expand",
              borderaxespad=0.,
              fancybox=True
              )
    #lgnd.draggable(True)
    for l in lgnd.get_lines():
        l.set_linewidth(3)
    ax.set_title(dataset_resolve[problem])
    
    texts = ax.set_xticklabels(plot_texts)
    for text in texts:
        text.set_rotation(270)
    if (plt_ax is None):
        plt.show()

    return (objs, nms)
    
def draw_plots(all_scores):
    for problem in sorted(all_scores.keys()):
        print(problem)
        draw_plot(all_scores, problem)


def load_models_info(root_dir, regularizer_index, learner_count,
                     data = None,
                     target = None):

    if (data is None):
        datas = os.listdir(root_dir)
        datas = [name for name in datas if os.path.isdir(
            root_dir + '/' + name)]
    elif isinstance(data, list):
        datas = data
    else:
        datas = list([data])

    for data in datas:
        print(data)
        if (target is None):
            targets = os.listdir(root_dir + '/' + data)
        elif isinstance(target, list):
            targets = target
        else:
            targets = list([target])

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

            vertex_map_name2index = dict()
            vertex_map_index2name = dict()

            for i in range(len(nodes)):
                vertex_map_name2index[nodes[i]] = i
                vertex_map_index2name[i] = nodes[i]

            node_confidence = dict()
            adj = np.zeros(shape=(len(nodes), len(nodes)))
            for s in structs:
                for m in s[:learner_count]:
                    for i in m:
                        if (i in node_confidence):
                            node_confidence[i] += 1
                        else:
                            node_confidence[i] = 1
                        for j in m:
                            if i != j:
                                v1 = nodes.index(i)
                                v2 = nodes.index(j)
                                adj[v1,v2] = adj[v1,v2] + 1
                                #adj[v2,v1] = adj[v2,v1] + 1

            density = gaussian_kde(adj[adj != 0])
            xs = np.linspace(0, max(adj[adj != 0]), 200)
            #density.covariance_factor = lambda : .25
            #density._compute_covariance()


            #find threshold for top X% of area under the density curve
            low = -1e3
            high = 1e3
            target_top = 0.1
            while(True):
                mid = (low + high) / 2
                mid_v = density.integrate_box_1d(mid, 1e4)
                print(mid, mid_v)
                if (abs(mid_v - target_top) < 1e-2):
                    break
                if (mid_v > target_top):
                    low = mid
                elif (mid_v < target_top):
                    high = mid

            plt.figure()
            plt.plot(xs, density(xs))
            plt.axvline(mid, c='g')
            #plt.show()
            plt.savefig('tmp/density-%s-%s-%d.eps' %
                        (data, target, regularizer_index // 2), dpi=100)
            plt.close()

            threshold = mid
            
            tmp_g = gt.Graph(directed = False)

            has_edge = dict()
            tmp_g.add_vertex(len(nodes))
            
            node_confidence_vprop = tmp_g.new_vertex_property('double')
            for key, value in node_confidence.items():
                node_confidence_vprop[tmp_g.vertex(nodes.index(key))] = value
                
            edge_weights = tmp_g.new_edge_property('double')
            for i in range(len(nodes)):
                for j in range(len(nodes)):
                    if i > j and adj[i,j] > threshold:
                        has_edge[i] = True
                        has_edge[j] = True
                        e = tmp_g.add_edge(i, j)
                        edge_weights[e] = 1 + 1/adj[i,j]

            feature_annotation = get_feature_annotation(root_dir, data, target)
            vlabel = tmp_g.new_vertex_property('string')
            vfontcolor = tmp_g.new_vertex_property('string')
            feature_list = list()
            for v in tmp_g.vertices():
                if (v.in_degree() > 0 or v.out_degree() > 0 or
                    node_confidence[vertex_map_index2name[int(v)]] > threshold):
                    vlabel[v] = feature_annotation[vertex_map_index2name[int(v)]]
                    feature_list.append(vertex_map_index2name[int(v)])
                    if (node_confidence[vertex_map_index2name[int(v)]] >
                        np.mean((max(node_confidence.values()),
                                 min(node_confidence.values())))):
                        vfontcolor[v] = 'white'
                    else:
                        vfontcolor[v] = 'black'

            gt.draw.graphviz_draw(tmp_g, layout='neato',
                          size=(18.5,10.5),
                          vcolor=node_confidence_vprop,
                          vcmap=plt.get_cmap('Blues'),
                          vprops = {'label': vlabel,
                                    'shape': 'ellipse',
                                    'vpenwidth': 1,
                                    'fontcolor': vfontcolor},
                          eprops = {'len': edge_weights},
                          #gprops = {'labelloc':'t',
                          #          'label':'High Confidence Features: %s %s (%g)' % \
                          #          (data, target, threshold)},
                          output = 'tmp/summary-%s-%s-%02d.eps' %
                          (data, target, regularizer_index // 2))

            #if (data == 'TCGA-BRCA'):
            #    continue


            generate_graph_plots(root_dir, data, target, learner_count,
                         regularizer_index,
                         feature_list,
                         node_confidence,
                         cv_index = 35,
                         threshold = threshold,
                         plot_titles = False)


if __name__ == '__main__':
    root_dir = ''
    for i in range(len(sys.argv)):
        print(sys.argv[i])
        if (sys.argv[i] == '--root-dir'):
            root_dir = sys.argv[i + 1]

    if (root_dir == ''):
        root_dir = "/scratch/TL/pool0/ajalali/ratboost/data_7/"

    all_scores = get_scores(root_dir)

    #print_scores(all_scores)

    #draw_plots(all_scores)
    
    #draw_plot(all_scores, 'vantveer-prognosis', ('regularizer_index', 4))

    '''
    at the moment there are 9 dataset/problems, plot them in
    3x3 subplots
    '''
    regularizer_indices = [2, 4, 6, 8, 10, 12, 14, 16, 18]
    for ri in regularizer_indices:
        print(ri)
        f, axarr = plt.subplots(3, 3, sharey = False)
        draw_plot(all_scores, 'TCGA-BRCA-T',
            ('regularizer_index', ri), axarr[0, 0], False)
        draw_plot(all_scores, 'TCGA-BRCA-N',
            ('regularizer_index', ri), axarr[0, 1], False)
        draw_plot(all_scores, 'TCGA-BRCA-ER',
            ('regularizer_index', ri), axarr[0, 2], False)
        draw_plot(all_scores, 'TCGA-BRCA-stage',
            ('regularizer_index', ri), axarr[1, 0], False)
        draw_plot(all_scores, 'TCGA-LAML-GeneExpression-risk_group',
            ('regularizer_index', ri), axarr[1, 1], False)
        draw_plot(all_scores, 'TCGA-LAML-GeneExpression-vital_status',
            ('regularizer_index', ri), axarr[1, 2], False)
        draw_plot(all_scores, 'TCGA-LAML-risk_group',
            ('regularizer_index', ri), axarr[2, 1], False)
        draw_plot(all_scores, 'TCGA-LAML-vital_status',
            ('regularizer_index', ri), axarr[2, 2], False)
        draw_plot(all_scores, 'vantveer-prognosis',
            ('regularizer_index', ri), axarr[2, 0], False)

        fig = plt.gcf()
        fig.set_size_inches(18.5,10.5)
        plt.tight_layout(pad=2)
        plt.savefig('tmp/performance-notaligned-%02d.eps' % (ri // 2), dpi=100)
        plt.close()
        #plt.show()

        f, axarr = plt.subplots(3, 3, sharey = True)
        draw_plot(all_scores, 'TCGA-BRCA-T',
            ('regularizer_index', ri), axarr[0, 0], False)
        draw_plot(all_scores, 'TCGA-BRCA-N',
            ('regularizer_index', ri), axarr[0, 1], False)
        draw_plot(all_scores, 'TCGA-BRCA-ER',
            ('regularizer_index', ri), axarr[0, 2], False)
        draw_plot(all_scores, 'TCGA-BRCA-stage',
            ('regularizer_index', ri), axarr[1, 0], False)
        draw_plot(all_scores, 'TCGA-LAML-GeneExpression-risk_group',
            ('regularizer_index', ri), axarr[1, 1], False)
        draw_plot(all_scores, 'TCGA-LAML-GeneExpression-vital_status',
            ('regularizer_index', ri), axarr[1, 2], False)
        draw_plot(all_scores, 'TCGA-LAML-risk_group',
            ('regularizer_index', ri), axarr[2, 1], False)
        draw_plot(all_scores, 'TCGA-LAML-vital_status',
            ('regularizer_index', ri), axarr[2, 2], False)
        draw_plot(all_scores, 'vantveer-prognosis',
            ('regularizer_index', ri), axarr[2, 0], False)

        fig = plt.gcf()
        fig.set_size_inches(18.5,10.5)
        plt.tight_layout(pad=2)
        plt.savefig('tmp/performance-aligned-%02d.eps' % (ri // 2), dpi=100)
        plt.close()

        learner_count = 4
        
        load_models_info(root_dir, ri, learner_count)
