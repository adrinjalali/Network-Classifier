import sys;
import os;
import numpy as np;
import graph_tool as gt;
from graph_tool import draw;
from graph_tool import spectral;
from graph_tool import stats;
from sklearn import svm;
from sklearn import cross_validation as cv;
from sklearn.metrics import roc_auc_score;
from sklearn.grid_search import GridSearchCV
import sklearn.ensemble
import sklearn.tree
from collections import defaultdict
import time
from joblib import Parallel, delayed, logger
import pickle
from graph_tool import topology as tpl
import graph_tool.draw
import matplotlib as mpl
import matplotlib.pyplot as plt


from constants import *;
from misc import *
import read_nordlund1 
import read_nordlund2
import read_vantveer
import read_tcga_brca
import read_tcga_laml
import read_tcga_laml_geneexpression
from rat import *

def add_vertex(g, vertex_map_name2index, vertex_map_index2name, name):
    if (not name in vertex_map_name2index):
        index = g.num_vertices()
        vertex_map_name2index[name] = index
        vertex_map_index2name[index] = name
        return((g.add_vertex(1), True))
    return((g.vertex(vertex_map_name2index[name]), False))
    
def plot_graph(hilight_learner, test_sample, color_scheme='gene'):
    node_shapes = ['box', 'ellipse',
                   'circle', 'triangle', 'diamond',
                   'trapezium', 'parallelogram',
                   'house', 'pentagon', 'hexagon',
                   'septagon', 'octagon', 'invtriangle']
    
    tmp_g = gt.Graph(directed = True)
    vertex_map_name2index = dict()
    vertex_map_index2name = dict()
    cur_learner = hilight_learner
    vcolor = tmp_g.new_vertex_property('double')
    vwidth = tmp_g.new_vertex_property('string')
    vheight = tmp_g.new_vertex_property('string')
    vxlabel = tmp_g.new_vertex_property('string')
    vshape = tmp_g.new_vertex_property('string')

    confidences = dict()
    
    for i in range(len(a.learners)):
        print('========================')
        print(a.learners[i].getClassifierFeatures())

        for gene in a.learners[i].getClassifierFeatures():
            cur_vertex, is_new = add_vertex(g = tmp_g,
                            vertex_map_name2index = vertex_map_name2index,
                            vertex_map_index2name = vertex_map_index2name,
                            name = gene)

            if (i == cur_learner):
                #vcolor[cur_vertex] = 'yellow'
                vwidth[cur_vertex] = '1.5'
            #elif (is_new == True):
            else:
                #vcolor[cur_vertex] = 'blue'
                vwidth[cur_vertex] = '1.5'

            if (color_scheme == 'gene'):
                vcolor[cur_vertex] = a.learners[i]._FCEs[gene].getConfidence(test_sample)[0] * a.learners[i].getClassifierFeatureWeights()[gene]
            elif (color_scheme == 'learner'):
                vcolor[cur_vertex] = a.learners[i].getConfidence(test_sample)
            print(vcolor[cur_vertex])
            vxlabel[cur_vertex] = feature_annotation[gene]
            vshape[cur_vertex] = node_shapes[i]
            vheight[cur_vertex] = '2'

            parents = a.learners[i]._FCEs[gene].getFeatures()

            for parent in parents:
                cur_vertex, is_new = add_vertex(g = tmp_g,
                           vertex_map_name2index = vertex_map_name2index,
                           vertex_map_index2name = vertex_map_index2name,
                           name = parent)
                tmp_g.add_edge(
                    vertex_map_name2index[parent],
                    vertex_map_name2index[gene])

            #print(gene, parents)


            #shortest_dists = [tpl.shortest_distance(g, g.vertex(gene),
            #                                    g.vertex(idx))
            #                  for idx in parents]
            #print(shortest_dists)
    gt.draw.graphviz_draw(tmp_g, layout='dot',
                          size=(25,25),
                          vcolor=vcolor,
                          vcmap=plt.get_cmap('Blues'),
                          vprops = {'label': vxlabel,
                                    'shape': vshape,
                                    'height': vheight,
                                    'width': vwidth})

# this plots a model with L_%d label for learners, and others as gene names.
def plot_learners(model, test_sample, feature_annotation,
                  plot_title = '', output = ''):
    node_shapes = ['box', 'ellipse',
                   'triangle', 'diamond',
                   'trapezium', 'parallelogram',
                   'house', 'pentagon', 'hexagon',
                   'septagon', 'octagon', 'invtriangle']
    
    tmp_g = gt.Graph(directed = False)
    vertex_map_name2index = dict()
    vertex_map_index2name = dict()
    vcolor = tmp_g.new_vertex_property('double')
    vwidth = tmp_g.new_vertex_property('string')
    vheight = tmp_g.new_vertex_property('string')
    vxlabel = tmp_g.new_vertex_property('string')
    vshape = tmp_g.new_vertex_property('string')
    vfontcolor = tmp_g.new_vertex_property('string')

    l_cdfs = dict()
    v_cdfs = dict()

    for i in range(len(model.learners)):
        print('========================')
        print(model.learners[i].getClassifierFeatures())

        cur_vertex, is_new = add_vertex(g = tmp_g,
            vertex_map_name2index = vertex_map_name2index,
            vertex_map_index2name = vertex_map_index2name,
            name = 'L_%d' % i)
        #vwidth[cur_vertex] = '1.5'
        #vheight[cur_vertex] = '2'
        l_cdfs[cur_vertex] = model.learners[i].getConfidence(test_sample)
        print(i, model.learners[i].getConfidence(test_sample))
        vxlabel[cur_vertex] = 'L_%d' % i
        vshape[cur_vertex] = node_shapes[i]
        target_vertex = cur_vertex
        
        for gene in model.learners[i].getClassifierFeatures():
            cur_vertex, is_new = add_vertex(g = tmp_g,
                            vertex_map_name2index = vertex_map_name2index,
                            vertex_map_index2name = vertex_map_index2name,
                            name = gene)

            #vwidth[cur_vertex] = '1.5'
            #vheight[cur_vertex] = '2'
            v_cdfs[cur_vertex] = model.learners[i]._FCEs[gene].\
              getConfidence(test_sample)[0] * \
              model.learners[i].getClassifierFeatureWeights()[gene]
            print(gene, model.learners[i]._FCEs[gene].\
              getConfidence(test_sample)[0])
            vxlabel[cur_vertex] = feature_annotation[gene]
            vshape[cur_vertex] = node_shapes[i]

            tmp_g.add_edge(cur_vertex, target_vertex)

    l_minmax = (min(list(l_cdfs.values())),
                max(list(l_cdfs.values())))
    v_minmax = (min(list(v_cdfs.values())),
                max(list(v_cdfs.values())))
    for v, c in l_cdfs.items():
        vcolor[v] = (c - l_minmax[0]) / (l_minmax[1] - l_minmax[0])
        if (c > np.mean(l_minmax)):
            vfontcolor[v] = 'white'
        else:
            vfontcolor[v] = 'black'
    for v, c in v_cdfs.items():
        vcolor[v] = (c - v_minmax[0]) / (v_minmax[1] - v_minmax[0])
        if (c > np.mean(v_minmax)):
            vfontcolor[v] = 'white'
        else:
            vfontcolor[v] = 'black'
    '''circo dot fdp neato nop nop1 nop2 osage patchwork sfdp twopi'''
    gt.draw.graphviz_draw(tmp_g, layout='neato',
                          size=(25,25),
                          vcolor=vcolor,
                          vcmap=plt.get_cmap('Blues'),
                          vprops = {'label': vxlabel,
                                    'shape': vshape,
                                    'height': vheight,
                                    'width': vwidth,
                                    'fontcolor': vfontcolor},
                          gprops = {'labelloc':'t',
                                    'fontsize':36,
                                    'label':plot_title},
                          output = output)

def plot_test_features_on_graph(rat, g, test_sample,
                                feature_annotation, plot_title = '',
                                output = ''):
    node_shapes = ['box', 'ellipse',
                   'triangle', 'diamond',
                   'trapezium', 'parallelogram',
                   'house', 'pentagon', 'hexagon',
                   'septagon', 'octagon', 'invtriangle']

    tmp_g = g.copy()
    vxlabel = tmp_g.new_vertex_property('string')
    vcolor = tmp_g.new_vertex_property('string')
    vshape = tmp_g.new_vertex_property('string')
    vpenwidth = tmp_g.new_vertex_property('double')
    included = tmp_g.new_vertex_property('bool')
    edge_included = tmp_g.new_edge_property('bool')
    is_feature = dict()
    vscore = dict()
    lvscore = dict()

    vertex_list = list()

    for l_idx, l in zip(range(len(rat.learners)), rat.learners):
        for i in l.getClassifierFeatures():
            vscore[tmp_g.vertex(i)] = \
              l._FCEs[i].getConfidence(test_sample).reshape(-1)[0]
            vshape[tmp_g.vertex(i)] = node_shapes[l_idx]
            is_feature[tmp_g.vertex(i)] = True
            lvscore[tmp_g.vertex(i)] = \
              l.getConfidence(test_sample).reshape(-1)[0]
            vertex_list.append(i)

    for vi in vertex_list:
        for vj in vertex_list:
            if (vi < vj):
                path_v, path_e = gt.topology.shortest_path(g,
                                                 g.vertex(vi),
                                                 g.vertex(vj))
                #print('------')
                for v in path_v:
                    included[v] = True
                    vxlabel[v] = feature_annotation[int(v)]
                    #print(feature_annotation[int(v)])
                    if (not v in is_feature):
                        vcolor[v] = 'white'
                        vshape[v] = 'circle'
                        vpenwidth[v] = 1
                for e in path_e:
                    edge_included[e] = True

    vcmap = mpl.cm.summer
    v_minmax = (min(list(vscore.values())),
                max(list(vscore.values())))
    vnorm = mpl.colors.Normalize(vmin=v_minmax[0], vmax=v_minmax[1])
    for v, s in vscore.items():
        color = tuple([int(c * 255.0) for c in vcmap(vnorm(s))])
        vcolor[v] = "#%.2x%.2x%.2x%.2x" % color

    lv_minmax = (min(list(lvscore.values())),
                 max(list(lvscore.values())))
    lvnorm = mpl.colors.Normalize(vmin=lv_minmax[0], vmax=lv_minmax[1])
    for v, s in lvscore.items():
        vpenwidth[v] = 1 + lvnorm(s) * 5

    tmp_g.set_vertex_filter(included)
    #tmp_g.set_edge_filter(edge_included)
    tmp_g.set_edge_filter(gt.topology.min_spanning_tree(tmp_g))

    while (True):
        flag = False
        tmp_g2 = gt.GraphView(tmp_g, vfilt = included)
        for v in tmp_g2.vertices():
            if (v.in_degree() == 1 or v.out_degree() == 1):
                if (not tmp_g.vertex(int(v)) in is_feature):
                    included[tmp_g.vertex(int(v))] = False
                    flag = True

        tmp_g.set_vertex_filter(included)
        if (not flag):
            break

    gt.draw.graphviz_draw(tmp_g, layout = 'neato',
                          vcolor=vcolor,
                          size=(25,25),
                          vprops = {'label': vxlabel,
                                    'width': 1,
                                    'shape': vshape,
                                    'penwidth': vpenwidth,
                                    'fontsize': 20},
                          gprops = {'labelloc':'t',
                                    'fontsize': 36,
                                    'label':plot_title},
                          output = output)

def plot_features_on_graph(g, feature_list,
                           feature_confidence,
                           feature_annotation, plot_title = '',
                           output = ''):
    node_shapes = ['box', 'ellipse',
                   'triangle', 'diamond',
                   'trapezium', 'parallelogram',
                   'house', 'pentagon', 'hexagon',
                   'septagon', 'octagon', 'invtriangle']

    tmp_g = g.copy()
    vxlabel = tmp_g.new_vertex_property('string')
    vcolor = tmp_g.new_vertex_property('string')
    vshape = tmp_g.new_vertex_property('string')
    vpenwidth = tmp_g.new_vertex_property('double')
    included = tmp_g.new_vertex_property('bool')
    edge_included = tmp_g.new_edge_property('bool')
    is_feature = dict()
    vscore = dict()
    lvscore = dict()

    for vi in feature_list:
        #included[g.vertex(vi)] = True
        #vxlabel[g.vertex(vi)] = feature_annotation[int(vi)]
        #vcolor[g.vertex(vi)] = 'blue'
        for vj in feature_list:
            if (vi < vj):
                path_v, path_e = gt.topology.shortest_path(tmp_g,
                                                 tmp_g.vertex(vi),
                                                 tmp_g.vertex(vj))
                #print('------', feature_annotation[vi], feature_annotation[vj])
                for v in path_v:
                    included[v] = True
                    vxlabel[v] = feature_annotation[int(v)]
                    #print(feature_annotation[int(v)])
                    if (not v in is_feature):
                        vcolor[v] = 'white'
                        vshape[v] = 'circle'
                        vpenwidth[v] = 1
                for e in path_e:
                    edge_included[e] = True

    for vi in feature_list:
        vcolor[tmp_g.vertex(vi)] = 'blue'
        is_feature[tmp_g.vertex(vi)] = True


    vcmap = mpl.cm.summer
    v_minmax = (min(list(feature_confidence.values())),
                max(list(feature_confidence.values())))
    vnorm = mpl.colors.Normalize(vmin=v_minmax[0], vmax=v_minmax[1])
    for v, s in feature_confidence.items():
        color = tuple([int(c * 255.0) for c in vcmap(vnorm(s))])
        vcolor[tmp_g.vertex(v)] = "#%.2x%.2x%.2x%.2x" % color

    '''
    lv_minmax = (min(list(lvscore.values())),
                 max(list(lvscore.values())))
    lvnorm = mpl.colors.Normalize(vmin=lv_minmax[0], vmax=lv_minmax[1])
    for v, s in lvscore.items():
        vpenwidth[v] = 1 + lvnorm(s) * 5
    '''
    tmp_g.set_vertex_filter(included)
    #tmp_g.set_edge_filter(edge_included)
    tmp_g.set_edge_filter(gt.topology.min_spanning_tree(tmp_g))

    while (True):
        flag = False
        tmp_g2 = gt.GraphView(tmp_g, vfilt = included)
        for v in tmp_g2.vertices():
            if (v.in_degree() == 1 or v.out_degree() == 1):
                if (not tmp_g.vertex(int(v)) in is_feature):
                    included[tmp_g.vertex(int(v))] = False
                    flag = True

        tmp_g.set_vertex_filter(included)
        if (not flag):
            break

    gt.draw.graphviz_draw(tmp_g, layout = 'twopi',
                          vcolor=vcolor,
                          size=(25,25),
                          vprops = {'label': vxlabel,
                                    'width': 1,
                          #          'shape': vshape,
                                    'penwidth': vpenwidth},
                          gprops = {'labelloc':'t',
                                    'fontsize': 36,
                                    'labelfontname': 'arial',
                                    'label':plot_title},
                          output = output)

def get_feature_annotation(root_dir, data, target):    
    working_dir = '%s/%s/%s/' % (root_dir, data, target)

    data_file = np.load(working_dir + '/npdata.npz')
    return(data_file['feature_annotation'])

def generate_graph_plots(root_dir, data, target, learner_count,
                         regularizer_index,
                         feature_list,
                         node_confidence,
                         cv_index,
                         threshold,
                         plot_titles = True):
    
            
    # feature extraction stuff
    working_dir = '%s/%s/%s/' % (root_dir, data, target)
    method = 'ratboost_linear_svc'
    #cv_index = 25

    print(working_dir, method, cv_index, file=sys.stderr)
    print("loading data...", file=sys.stderr)

    data_file = np.load(working_dir + '/npdata.npz')
    tmpX = data_file['tmpX']
    X_prime = data_file['X_prime']
    y = data_file['y']
    sample_annotation = data_file['sample_annotation']
    feature_annotation = data_file['feature_annotation']
    g = gt.load_graph(working_dir + '/graph.xml.gz')
    cvs = pickle.load(open(working_dir + '/cvs.dmp', 'rb'))

    #choosing only one cross-validation fold
    tmp = list()
    tmp.append((cvs[cv_index]))
    cvs = tmp

    #with open("./rat.py") as f:
    #    code = compile(f.read(), "rat.py", 'exec')
    #    exec(code)
    a = Rat(learner_count = learner_count,
        learner_type = 'linear svc',
        regularizer_index = regularizer_index,
        n_jobs = 1)

    train, test = cvs[0]
    a.fit(tmpX[train,], y[train])

    # I'm not interested in this plot I guess.
    #plot_graph(hilight_learner = 4, test_sample=tmpX[[0],:],
    #       color_scheme = 'learner')

    if (plot_titles):
        plot_learners(a, tmpX[[test[5]],:], feature_annotation,
                           plot_title = 'Test Sample Model: %s %s' % \
                           (data, target),
                           output = 'tmp/testdata-model-%s-%s-%d.eps' % \
                           (data, target, regularizer_index))
    else:
        plot_learners(a, tmpX[[test[5]],:], feature_annotation,
                           plot_title = '',
                           output = 'tmp/testdata-model-%s-%s-%d.eps' % \
                           (data, target, regularizer_index))


    if (plot_titles):
        plot_test_features_on_graph(a, g, tmpX[[test[5]],],
                                feature_annotation,
                                plot_title = 'Test Sample Features: %s %s' % \
                                (data, target),
                                output = 'tmp/testdata-features-PPI-%s-%s-%d.eps' \
                                % (data, target, regularizer_index))
    else:
        plot_test_features_on_graph(a, g, tmpX[[test[5]],],
                                feature_annotation,
                                plot_title = '',
                                output = 'tmp/testdata-features-PPI-%s-%s-%d.eps' \
                                % (data, target, regularizer_index))

    if (plot_titles):
        plot_features_on_graph(g, feature_list, node_confidence,
                           feature_annotation,
                           plot_title = 'High Confidence Features Over PPI: %s %s (%g)' % \
                           (data, target, threshold),
                           output = 'tmp/features-PPI-%s-%s-%d.eps' % \
                           (data, target, regularizer_index))
    else:
        plot_features_on_graph(g, feature_list, node_confidence,
                           feature_annotation,
                           plot_title = '',
                           output = 'tmp/features-PPI-%s-%s-%d.eps' % \
                           (data, target, regularizer_index))

'''
scores = cv.cross_val_score(
    a, tmpX, y,
    cv=5,
    scoring = 'roc_auc',
    n_jobs = 1,
    verbose=1)
print(np.average(scores))

machine = svm.NuSVC(nu=0.25,
    kernel='linear',
    verbose=False,
    probability=False)
machine.fit(tmpX[:60,], y[:60])
threshold = np.min(np.abs(machine.coef_)) + (np.max(np.abs(machine.coef_)) - np.min(np.abs(machine.coef_))) * 0.8
np.arange(machine.coef_.shape[1])[(abs(machine.coef_) > threshold).flatten()]


local_X = tmpX[train,]
local_y = y[train,]
index = 5
cs = sklearn.svm.l1_min_c(local_X, local_y, loss='l2') * np.logspace(0,5)
learner = sklearn.svm.LinearSVC(C = 0.001,
            penalty = 'l1',
            dual = False)
while (index < len(cs)):
    learner.set_params(C = cs[index])
    learner.fit(local_X, local_y)
    #if (len(self.getClassifierFeatures()) > 0):
    #    return(self.learner)
    index += 5
    print(sklearn.metrics.roc_auc_score(y[train],
                                        learner.decision_function(tmpX[train,])))
    print(sklearn.metrics.roc_auc_score(y[test],
                                        learner.decision_function(tmpX[test,])))
    print(index, cs[index])
    scores = learner.coef_.reshape(-1)
    print(len([i for i in range(len(scores)) if scores[i] != 0]))
return(self.learner)

sklearn.metrics.roc_auc_score(y[train], a.decision_function(tmpX[train,]))
'''
