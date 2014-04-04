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
                          size=(25,15),
                          vcolor=vcolor,
                          vcmap=plt.get_cmap('Blues'),
                          vprops = {'label': vxlabel,
                                    'shape': vshape,
                                    'height': vheight,
                                    'width': vwidth})

def plot_learners(test_sample):
    node_shapes = ['box', 'ellipse',
                   'circle', 'triangle', 'diamond',
                   'trapezium', 'parallelogram',
                   'house', 'pentagon', 'hexagon',
                   'septagon', 'octagon', 'invtriangle']
    
    tmp_g = gt.Graph(directed = True)
    vertex_map_name2index = dict()
    vertex_map_index2name = dict()
    vcolor = tmp_g.new_vertex_property('double')
    vwidth = tmp_g.new_vertex_property('string')
    vheight = tmp_g.new_vertex_property('string')
    vxlabel = tmp_g.new_vertex_property('string')
    vshape = tmp_g.new_vertex_property('string')

    confidences = dict()
    
    for i in range(len(a.learners)):
        print('========================')
        print(a.learners[i].getClassifierFeatures())

        cur_vertex, is_new = add_vertex(g = tmp_g,
            vertex_map_name2index = vertex_map_name2index,
            vertex_map_index2name = vertex_map_index2name,
            name = 'l_%d' % i)
        #vwidth[cur_vertex] = '1.5'
        #vheight[cur_vertex] = '2'
        vcolor[cur_vertex] = a.learners[i].getConfidence(test_sample) / 5
        vxlabel[cur_vertex] = 'l_%d' % i
        vshape[cur_vertex] = node_shapes[i]
        target_vertex = cur_vertex
        
        for gene in a.learners[i].getClassifierFeatures():
            cur_vertex, is_new = add_vertex(g = tmp_g,
                            vertex_map_name2index = vertex_map_name2index,
                            vertex_map_index2name = vertex_map_index2name,
                            name = gene)

            #vwidth[cur_vertex] = '1.5'
            #vheight[cur_vertex] = '2'
            vcolor[cur_vertex] = a.learners[i]._FCEs[gene].\
              getConfidence(test_sample)[0] * \
              a.learners[i].getClassifierFeatureWeights()[gene]
            vxlabel[cur_vertex] = feature_annotation[gene]
            vshape[cur_vertex] = node_shapes[i]

            tmp_g.add_edge(cur_vertex, target_vertex)

            '''
            parents = a.learners[i]._FCEs[gene].getFeatures()

            for parent in parents:
                cur_vertex, is_new = add_vertex(g = tmp_g,
                           vertex_map_name2index = vertex_map_name2index,
                           vertex_map_index2name = vertex_map_index2name,
                           name = parent)
                tmp_g.add_edge(
                    vertex_map_name2index[parent],
                    vertex_map_name2index[gene])
            '''
    gt.draw.graphviz_draw(tmp_g, layout='twopi',
                          size=(25,15),
                          vcolor=vcolor,
                          vcmap=plt.get_cmap('Blues'),
                          vprops = {'label': vxlabel,
                                    'shape': vshape,
                                    'height': vheight,
                                    'width': vwidth})
    
# feature extraction stuff
working_dir = '/scratch/TL/pool0/ajalali/ratboost/data_3/TCGA-LAML/risk_group/'
#working_dir = '/scratch/TL/pool0/ajalali/ratboost/data_3/TCGA-BRCA/N/'
method = 'ratboost_linear_svc'
cv_index = 70
#cv_index = 20
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

with open("./rat.py") as f:
    code = compile(f.read(), "rat.py", 'exec')
    exec(code)
a = Rat(learner_count = 10,
        learner_type = 'linear svc',
        C = 0.3,
        n_jobs = 15)
#a.fit(tmpX[:60,], y[:60])
train, test = cvs[0]
a.fit(tmpX[train,], y[train])

plot_graph(hilight_learner = 4, test_sample=tmpX[[0],:],
           color_scheme = 'learner')

plot_learners(test_sample=tmpX[[test[0]],:])


sm = mpl.cm.ScalarMappable(norm=mpl.cm.colors.Normalize(),
                      cmap=plt.get_cmap('spectral'))

machine = svm.NuSVC(nu=0.25,
    kernel='linear',
    verbose=False,
    probability=False)
machine.fit(tmpX[:60,], y[:60])
threshold = np.min(np.abs(machine.coef_)) + (np.max(np.abs(machine.coef_)) - np.min(np.abs(machine.coef_))) * 0.8
np.arange(machine.coef_.shape[1])[(abs(machine.coef_) > threshold).flatten()]
