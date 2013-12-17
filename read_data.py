import sys;
import os;
import csv;
import numpy as np;
from constants import *;
import graph_tool as gt;
from graph_tool import draw;
from graph_tool import spectral;
from graph_tool import stats;
from sklearn import svm;
from sklearn import cross_validation as cv;
from sklearn.metrics import roc_auc_score;
import bidict;
#from itertools import izip;

def read_csv(file_name, skip_header, delimiter = '\t'):
    file = open(file_name, 'r');
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

def get_gene_expression_indices(entrezid, expressions_colnames, probe2gene_array):
    probes = probe2gene_array[np.where(probe2gene_array[:,1] ==
                                       entrezid),0].reshape(-1)
    probes = [x for x in probes if x in expressions_colnames]
    indices = [i for i in range(expressions_colnames.shape[0])
               if expressions_colnames[i] in probes]
    return indices;

def get_genename_entrezids(genename, genename2entrez_array):
    entrezids = genename2entrez_array[np.where(genename2entrez_array[:,0] ==
                                               genename),1].reshape(-1)
    return entrezids;
    
if __name__ == '__main__':
    print('hi');
    # read PPI network.
    table = read_csv(Globals.ppi_file, True);
    refseq_ids = get_column(table, 0)
    refseq_ids.extend(get_column(table, 3));
    refseq_ids = list(set(refseq_ids));
    interactions = np.array(table)[:,[0,3]]

    #dump_list(refseq_ids, 'refseq_ids.txt');
    genename2entrez_raw = read_csv(Globals.genename2entrez_file, True);
    genename2entrez_array = np.array(genename2entrez_raw)[:,[0,2]];
    genename2entrez = {}
    entrez2genename = {}
    for row in genename2entrez_raw:
        genename2entrez[row[0]] = row[2];
        #entrez2genename[row[2]] = row[0];

    expressions = read_csv(Globals.expressions_file, False);

    probe2gene_raw = read_csv(Globals.probe2gene_file, False);
    probe2gene_array = np.array(probe2gene_raw);
    probe2gene = {}
    gene2probe = {}
    for row in probe2gene_raw:
        probe2gene[row[0]] = row[1];
        #gene2probe[row[1]] = row[0];

    tmp = np.asarray(expressions);
        
    expressions_array = tmp[1:,1:].T;
    expressions_colnames = tmp[1:,0];
    expressions_rownames = tmp[0,1:];

    print("converting expression values to floats...");
    tmp = np.empty(expressions_array.shape);
    for i in range(tmp.shape[0]):
        tmp[i,:] = [(np.nan if len(x.strip()) == 0 else float(x))
                    for x in expressions_array[i,:]]

    print("replacing nans with medians...");
    tmp_masked = np.ma.masked_array(tmp, [np.isnan(x) for x in tmp]);
    medians = np.ma.median(tmp_masked, axis=0);
    expressions_array = tmp_masked.filled(medians);

    print("refactoring interactions into entrezid format...")
    new_interactions = np.empty([0,2])
    for i in range(interactions.shape[0]):
        gene_1_list = get_genename_entrezids(interactions[i,0],
                                             genename2entrez_array)
        gene_2_list = get_genename_entrezids(interactions[i,1],
                                             genename2entrez_array)
        if gene_1_list.shape[0] > 0 and gene_2_list.shape[0] > 0:
            for g1 in gene_1_list:
                for g2 in gene_2_list:
                    new_interactions = np.append(new_interactions, [[g1, g2]], 0);

    print("refactoring expressions into entrezid format...")
    print("\tAlso calculating expression median for multiple mapped probes")
    genes = set(genename2entrez_array[:,1])
    expressions_colgenes = list()
    X = np.empty([expressions_array.shape[0],0])
    for entrezid in genes:
        indices = get_gene_expression_indices(entrezid,
                                              expressions_colnames,
                                              probe2gene_array)
        if (len(indices) == 0):
            continue;
        expressions_colgenes.append(entrezid)
        new_col = np.median(expressions_array[:,indices], axis=1)
        X = np.append(X, new_col.reshape([-1,1]), 1)

    print("extracting common genes between expressions and network...");
    usable_interaction_indices = [i for i in range(new_interactions.shape[0])
                                  if new_interactions[i,0] in expressions_colgenes
                                  and new_interactions[i,1] in expressions_colgenes]
    common_genes = set(new_interactions[usable_interaction_indices,:].
                       reshape(-1)).intersection(set(expressions_colgenes))
    interactions = new_interactions[usable_interaction_indices,:]

    print("rearrange expressions array into X...")
    common_genes_list = list(common_genes)
    rearrange_indices = [expressions_colgenes.index(gene)
                         for gene in common_genes_list]
    X = X[:,rearrange_indices]

    print("creating graph from network data...");
    node_indices_t = bidict.namedbidict('node_indices_t', 'entrezids', 'indices')
    node_indices = node_indices_t({common_genes_list[x]:x
                                   for x in range(len(common_genes_list))})
    g = gt.Graph(directed=False);
    vlist = g.add_vertex(len(node_indices))
    for i in range(interactions.shape[0]):
        tmp_e = g.add_edge(node_indices.entrezids[interactions[i,0]],
                           node_indices.entrezids[interactions[i,1]])

    print("reading sample descriptions and setting Y...")
    descriptions_raw = read_csv(Globals.description_file, True)
    descriptions_array = np.array(descriptions_raw)
    Y = np.empty([descriptions_array.shape[0]], dtype=np.int32)
    Y[:] = 0
    Y[[i for i in range(Y.shape[0]) if descriptions_array[i,15] == 'Good']] = 1
    Y[[i for i in range(Y.shape[0]) if descriptions_array[i,15] == 'Poor']] = -1
    samples = [i for i in range(Y.shape[0]) if Y[i] != 0]
    Y = Y[samples]
    sample_names = descriptions_array[samples,0]
    expression_sample_indices = [list(expressions_rownames).index(sample_names[i])
                                      for i in range(sample_names.shape[0])]
    X = X[expression_sample_indices,:]

    print("calculating L and transformation of the data...")
    B = gt.spectral.laplacian(g)
    M = np.identity(B.shape[0]) + Globals.beta * B
    M_inv = np.linalg.inv(M)
    L = np.linalg.cholesky(M_inv)
    X_prime = X.dot(L)

    print("cross-validation...")
    cfolds = cv.StratifiedShuffleSplit(Y, n_iter=Globals.cfold_count, test_size=0.2,
                                       random_state=0)
    train_auc = list()
    test_auc = list()
    train_tr_auc = list()
    test_tr_auc = list()

    for train_index, test_index in cfolds:
        machine = svm.NuSVC(nu=Globals.nu,
                            kernel='linear',
                            verbose=False,
                            probability=False)
        train_data = X[train_index,:]
        train_labels = Y[train_index]
        test_data = X[test_index,:]
        test_labels = Y[test_index]
        machine.fit(train_data, train_labels)
        out = machine.predict(train_data)
        out_test = machine.predict(test_data)
        train_auc.append(roc_auc_score(train_labels, out))
        test_auc.append(roc_auc_score(test_labels, out_test))

        train_data = X_prime[train_index,:]
        train_labels = Y[train_index]
        test_data = X_prime[test_index,:]
        test_labels = Y[test_index]
        machine.fit(train_data, train_labels)
        out = machine.predict(train_data)
        out_test = machine.predict(test_data)
        train_tr_auc.append(roc_auc_score(train_labels, out))
        test_tr_auc.append(roc_auc_score(test_labels, out_test))

    print("test auc: ", np.mean(test_auc))
    print("test transformed auc: ", np.mean(test_tr_auc))
    print("train auc: ", np.mean(train_auc))
    print("train transformed auc: ", np.mean(train_tr_auc))

    writer = csv.writer(open("results.csv", "w"))
    writer.writerow(['test_auc', 'train_auc', 'test_tr_auc', 'train_tr_auc'])
    for i in range(len(test_auc)):
        writer.writerow([test_auc[i], train_auc[i],
                        test_tr_auc[i], train_tr_auc[i]])

    print("constructing edge feature vector...")
    edge_X = np.empty([X.shape[0], g.num_edges()], dtype=np.float64)
    i = 0
    for e in g.edges():
        edge_X[:,i] = np.mean([X[:,int(e.source())], X[:,int(e.target())]],
                              axis=0)
        i += 1


    train_auc = list()
    test_auc = list()
    for train_index, test_index in cfolds:
        machine = svm.NuSVC(nu=Globals.nu,
                            kernel='linear',
                            verbose=False,
                            probability=False)
        train_data = edge_X[train_index,:]
        train_labels = Y[train_index]
        test_data = edge_X[test_index,:]
        test_labels = Y[test_index]
        machine.fit(train_data, train_labels)
        out = machine.predict(train_data)
        out_test = machine.predict(test_data)
        train_auc.append(roc_auc_score(train_labels, out))
        test_auc.append(roc_auc_score(test_labels, out_test))

    print("edge features test auc: ", np.mean(test_auc))
    print("edge features train auc: ", np.mean(train_auc))
    print 'bye';
