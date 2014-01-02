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
from collections import defaultdict
from itertools import chain

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

#functions to read methylation data
def read_beta_values():
    #beta values are already saved in an npz file.
    tmp = np.load(Globals.beta_file)
    return tmp['header'], tmp['met_sites'], tmp['beta']

def read_methylation_annotation():
    tmp = read_csv(Globals.met_annot_file, skip_header=True, delimiter=',')
    tmp = [[row[i] for i in [1, 4, 9, 16, 17]] for row in tmp]
    tmp = np.array(tmp)
    tmp = tmp.view(dtype=[('TargetID', 'U367'),
                          ('CHR', 'U367'),
                          ('GeneNames', 'U367'),
                          ('snp_hit', 'U367'),
                          ('bwa_multi_hit', 'U367')])
                           
    tmp['CHR'][(tmp['CHR'] == 'X') | (tmp['CHR'] == 'Y')] = '23'
    tmp['CHR'][tmp['CHR'] == 'NA'] = '24'
    
    tmp = tmp.astype([('TargetID', 'U367'),
                      ('CHR', 'int32'),
                      ('GeneNames', 'U367'),
                      ('snp_hit', 'U367'),
                      ('bwa_multi_hit', 'U367')]).view(np.recarray)
    return tmp
    

if __name__ == '__main__':
    print('hi');
    # read PPI network.
    print('reading the network...')
    table = read_csv(Globals.ppi_file, True);
    refseq_ids = get_column(table, 0)
    refseq_ids.extend(get_column(table, 3));
    refseq_ids = list(set(refseq_ids));
    interactions = np.array(table)[:,[0,3]]
    del table

    #read beta values
    print('reading betas from binary dump...')
    [sample_names, met_sites, beta] = read_beta_values()
    beta = np.transpose(beta)

    print("replacing nans with medians...");
    tmp_masked = np.ma.masked_array(beta, [np.isnan(x) for x in beta]);
    medians = np.ma.median(tmp_masked, axis=0);
    beta = tmp_masked.filled(medians);
    del medians, tmp_masked

    print('reading methylation annotation data...')
    met_annot = read_methylation_annotation()
    site_idx = ((met_annot.CHR > 0) & (met_annot.CHR < 23) &
                (met_annot.snp_hit == 'FALSE') &
                (met_annot.bwa_multi_hit == 'FALSE')).reshape(-1)
    met_annot = met_annot[site_idx,]
    beta = beta[:, site_idx]
    met_sites = met_sites[site_idx,]
    probe_genes = set(chain(*chain(*met_annot.GeneNames.strip().split(';')))) - {''}
    # gene2met contains indices of met_annot for each gene, after slicing
    # using site_idx. the methylation data is also sliced using site_idx
    # to have the same indices as met_annot, to be able to use gene2met on it.
    gene2met = defaultdict(list)
    met2gene = defaultdict(set)
    for i in range(met_annot.shape[0]):
        genes = set(chain(*met_annot[i].GeneNames.strip().split(';'))) - {''}
        if (len(genes) > 0):
            met2gene[met_annot[i].TargetID[0]] = genes
            for gene in genes:
                gene2met[gene].append(i)
    
    print("refactoring betas into genename format...")
    print("\tAlso calculating expression median for multiple mapped probes")
    genes = refseq_ids
    expressions_colgenes = list()
    X = np.empty([beta.shape[0],0])
    for gene in genes:
        indices = gene2met[gene]
        if (len(indices) == 0):
            continue;
        expressions_colgenes.append(gene)
        new_col = np.median(beta[:,indices], axis=1)
        X = np.append(X, new_col.reshape([-1,1]), 1)
    del indices

    print("extracting common genes between expressions and network...");
    usable_interaction_indices = [i for i in range(interactions.shape[0])
                                  if interactions[i,0] in expressions_colgenes
                                  and interactions[i,1] in expressions_colgenes]
    interactions = interactions[usable_interaction_indices,:]
    del usable_interaction_indices

    print("creating graph from network data...");
    g = gt.Graph(directed=False);
    vlist = g.add_vertex(len(expressions_colgenes))
    for i in range(interactions.shape[0]):
        tmp_e = g.add_edge(expressions_colgenes.index(interactions[i,0]),
        expressions_colgenes.index(interactions[i,1]))
    del tmp_e, vlist

    print("reading sample descriptions and setting Y...")
    descriptions_raw = read_csv(Globals.sample_annotation_file, True, delimiter = ',')
    descriptions_array = np.array(descriptions_raw)
    # columns: id, sample.type, immunophenotype, subtype
    sample_annotation = descriptions_array[:,[0,3,5,6]]
    usable_samples = [i for i in range(sample_annotation.shape[0])
                           if sample_annotation[i,1] == 'diagnosis']
    del descriptions_raw, descriptions_array

    Y = np.empty(len(usable_samples), dtype=np.int32)
    Y[:] = 0
    Y[[i for i in range(Y.shape[0])
       if sample_annotation[usable_samples[i],2] == 'T-ALL']] = 1
    Y[[i for i in range(Y.shape[0])
       if sample_annotation[usable_samples[i],2] == 'BCP-ALL']] = -1
    samples = [i for i in range(Y.shape[0]) if Y[i] != 0]
    Y = Y[samples]
    usable_samples = [usable_samples[i] for i in samples]
    
    tmp_sample_names = sample_annotation[usable_samples,0]
    expression_sample_indices = [list(sample_names).index(tmp_sample_names[i])
                                      for i in range(tmp_sample_names.shape[0])]
    X = X[expression_sample_indices,:]

    print("calculating L and transformation of the data...")
    B = gt.spectral.laplacian(g)
    M = np.identity(B.shape[0]) + Globals.beta * B
    M_inv = np.linalg.inv(M)
    L = np.linalg.cholesky(M_inv)
    X_prime = X.dot(L)

    print("cross-validation...")
    cfolds = cv.StratifiedShuffleSplit(Y, n_iter=Globals.cfold_count, test_size=0.30,
                                       random_state=0)
    train_auc = list()
    test_auc = list()
    train_tr_auc = list()
    test_tr_auc = list()

    i = 0;
    for train_index, test_index in cfolds:
        machine = svm.NuSVC(nu=Globals.nu,
                            kernel='linear',
                            verbose=False,
                            probability=False)
        print(i)
        i = i + 1
        print('normal')
        train_data = X[train_index,:]
        train_labels = Y[train_index]
        test_data = X[test_index,:]
        test_labels = Y[test_index]
        machine.fit(train_data, train_labels)
        out = machine.predict(train_data)
        out_test = machine.predict(test_data)
        train_auc.append(roc_auc_score(train_labels, out))
        test_auc.append(roc_auc_score(test_labels, out_test))

        print('transformed')
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
