'''
    License: GPLv3.
    Adrin Jalali.
    Jan 2014, Saarbruecken, Germany.

read Van't Veer microarray data got from Ofer.
'''
import graph_tool as gt;
from collections import defaultdict
import numpy as np
import bidict;

from constants import *
from misc import *

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


def load_data():
    '''
    This function loads the data as (X, y), and also creates a graph
    using the PPI network data.
    the global Globals object has everything this function needs to
    read the data.

    The output is (X, y, graph, sample_annotation, feature_annotation)
    '''
    print('reading PPI network...')
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
    node_indices = {common_genes_list[x]:x
                                   for x in range(len(common_genes_list))}
    g = gt.Graph(directed=False);
    vlist = g.add_vertex(len(node_indices))
    for i in range(interactions.shape[0]):
        tmp_e = g.add_edge(node_indices[interactions[i,0]],
                           node_indices[interactions[i,1]])

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

    return (X, Y, g, descriptions_array[samples,].view(np.ndarray), np.array(common_genes_list))
