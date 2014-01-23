'''
    License: GPLv3.
    Adrin Jalali.
    Jan 2014, Saarbruecken, Germany.

read Nordlund data for set A subtype vs set B classification.

original publicaion for the data:
Genome-wide signatures of differential DNA methylation in
pediatric acute lymphoblastic leukemia
'''

from collections import defaultdict
import numpy as np
from itertools import chain
import graph_tool as gt;

from constants import *
from misc import *


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


def load_data(A, B):
    '''
    This function loads the data as (X, y), and also creates a graph
    using the PPI network data.
    the global Globals object has everything this function needs to
    read the data.
    variable y would be 1 for subtypes in A and -1 for B subtypes.
    '''

    if (isinstance(A, list)):
        A = set(A)
    if (isinstance(B, list)):
        B = set(B)

    
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

    print("reading sample descriptions and setting Y...")
    descriptions_raw = read_csv(Globals.sample_annotation_file, True, delimiter = ',')
    descriptions_array = np.array(descriptions_raw)
    # columns: id, sample.type, immunophenotype, subtype
    sample_annotation = descriptions_array[:,[0,3,5,6]]
    usable_samples = [i for i in range(sample_annotation.shape[0])
                           if sample_annotation[i,1] == 'diagnosis']
    del descriptions_raw, descriptions_array

    Y = np.empty(len(usable_samples), dtype=int)
    Y[:] = 0
    Y[[i for i in range(Y.shape[0])
       if sample_annotation[usable_samples[i],3] in A]] = 1
    Y[[i for i in range(Y.shape[0])
       if sample_annotation[usable_samples[i],3] in B]] = -1
    samples = [i for i in range(Y.shape[0]) if Y[i] != 0]
    Y = Y[samples]
    usable_samples = [usable_samples[i] for i in samples]
    
    tmp_sample_names = sample_annotation[usable_samples,0]
    expression_sample_indices = [list(sample_names).index(tmp_sample_names[i])
                                      for i in range(tmp_sample_names.shape[0])]
    X = X[expression_sample_indices,:]
                           
    print("creating graph from network data...");
    g = gt.Graph(directed=False);
    vlist = g.add_vertex(len(expressions_colgenes))
    for i in range(interactions.shape[0]):
        tmp_e = g.add_edge(expressions_colgenes.index(interactions[i,0]),
        expressions_colgenes.index(interactions[i,1]))
    del tmp_e, vlist

    sample_annotation = sample_annotation[np.array(usable_samples),].view(np.ndarray)

    return (X, Y, g, sample_annotation)