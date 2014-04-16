'''
    License: GPLv3.
    Adrin Jalali.
    March 2014, Saarbruecken, Germany.

read TCGA-BRCA data.

'''

import numpy as np
import os
import graph_tool as gt;
from itertools import chain
from collections import defaultdict

from misc import *
from constants import *


TCGA_root_dir = "/TL/stat_learn/work/ajalali/Data/TCGA-LAML"

def networkize_illuminaU133(X, gene_names):
    # read PPI network.
    print('reading the network...')
    table = read_csv(Globals.ppi_file, True);
    refseq_ids = get_column(table, 0)
    refseq_ids.extend(get_column(table, 3));
    refseq_ids = list(set(refseq_ids));
    interactions = np.array(table)[:,[0,3]]
    del table
    
    print("extracting common genes between expressions and network...");
    usable_interaction_indices = [i for i in range(interactions.shape[0])
                                  if interactions[i,0] in gene_names
                                  and interactions[i,1] in gene_names]
    interactions = interactions[usable_interaction_indices,:]
    del usable_interaction_indices

    genes = list(np.union1d(interactions[:,0], interactions[:,1]))
    gene_names = list(gene_names)
    gene_idx = [gene_names.index(genes[i]) for i in range(len(genes))]
    tmpX = X[:,gene_idx]

    print("creating graph from network data...");
    g = gt.Graph(directed=False);
    vlist = g.add_vertex(len(genes))
    for i in range(interactions.shape[0]):
        tmp_e = g.add_edge(genes.index(interactions[i,0]),
            genes.index(interactions[i,1]))
    del tmp_e, vlist

    return(tmpX, g, np.array(genes))


def load_data(target_output):
    patient_file = TCGA_root_dir + '/Clinical/Biotab/nationwidechildrens.org_clinical_patient_laml.txt'
    expressions_file = TCGA_root_dir + '/expressions.npz'
    processed_expressions_file = TCGA_root_dir + '/expressions-processed.npz'
    graph_dump_file = TCGA_root_dir + '/graph-geneexpression.xml.gz'
    
    if (os.path.isfile(expressions_file)):
        data_file = np.load(expressions_file)
        expressions = data_file['expressions']
        col_names = data_file['col_names']
        patient_data = data_file['patient_data']
        sample_names = data_file['sample_names']
        print('fount expressions_file, shape: %s' % (expressions.shape.__str__()))
    else:
        patient_skipped_lines = 3
    
        patient_data = np.array(read_csv(patient_file, skip_header = False))
        patient_data = patient_data[patient_skipped_lines:,]
        sample_names = patient_data[:,0]

        data_dir = TCGA_root_dir + '/Expression-Genes/WUSM__HG-U133_Plus_2/Level_3/'
        files = os.listdir(data_dir)

        col_names = np.empty(0)
        used_samples = np.empty(0)
        unused_samples = np.empty(0)
        multiple_data_samples = np.empty(0)

        i = 0
        for name in sample_names:
            i += 1
            print('processing %3d/%3d ' %(i, len(sample_names)) + name)
            # 03A : Primary Blood Derived Cancer - Peripheral Blood
            matched = [f for f in files if f.find(name+'-03A') > -1]
            if (len(matched) > 1):
                multiple_data_samples = np.append(multiple_data_samples, name)
                continue
            elif len(matched) == 0:
                print('no files found.')
                unused_samples = np.append(unused_samples, name)
                continue

            used_samples = np.append(used_samples, name)
            matched = matched[0]

            sample_data = np.array(read_csv(data_dir +
                                            matched, skip_header = False))
            data_skipped_lines = 2
            
            sample_col_names = sample_data[data_skipped_lines:,0]

            if col_names.shape[0] == 0:
                col_names = sample_col_names
                expressions = np.empty((0,sample_col_names.shape[0]), dtype=float)
            else:
                if all(col_names == sample_col_names) == False:
                    raise RuntimeError("column names don't match")

            v = sample_data[data_skipped_lines:, 1]
            v[v == 'NA'] = -1
            v = np.array(v, dtype=float)
            v[v == -1] = np.nan
            expressions = np.vstack((expressions, v.reshape(1,-1)))

        indices = np.array([i for i in range(expressions.shape[1])
                            if not any(np.isnan(expressions[:,i]))])
        expressions = expressions[:,indices]
        col_names = col_names[indices]

        sample_indices = np.array([list(sample_names).index(used_samples[i])
                                   for i in range(len(used_samples))])
        patient_data = patient_data[sample_indices,:]
        np.savez(open(expressions_file, 'wb'),
                 expressions = expressions, col_names = col_names,
                 patient_data = patient_data,
                 sample_names = sample_names)
    
    if (os.path.isfile(processed_expressions_file)
        and os.path.isfile(graph_dump_file)):
        g = gt.load_graph(graph_dump_file)
        data_file = np.load(processed_expressions_file)
        X = data_file['X']
        genes = data_file['genes']
        patient_data = patient_data
        print('processed data found, X: %s' % (X.shape.__str__()))
    else:
        X, g, genes = networkize_illuminaU133(expressions, col_names)
        print (X.__class__)
        print (genes.__class__)
        print (patient_data.__class__)
        print (g.__class__)
        np.savez(open(processed_expressions_file, 'wb'),
                 X = X, genes=genes,patient_data=patient_data)
        g.save(graph_dump_file)

    if (target_output == 'risk_group'):
        # cyto_risk_group status is column index 50
        labels = patient_data[:,50]
        y = np.zeros(len(patient_data), dtype=int)
        y[labels == 'Favorable'] = -1
        y[labels == 'Intermediate/Normal'] = 1
        y[labels == 'Poor'] = 1
        
        final_sample_indices = (y != 0)

        y = y[final_sample_indices]
        X = X[final_sample_indices,]
        patient_data = patient_data[final_sample_indices,:]
    elif (target_output == 'vital_status'):
        # vital_status status is column index 15
        labels = patient_data[:,15]
        y = np.zeros(len(patient_data), dtype=int)
        y[labels == 'Alive'] = -1
        y[labels == 'Dead'] = 1
        
        final_sample_indices = (y != 0)

        y = y[final_sample_indices]
        X = X[final_sample_indices,]
        patient_data = patient_data[final_sample_indices,:]
    else:
        raise RuntimeError("target_output not in ('risk_group', 'vital_status')")

    return (X, y, g, patient_data, genes)
    
