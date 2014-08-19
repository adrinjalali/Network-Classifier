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

def read_methylation_annotation():
    tmp = read_csv(Globals.met_annot_file, skip_header=True, delimiter=',')
    tmp = [[row[i] for i in [1, 4, 9, 16, 17, 11]] for row in tmp]
    tmp = np.array(tmp)
    
    promoter_meta = set(['TSS200', 'TSS1500', "5'UTR", '1stExon'])
    promoter = [len(promoter_meta.intersection(row[5].split(';'))) > 0
                for row in tmp]

    boz = np.hstack((tmp, np.array(promoter).astype(int).reshape(-1,1)))

    tmp2 = boz.view(dtype=[('TargetID', 'U367'),
                          ('CHR', 'U367'),
                          ('GeneNames', 'U367'),
                          ('snp_hit', 'U367'),
                          ('bwa_multi_hit', 'U367'),
                          ('UCSC_REFGENE_REGION', 'U367'),
                          ('is_promoter', 'U367')])
    
    tmp2['CHR'][(tmp2['CHR'] == 'X') | (tmp2['CHR'] == 'Y')] = '23'
    tmp2['CHR'][tmp2['CHR'] == 'NA'] = '24'
    
    tmp3 = tmp2.astype([('TargetID', 'U367'),
                      ('CHR', 'int32'),
                      ('GeneNames', 'U367'),
                      ('snp_hit', 'U367'),
                      ('bwa_multi_hit', 'U367'),
                      ('UCSC_REFGENE_REGION', 'U367'),
                      ('is_promoter', 'bool')]).view(np.recarray)
    return tmp3

def networkize_illumina450k(X, probe_names):
    # read PPI network.
    print('reading the network...')
    table = read_csv(Globals.ppi_file, True);
    refseq_ids = get_column(table, 0)
    refseq_ids.extend(get_column(table, 3));
    refseq_ids = list(set(refseq_ids));
    interactions = np.array(table)[:,[0,3]]
    del table 
   
    print('reading methylation annotation data...')
    met_annot = read_methylation_annotation()

    print('coordinate met_annot rows with probe_names')
    tmp_list = list(met_annot.TargetID)
    tmp_indices = list()
    last_index = 0
    for i in range(len(probe_names)):
    #for i in range(100):
        try:
            index = next((j for j in range(last_index, len(tmp_list))
                          if tmp_list[j] == probe_names[i]))
        except StopIteration:
            index = next((j for j in range(0, len(tmp_list))
                          if tmp_list[j] == probe_names[i]))
        tmp_indices.append(index)
        last_index = index
    met_annot = met_annot[np.array(tmp_indices),]
    
    # should I filter probes according to ... ?
    '''
    site_idx = ((met_annot.CHR > 0) & (met_annot.CHR < 23) &
                (met_annot.snp_hit == 'FALSE') &
                (met_annot.bwa_multi_hit == 'FALSE')).reshape(-1)
    '''
    site_idx = met_annot.is_promoter.reshape(-1)
    met_annot = met_annot[site_idx,]
    X = X[:,site_idx]

    probe_genes = set(chain(*chain(*met_annot.GeneNames.strip().split(';')))) - {''}

    # gene2met contains indices of met_annot for each gene.
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
    tmpX = np.empty([X.shape[0],0])
    for gene in genes:
        indices = gene2met[gene]
        if (len(indices) == 0):
            continue;
        expressions_colgenes.append(gene)
        new_col = np.median(X[:,indices], axis=1)
        tmpX = np.append(tmpX, new_col.reshape([-1,1]), 1)
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

    return(tmpX, g, np.array(expressions_colgenes))

    
def load_data(target_output):
    patient_file = TCGA_root_dir + '/Clinical/Biotab/nationwidechildrens.org_clinical_patient_laml.txt'
    betas_file = TCGA_root_dir + '/betas.npz'
    processed_betas_file = TCGA_root_dir + '/betas-processed.npz'
    graph_dump_file = TCGA_root_dir + '/graph.xml.gz'
    
    if (os.path.isfile(betas_file)):
        data_file = np.load(betas_file)
        betas = data_file['betas']
        col_names = data_file['col_names']
        patient_data = data_file['patient_data']
        sample_names = data_file['sample_names']
        print('fount betas_file, shape: %s' % (betas.shape.__str__()))
    else:
        patient_skipped_lines = 3
    
        patient_data = np.array(read_csv(patient_file, skip_header = False))
        patient_data = patient_data[patient_skipped_lines:,]
        sample_names = patient_data[:,0]

        data_dir = TCGA_root_dir + '/DNA_Methylation/JHU_USC__HumanMethylation450/Level_3/'
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
                betas = np.empty((0,sample_col_names.shape[0]), dtype=float)
            else:
                if all(col_names == sample_col_names) == False:
                    raise RuntimeError("column names don't match")

            v = sample_data[data_skipped_lines:, 1]
            v[v == 'NA'] = -1
            v = np.array(v, dtype=float)
            v[v == -1] = np.nan
            betas = np.vstack((betas, v.reshape(1,-1)))

        indices = np.array([i for i in range(betas.shape[1])
                            if not any(np.isnan(betas[:,i]))])
        betas = betas[:,indices]
        col_names = col_names[indices]

        sample_indices = np.array([list(sample_names).index(used_samples[i])
                                   for i in range(len(used_samples))])
        patient_data = patient_data[sample_indices,:]
        np.savez(open(betas_file, 'wb'),
                 betas = betas, col_names = col_names,
                 patient_data = patient_data,
                 sample_names = sample_names)
    
    if (os.path.isfile(processed_betas_file)
        and os.path.isfile(graph_dump_file)):
        g = gt.load_graph(graph_dump_file)
        data_file = np.load(processed_betas_file)
        X = data_file['X']
        genes = data_file['genes']
        patient_data = patient_data
        print('processed data found, X: %s' % (X.shape.__str__()))
    else:
        X, g, genes = networkize_illumina450k(betas, col_names)
        print (X.__class__)
        print (genes.__class__)
        print (patient_data.__class__)
        print (g.__class__)
        np.savez(open(processed_betas_file, 'wb'),
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
    
