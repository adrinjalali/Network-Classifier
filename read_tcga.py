'''
    license: GPLv3.
    Adrin Jalali.
    March 2014, Saarbruecken, Germany.

read TCGA data.

you probably want to use only load_data function only!
'''

import numpy as np
import os
import pickle
import subprocess
import glob
import graph_tool as gt;
from itertools import chain
from collections import defaultdict
from sklearn import cross_validation as cv;

from misc import *
from constants import *

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
    #site_idx = met_annot.is_promoter.reshape(-1)
    #met_annot = met_annot[site_idx,]
    #X = X[:,site_idx]

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

def load_450k_methylation(data_dir, patient_codes, sample_type):
    files = os.listdir(data_dir)
        
    suffix = sample_type

    col_names = np.empty(0)
    used_samples = np.empty(0)
    unused_samples = np.empty(0)
    multiple_data_samples = np.empty(0)

    i = 0
    for name in patient_codes:
        i += 1
        print('processing %3d/%3d ' %(i, len(patient_codes)) + name)
        matched = [f for f in files if f.find(name+'-'+suffix) > -1]
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

    sample_indices = np.array([list(patient_codes).index(used_samples[i])
        for i in range(len(used_samples))])

    return (sample_indices, col_names, betas,
        {'unused_samples': unused_samples,
        'multiple_data_samples': multiple_data_samples})

'''
    X is the main matrix.
    target_labels: {'vital_status': {-1: 'Dead', 1: 'Alive'},...}
    patient_annot and patient_annot_colnames are patient annotation file data
    and its column names
    sample_indices are the indices of the patient_annot that make the X matrix
    dump_dir is the output dir, in which a folder for each key in target_labels
    will be created.
    The function returns a dictionary having the same set of keys as
    target_labels, and on each member of the dictionary we have:
    (X, y, patient_annot, original_sample_indices)
    it also saves the cross validations in two files, one for random 100 x 80% vs 20%
    and one for batch based cross validation
'''
def dump_by_target_label(X, target_labels, patient_annot,
                            patient_annot_colnames, sample_indices, L,
                            dump_dir):
    result = dict()
    for key, value in target_labels.items():
        print (key)
        print(value)
        target_index = list(patient_annot_colnames).index(key)
        tmp_annot = patient_annot[sample_indices,:]
        labels = tmp_annot[:,target_index]
        y = np.zeros(len(sample_indices), dtype=int)

        '''
        if no label is a prefix of another label, then I'll lookup for them
        in labels with "startswith", otherwise will be exact match.
        '''
        vague = False
        for jkey in value.keys():
            if isinstance(value[jkey], list):
                jvalue = value[jkey]
            else:
                jvalue = [value[jkey]]
                
            for kkey in value.keys():
                if jkey <= kkey:
                    continue
                print(jkey, kkey)
                
                if isinstance(value[kkey], list):
                    kvalue = value[kkey]
                else:
                    kvalue = [value[kkey]]

                print(jvalue, kvalue)

                for jl in jvalue:
                    for kl in kvalue:
                        if (jl.startswith(kl) or kl.startswith(jl)):
                            vague = True

        print('labels are vage:', vague)

        for jkey, jvalue in value.items():
            if (jkey == 0):
                print("class label 0 is not allowed, maybe you meant 1,-1?")
                continue
            if isinstance(jvalue, list):
                for t_label in jvalue:
                    print(jkey, t_label)
                    if vague:
                        y[labels == t_label] = jkey
                    else:
                        y[np.array([l.startswith(t_label) for l in labels])] = jkey
            else:
                print(jkey, jvalue)
                if vague:
                    y[labels == jvalue] = jkey
                else:
                    y[np.array([l.startswith(jvalue) for l in labels])] = jkey
        
        final_sample_indices = (y != 0)

        tmp_y = y[final_sample_indices]
        tmp_X = X[final_sample_indices,]
        tmp_annot = tmp_annot[final_sample_indices,:]
        tmp_sample_indices = sample_indices[final_sample_indices]

        if L is not None:
            X_prime = tmp_X.dot(L)
        else:
            X_prime = None

        tmp_dump_dir = dump_dir + '/' + key
        if (not os.path.exists(tmp_dump_dir)):
            os.mkdir(tmp_dump_dir)
        np.savez(open(tmp_dump_dir + '/data.npz', 'wb'),
                 X = tmp_X, X_prime = X_prime, y = tmp_y,
                 patient_annot = tmp_annot,
                 original_sample_indices = tmp_sample_indices)
        result[key] = (tmp_X, tmp_y, tmp_annot, tmp_sample_indices)

        '''
        save cross validation sets, both batch based and random.
        '''        
        cvs = list(cv.StratifiedShuffleSplit(tmp_y, n_iter = 100, test_size = 0.2))
        pickle.dump(cvs, open(tmp_dump_dir + '/normal_cvs.dmp', 'wb'))
        
        sample_batches = tmp_annot[:,
                list(patient_annot_colnames).index('batch_number')]
        batches = np.unique(sample_batches)

        cvs = list()
        for i in range(len(batches)):
            print('batch size:', sum(sample_batches == batches[i]))
            cvs.append((np.arange(len(sample_batches))[sample_batches != batches[i]],
                      np.arange(len(sample_batches))[sample_batches == batches[i]]))
        pickle.dump(cvs, open(tmp_dump_dir + '/batch_cvs.dmp', 'wb'))
        
    return result


'''
This function loads data from the input_dir. This input_dir is supposed to have
a folder for each data type, for example DNA_Methylation, and a folder for
clinical data, which is named Clinical. These are the standard folder structure
of the TCGA data. The function will find the patient information and the data.
In Clinical section, both xml and biotab data types are required. XMLs are mostly
used for extracting the batch information in this code.
The function returns the output of dump_by_target_label, as well as the PPI graph
and the list of gene names of the graph.
target_labels is of the shape {'vital_status': {-1: 'Dead', 1: 'Alive'},...}
sample_type shows the type of the sample, for example main tumor, and is a prefix
to the patient code, for example 01A.
patient_annot_file can easily be found from the Clinical folder, no need to be given.
'''
def load_data(input_dir,
              target_labels, sample_type=None, patient_annot_file=None,
              final_dump_folder = None, networkize_data = False):
    if (sample_type == None):
        print("sample type must be given. For example 01A (as suffix to patient codes.)")
        return
    
    dump_dir = input_dir + '/processed'
    if (not os.path.exists(dump_dir)):
        os.mkdir(dump_dir)

    if (patient_annot_file == None):
        patient_file_candidates = glob.glob(input_dir + '/Clinical/Biotab/nationwidechildrens.org_clinical_patient*.txt')
        if (len(patient_file_candidates) != 1):
            print('ERROR: patient_file_candidates: ', patient_file_candidates)
            return(None)
        patient_annot_file = patient_file_candidates[0]

    patient_annot_processed_file = dump_dir + '/patient_annot.npz'
    betas_file = dump_dir + '/betas.npz'
    processed_betas_file = dump_dir + '/betas-processed.npz'
    gene_annot_file = dump_dir + '/genes.npz'
    graph_dump_file = dump_dir + '/graph.xml.gz'
    calculated_L_matrix = dump_dir + '/L.npz'

    '''
        here we load the annotation and batch information of the samples
    '''
    if (os.path.isfile(patient_annot_processed_file)):
        data_file = np.load(patient_annot_processed_file)
        patient_annot = data_file['patient_annot']
        patient_annot_colnames = data_file['patient_annot_colnames']
        patient_codes = data_file['patient_codes']
    else:
        patient_skipped_lines = 3
    
        patient_data = np.array(read_csv(patient_annot_file, skip_header = False))
        patient_annot_colnames = patient_data[0,:]
        patient_annot = patient_data[patient_skipped_lines:,]
        patient_codes = patient_data[patient_skipped_lines:,0]

        xml_dir = input_dir + '/Clinical/XML'

        '''
        here I look for the admin:batch_number key in xml files of the patients,
        extract that line, remove extra stuff with sed, and get a two column text
        with patient ids and batch numbers.
        '''
        output = subprocess.check_output("grep \"admin:batch_number xsd_ver=\" %s/*_clinical*.xml | awk '{print $1 \"\t\" $3}' | sed \"s/.*clinical\.//g\" | sed \"s/\.xml:\t.*\\\">/\t/g\" | sed \"s/\..*//g\"" % (xml_dir),
                                         shell=True,
                                         universal_newlines=True).splitlines()
        patient_batches_dict = {output[i].split('\t')[0]:output[i].split('\t')[1]
                                for i in range(len(output))}

        patient_batches = np.zeros(len(patient_codes), dtype=int)
        for i in range(len(patient_codes)):
            patient_batches[i] = patient_batches_dict[patient_codes[i]]
        patient_annot = np.hstack((patient_annot, patient_batches.reshape(-1,1)))
        patient_annot_colnames = np.append(patient_annot_colnames, 'batch_number')
        
        np.savez(open(patient_annot_processed_file, 'wb'),
                 patient_annot = patient_annot,
                 patient_annot_colnames = patient_annot_colnames,
                 patient_codes = patient_codes)
            

    '''
        in this section the methylation beta values are extracted and put into
        a matrix loaded from 450k illumina chip.
    '''
    if (os.path.isfile(betas_file)):
        data_file = np.load(betas_file)
        betas = data_file['betas']
        col_names = data_file['col_names']
        sample_indices = data_file['methylation_45k_sample_indices']
        print('fount betas_file, shape: %s' % (betas.shape.__str__()))
    else:
        data_dir = input_dir + '/DNA_Methylation/JHU_USC__HumanMethylation450/Level_3/'
        if (os.path.exists(data_dir)):
            sample_indices, col_names, betas, debug_info = \
                load_450k_methylation(data_dir, patient_codes, sample_type)
            print(debug_info)
                
            np.savez(open(betas_file, 'wb'),
                 betas = betas, col_names = col_names,
                 methylation_45k_sample_indices = sample_indices)


    """
    Don't use the PPI network if no network is needed, and return raw
    beta values.
    """
    if not networkize_data:
        processed_data = dump_by_target_label(betas, target_labels, patient_annot,
                        patient_annot_colnames, sample_indices, None,
                        dump_dir)

        return (processed_data, None, col_names)
    
    '''
        use the graph to map nodes to genes and get the graph itself.
    '''
    if (os.path.isfile(processed_betas_file)
        and os.path.isfile(graph_dump_file)
        and os.path.isfile(gene_annot_file)):
        g = gt.load_graph(graph_dump_file)
        data_file = np.load(processed_betas_file)
        X = data_file['X']
        data_file = np.load(gene_annot_file)
        genes = data_file['genes']
        print('processed data found, X: %s' % (X.shape.__str__()))
    else:
        X, g, genes = networkize_illumina450k(betas, col_names)
        print (X.__class__)
        print (genes.__class__)
        print (g.__class__)
        np.savez(open(processed_betas_file, 'wb'), X = X)
        np.savez(open(gene_annot_file, 'wb'), genes=genes)
        g.save(graph_dump_file)


    if (os.path.isfile(calculated_L_matrix)):
        data_file = np.load(calculated_L_matrix)
        L = data_file['L']
        print('fount L matrix, shape: %s' % (L.shape.__str__()))
    else:
        print("calculating L and transformation of the data...")
        B = gt.spectral.laplacian(g)
        M = np.identity(B.shape[0]) + Globals.beta * B
        M_inv = np.linalg.inv(M)
        L = np.linalg.cholesky(M_inv)
        np.savez(open(calculated_L_matrix, 'wb'),
                 L = L)
        
    if (final_dump_folder != None):
        dump_dir = final_dump_folder
        
    processed_data = dump_by_target_label(X, target_labels, patient_annot,
                        patient_annot_colnames, sample_indices, L,
                        dump_dir)

    return (processed_data, g, genes)
    
