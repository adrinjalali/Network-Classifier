"""
    license: GPLv3.
    Adrin Jalali.
    March 2014, Saarbruecken, Germany.

read TCGA data.

you probably want to use only load_data function only!
"""

import pandas
import numpy as np
from sklearn import cross_validation as cv
import pickle


def load_data():
    print('reading data files')

    expression_data_file = '/home/adrin/Projects/Data/ICGC-LYMPH-DE/filtered_expressions.tsv'
    edata = pandas.read_csv(expression_data_file, sep='\t', low_memory=False)

    donor_info_file = '/home/adrin/Projects/Data/ICGC-LYMPH-DE/donor.tsv'
    donor_info = pandas.read_csv(donor_info_file, sep='\t')

    ensembl2genename_fle = '/home/adrin/Projects/Data/ICGC-LYMPH-DE/ensembl2genename.txt'
    ensembl2genename = pandas.read_csv(ensembl2genename_fle, sep='\t')

    #ppi_file = '/home/adrin/Projects/Data/PPI/HPRD/HPRD_Release9_062910/just_genes.csv'
    #ppi_data = pandas.read_csv(ppi_file, sep=',', header=0)

    print('setting labels')

    donor_info['label'] = ''
    FOLL_icd10 = ['C82', 'C82.0', 'C82.1', 'C82.2', 'C82.3', 'C82.4']
    for i in FOLL_icd10:
        donor_info.loc[donor_info["donor_diagnosis_icd10"] == i, 'label'] = 'FOLL'

    DLBCL_icd10 = ['C83.3']
    for i in DLBCL_icd10:
        donor_info.loc[donor_info["donor_diagnosis_icd10"] == i, 'label'] = 'DLBCL'

    donors = np.unique(edata['icgc_sample_id'])
    groups = edata.groupby(['icgc_donor_id', 'icgc_sample_id'])
    b = np.array(list(groups.groups.keys()))
    b = b[b[:, 0].argsort()]

    print('finding common columns')

    gene_set = None
    for i in range(b.shape[0]):
        if any(donor_info.loc[donor_info['icgc_donor_id'] == b[i, 0], 'label'] == 'DLBCL') \
            or any(donor_info.loc[donor_info['icgc_donor_id'] == b[i, 0], 'label'] == 'FOLL'):
            print(donor_info.loc[donor_info['icgc_donor_id'] == b[i, 0], ['icgc_donor_id', 'label']])
            print(edata.ix[groups.groups[(b[i, 0], b[i, 1])]].shape)

            sample_data = edata.ix[groups.groups[(b[i, 0], b[i, 1])]]
            if gene_set is None:
                gene_set = set(sample_data['gene_id'])
            else:
                gene_set = gene_set.intersection(sample_data['gene_id'])

            print('gene set size:', len(gene_set))

    print('constructing data matrix')

    data = pandas.DataFrame(columns=list(gene_set))
    donor_ids = list()
    for i in range(b.shape[0]):
        if any(donor_info.loc[donor_info['icgc_donor_id'] == b[i, 0], 'label'] == 'DLBCL') \
            or any(donor_info.loc[donor_info['icgc_donor_id'] == b[i, 0], 'label'] == 'FOLL'):
            print(donor_info.loc[donor_info['icgc_donor_id'] == b[i, 0], ['icgc_donor_id', 'label']])
            print(edata.ix[groups.groups[(b[i, 0], b[i, 1])]].shape)

            donor_ids.append(b[i, 0])
            sample_data = edata.ix[groups.groups[(b[i, 0], b[i, 1])]]
            tdict = {r['gene_id']: r['normalized_read_count'] for ri, r in sample_data.iterrows()
                     if r['gene_id'] in gene_set}
            data = data.append(tdict, ignore_index=True)

    data['donor_id'] = donor_ids
    data.set_index('donor_id', append=False, inplace=True, drop=True)

    print('gene to ensembl ID conversion')

    g2e = dict()
    for e in gene_set:
        g = np.unique(ensembl2genename.loc[ensembl2genename['Ensembl Gene ID'] == e.split('.')[0], 'Associated Gene Name'])
        if len(g) > 1:
            print('more than one mapped gene for %s: %s' % (e, g))
            continue
        if len(g) == 0:
            #print('no mapped gene found for ensembl id: %s' % e)
            continue

        g = g[0]
        if g in g2e:
            g2e[g].append(e)
        else:
            g2e[g] = [e]

    print('constructing final X,y matrix')
    X = np.array([np.mean(data[el], axis=1) for g, el in g2e.items()])
    X = pandas.DataFrame(np.transpose(X))
    X.columns = list(g2e.keys())
    X.index = data.index

    tmp = donor_info
    tmp.index = tmp['icgc_donor_id']
    y = tmp.ix[X.index]['label']

    print("X.shape: ", X.shape)
    print("y.shape: ", y.shape)
    return {'X': X, 'y': y}

if __name__ == '__main__':
    print("HI")
    data = load_data()
    data['X'].to_pickle('pandas_X.pickle')
    data['y'].to_pickle('pandas_y.pickle')
    cvs = cv.StratifiedShuffleSplit(data['y'], n_iter=50, test_size=0.2)
    pickle.dump(cvs, open('normal_cvs.dmp', 'wb'))
    print("BYE")
