class Globals:
    data_root = '/TL/stat_learn/work/ajalali/Data/Data-leni/'
    nordlund_data = '/TL/stat_learn/work/ajalali/Nordlund-Backlin-2013/data/'
    ppi_file = data_root + '/PPI Networks/HPRD/HPRD_Release9_062910/BINARY_PROTEIN_PROTEIN_INTERACTIONS.txt'
    genename2entrez_file = data_root + '/PPI Networks/HPRD/HPRD_Release9_062910/GeneName2EntrezID.txt'

    #for Ofer data analysis - microarray expressions
    expressions_file = data_root + '/From Ofer/data/VantVeer02.txt'
    probe2gene_file = data_root + '/From Ofer/data/VantVeer02_probe2gene.txt'
    description_file = data_root + '/From Ofer/data/VantVeer02_characteristics.txt'

    #for Nordlund data analysis - methylation data
    beta_file = nordlund_data + '/betas.npz'
    pval_file = nordlund_data + '/pvals.csv'
    sample_annotation_file = nordlund_data + '/samples.csv'
    met_annot_file = nordlund_data + '/met_annot.csv'
    
    beta = 10
    cfold_count = 100
    nu = 0.25
