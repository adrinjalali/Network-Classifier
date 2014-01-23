'''
    License: GPLv3.
    Adrin Jalali.
    Jan 2014, Saarbruecken, Germany.

miscellaneous functions used mostly in the data loading/preprocessing
phase.
'''

import csv;
import sys;
import os;

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

def print_stats(mine):
    print("MIC", mine.mic())
    print("MAS", mine.mas())
    print("MEV", mine.mev())
    print("MCN (eps=0)", mine.mcn(0))
    print("MCN (eps=1-MIC)", mine.mcn_general())
                        
