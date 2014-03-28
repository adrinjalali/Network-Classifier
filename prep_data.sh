#!/bin/sh
#usage: ./prep_data.sh target_folder

process_problem(){
    echo $1 $2 $3
    mkdir -p $1/$2/$3/results
    python3.3 prep_data.py --data $2 --target $3 --dump-dir $1/$2/$3
}

process_problem $1 vantveer prognosis

process_problem $1 nordlund TvsB
process_problem $1 nordlund HeHvst1221

process_problem $1 TCGA-BRCA ER
process_problem $1 TCGA-BRCA T
process_problem $1 TCGA-BRCA N
process_problem $1 TCGA-BRCA stage

process_problem $1 TCGA-LAML vital_status
process_problem $1 TCGA-LAML risk_group

