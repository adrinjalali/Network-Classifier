#!/bin/bash
#usage: ./clearresults.sh target_folder

process_problem(){
    echo $1 $2 $3
    rm  $1/$2/$3/results/*
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

process_problem $1 TCGA-LAML-GeneExpression vital_status
process_problem $1 TCGA-LAML-GeneExpression risk_group
