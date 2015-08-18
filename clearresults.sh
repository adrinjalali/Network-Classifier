#!/bin/bash
#usage: ./clearresults.sh target_folder

process_problem(){
    echo $1 $2 $3
    #rm  $1/$2/$3/results/*rat*
    rm  $1/$2/$3/results/*raccoon*
    #rm  $1/$2/$3/models/*
}

#process_problem $1 vantveer prognosis

#process_problem $1 nordlund TvsB
#process_problem $1 nordlund HeHvst1221

#process_problem $1 TCGA-LAML vital_status
#process_problem $1 TCGA-LAML risk_group

#process_problem $1 TCGA-LAML-GeneExpression vital_status
#process_problem $1 TCGA-LAML-GeneExpression risk_group

process_problem $1 TCGA-BRCA er_status_by_ihc
process_problem $1 TCGA-BRCA ajcc_pathologic_tumor_stage
process_problem $1 TCGA-BRCA ajcc_tumor_pathologic_pt
process_problem $1 TCGA-BRCA ajcc_nodes_pathologic_pn

process_problem $1 TCGA-SARC residual_tumor
process_problem $1 TCGA-SARC vital_status
process_problem $1 TCGA-SARC tumor_status


process_problem $1 TCGA-THCA tumor_focality
process_problem $1 TCGA-THCA ajcc_pathologic_tumor_stage

process_problem $1 TCGA-UCEC tumor_status
process_problem $1 TCGA-UCEC vital_status
process_problem $1 TCGA-UCEC retrospective_collection

process_problem $1 TCGA-LGG vital_status
process_problem $1 TCGA-LGG tumor_status
process_problem $1 TCGA-LGG histologic_diagnosis
process_problem $1 TCGA-LGG tumor_grade

process_problem $1 TCGA-COAD ajcc_pathologic_tumor_stage
process_problem $1 TCGA-COAD ajcc_tumor_pathologic_pt
process_problem $1 TCGA-COAD ajcc_nodes_pathologic_pn

process_problem $1 TCGA-KIRC ajcc_pathologic_tumor_stage
process_problem $1 TCGA-KIRC ajcc_tumor_pathologic_pt
process_problem $1 TCGA-KIRC ajcc_nodes_pathologic_pn
process_problem $1 TCGA-KIRC vital_status

process_problem $1 TCGA-LIHC ajcc_pathologic_tumor_stage
process_problem $1 TCGA-LIHC ajcc_tumor_pathologic_pt
process_problem $1 TCGA-LIHC ajcc_nodes_pathologic_pn
process_problem $1 TCGA-LIHC vital_status
process_problem $1 TCGA-LIHC tumor_grade

