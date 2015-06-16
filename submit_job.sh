#!/bin/sh

#usage: submit_job root_data_dir

#disable core dumps
ulimit -c 0

cd /scratch/TL/pool0/ajalali/tmp


#submit_job usage: submit_job working_dir data target method
submit_job(){
qsub -cwd -V -N $3-$2-$4 -t 1:100 -j y -o /scratch/TL/pool0/ajalali/ratboost_log -l h_rt=20:: -l mem_free=$5,h_vmem=$5,h_stack=256M /TL/stat_learn/work/ajalali/Network-Classifier/runjob.sh $1/$2/$3 $4
echo $1 $2 $3 $4 $5
}

run_for_data_all(){
    submit_job $1 $2 $3 'all' 40G
    #submit_job $1 $2 $3 'rat' 40G
}

#run_for_data usage: run_for_data working_dir data target regularizer_index
run_for_data(){
    #submit_job $1 $2 $3 'ratboost_logistic_regression' 10G $4
    submit_job $1 $2 $3 'ratboost_linear_svc' $5 $4
    #submit_job $1 $2 $3 'ratboost_nu_svc' 10G $4
}

#run_for_data_others $1 vantveer prognosis

#run_for_data_others $1 TCGA-LAML vital_status
#run_for_data_others $1 TCGA-LAML risk_group

#run_for_data_others $1 TCGA-LAML-GeneExpression vital_status
#run_for_data_others $1 TCGA-LAML-GeneExpression risk_group

run_for_data_all $1 TCGA-THCA ajcc_pathologic_tumor_stage
run_for_data_all $1 TCGA-THCA tumor_focality

run_for_data_all $1 TCGA-UCEC retrospective_collection
run_for_data_all $1 TCGA-UCEC tumor_status
run_for_data_all $1 TCGA-UCEC vital_status

run_for_data_all $1 TCGA-SARC vital_status
run_for_data_all $1 TCGA-SARC tumor_status
run_for_data_all $1 TCGA-SARC residual_tumor

run_for_data_all $1 TCGA-LGG vital_status
run_for_data_all $1 TCGA-LGG tumor_status
run_for_data_all $1 TCGA-LGG histologic_diagnosis
run_for_data_all $1 TCGA-LGG tumor_grade

run_for_data_all $1 TCGA-BRCA er_status_by_ihc
run_for_data_all $1 TCGA-BRCA ajcc_pathologic_tumor_stage
run_for_data_all $1 TCGA-BRCA ajcc_tumor_pathologic_pt
run_for_data_all $1 TCGA-BRCA ajcc_nodes_pathologic_pn

#for RI in 2 4 6 8 10 12 14 16 18
#do

    ##run_for_data $1 vantveer prognosis $RI 5G

    ##run_for_data $1 TCGA-LAML vital_status $RI 5G
    ##run_for_data $1 TCGA-LAML risk_group $RI 5G

    ##run_for_data $1 TCGA-LAML-GeneExpression vital_status $RI 5G
    ##run_for_data $1 TCGA-LAML-GeneExpression risk_group $RI 5G

    ##run_for_data $1 nordlund TvsB $RI
    ##run_for_data $1 nordlund HeHvst1221 $RI

    #run_for_data $1 TCGA-UCEC retrospective_collection $RI 15G
    #run_for_data $1 TCGA-UCEC tumor_status $RI 15G
    #run_for_data $1 TCGA-UCEC vital_status $RI 15G

    #run_for_data $1 TCGA-SARC vital_status $RI 15G
    #run_for_data $1 TCGA-SARC tumor_status $RI 15G
    #run_for_data $1 TCGA-SARC residual_tumor $RI 15G

    #run_for_data $1 TCGA-BRCA er_status_by_ihc $RI 20G
    #run_for_data $1 TCGA-BRCA ajcc_pathologic_tumor_stage $RI 20G
    #run_for_data $1 TCGA-BRCA ajcc_tumor_pathologic_pt $RI 20G
    #run_for_data $1 TCGA-BRCA ajcc_nodes_pathologic_pn $RI 20G

    #run_for_data $1 TCGA-THCA ajcc_pathologic_tumor_stage $RI 20G
    #run_for_data $1 TCGA-THCA tumor_focality $RI 20G
    
    #run_for_data $1 TCGA-LGG vital_status $RI 20G
    #run_for_data $1 TCGA-LGG tumor_status $RI 20G
    #run_for_data $1 TCGA-LGG histologic_diagnosis $RI 20G
    #run_for_data $1 TCGA-LGG tumor_grade $RI 20G
#done

