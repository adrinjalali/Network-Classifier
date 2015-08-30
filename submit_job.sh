#!/bin/sh

#usage: submit_job output_root_dir

#disable core dumps
ulimit -c 0

#cd /scratch/TL/pool0/ajalali/tmp

INPUT_ROOT="/scratch/TL/pool0/ajalali/ratboost/shared/Data"
OUTPUT_DIR="`pwd`/../../Data"

#submit_job usage: submit_job working_dir data target method
submit_job(){
qsub -cwd -V -N $3-$2-$4 -t 1:50 -j y -o /scratch/TL/pool0/ajalali/ratboost_log -l h_rt=$6 -l mem_free=$5,h_vmem=$5,h_stack=256M runjob.sh $INPUT_ROOT/$2/$3 $1/$2/$3 $4
echo $1 $2 $3 $4 $5
}

run_for_data_all(){
    if [ "$1" == "others" ]; then
        TIME="12::"
        MEM=$4
    elif [ "$1" == "ratboost" ]; then
        TIME="120::"
        MEM=$5
    elif [ "$1" == "raccoon" ]; then
        TIME="48::"
        MEM=$6
    fi
    #submit_job $OUTPUT_DIR $1 $2 'others' $3 "12::"
    #submit_job $OUTPUT_DIR $1 $2 'rat' $4 "48::"
    #submit_job $OUTPUT_DIR $1 $2 'ratboost' $4 "120::"
    #submit_job $OUTPUT_DIR $2 $3 'raccoon' $6 "48::"
    submit_job $OUTPUT_DIR $2 $3 $1 $MEM $TIME
}

run_for_data_all $1 TCGA-THCA ajcc_pathologic_tumor_stage 10G 40G 20G
run_for_data_all $1 TCGA-THCA tumor_focality 10G 40G 20G

run_for_data_all $1 TCGA-UCEC retrospective_collection 10G 40G 20G
run_for_data_all $1 TCGA-UCEC tumor_status 10G 40G 20G
run_for_data_all $1 TCGA-UCEC vital_status 10G 40G 20G

run_for_data_all $1 TCGA-SARC vital_status 10G 40G 20G
run_for_data_all $1 TCGA-SARC tumor_status 10G 40G 20G
run_for_data_all $1 TCGA-SARC residual_tumor 10G 40G 20G

run_for_data_all $1 TCGA-LGG vital_status 10G 40G 20G
run_for_data_all $1 TCGA-LGG tumor_status 10G 40G 20G
run_for_data_all $1 TCGA-LGG histologic_diagnosis 10G 40G 20G
run_for_data_all $1 TCGA-LGG tumor_grade 10G 40G 20G

run_for_data_all $1 TCGA-BRCA er_status_by_ihc 10G 40G 40G
run_for_data_all $1 TCGA-BRCA ajcc_pathologic_tumor_stage 10G 40G 40G
run_for_data_all $1 TCGA-BRCA ajcc_tumor_pathologic_pt 10G 40G 40G
run_for_data_all $1 TCGA-BRCA ajcc_nodes_pathologic_pn 10G 40G 40G

run_for_data_all $1 TCGA-COAD ajcc_pathologic_tumor_stage 10G 40G 20G
run_for_data_all $1 TCGA-COAD ajcc_tumor_pathologic_pt 10G 40G 20G
run_for_data_all $1 TCGA-COAD ajcc_nodes_pathologic_pn 10G 40G 20G

run_for_data_all $1 TCGA-KIRC ajcc_pathologic_tumor_stage 10G 40G 20G
run_for_data_all $1 TCGA-KIRC ajcc_tumor_pathologic_pt 10G 40G 20G
run_for_data_all $1 TCGA-KIRC ajcc_nodes_pathologic_pn 10G 40G 20G
run_for_data_all $1 TCGA-KIRC vital_status 10G 40G 20G

run_for_data_all $1 TCGA-LIHC ajcc_pathologic_tumor_stage 10G 40G 20G
run_for_data_all $1 TCGA-LIHC ajcc_tumor_pathologic_pt 10G 40G 20G
run_for_data_all $1 TCGA-LIHC ajcc_nodes_pathologic_pn 10G 40G 20G
run_for_data_all $1 TCGA-LIHC vital_status 10G 40G 20G
run_for_data_all $1 TCGA-LIHC tumor_grade 10G 40G 20G

