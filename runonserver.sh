#!/bin/bash

ulimit -c 0

LOG_DIR="/scratch/TL/pool0/ajalali/ratboost_log"
INPUT_ROOT="/scratch/TL/pool0/ajalali/ratboost/shared/Data"

N_JOBS=1
PARALLEL_JOBS=9

export active=0

check_wait(){
    ((active++))
    export active=$active
    #active=$[active + 1]
    echo "check wait function"
    echo $active
    if [ $active -eq $PARALLEL_JOBS ]
    then
	echo "waiting"
	wait
	active=0
    fi
}

submit_job(){
    #env
    #date
    echo $1 $2 $3 $4 $5
    logfile=$LOG_DIR/$2-$3-$4-$5.log
    python3.3 run.py --input-dir $INPUT_ROOT/$2/$3 --working-dir $1/$2/$3 --method $4 --cv-index $5 --batch-based --cpu-count $N_JOBS &>$logfile
}

run_for_data_all(){
    #submit_job $1 $2 $3 'all' 40G
    #submit_job $1 $2 $3 'others' $4 &
    #submit_job $1 $2 $3 'rat' $4 &
    submit_job $1 $2 $3 'raccoon' $4 &
    #heck_wait
}


for i in `seq 1 50`; do
    echo "============================================"
    echo $i
    echo "============================================"
    run_for_data_all $1 TCGA-THCA ajcc_pathologic_tumor_stage $i
    run_for_data_all $1 TCGA-THCA tumor_focality $i

    run_for_data_all $1 TCGA-UCEC retrospective_collection $i
    run_for_data_all $1 TCGA-UCEC tumor_status $i
    run_for_data_all $1 TCGA-UCEC vital_status $i

    run_for_data_all $1 TCGA-SARC vital_status $i
    run_for_data_all $1 TCGA-SARC tumor_status $i
    run_for_data_all $1 TCGA-SARC residual_tumor $i

    run_for_data_all $1 TCGA-LGG vital_status $i
    run_for_data_all $1 TCGA-LGG tumor_status $i
    run_for_data_all $1 TCGA-LGG histologic_diagnosis $i
    run_for_data_all $1 TCGA-LGG tumor_grade $i

    run_for_data_all $1 TCGA-BRCA er_status_by_ihc $i
    run_for_data_all $1 TCGA-BRCA ajcc_pathologic_tumor_stage $i
    run_for_data_all $1 TCGA-BRCA ajcc_tumor_pathologic_pt $i
    run_for_data_all $1 TCGA-BRCA ajcc_nodes_pathologic_pn $i

    run_for_data_all $1 TCGA-COAD ajcc_pathologic_tumor_stage $i
    run_for_data_all $1 TCGA-COAD ajcc_tumor_pathologic_pt $i
    run_for_data_all $1 TCGA-COAD ajcc_nodes_pathologic_pn $i

    run_for_data_all $1 TCGA-KIRC ajcc_pathologic_tumor_stage $i
    run_for_data_all $1 TCGA-KIRC ajcc_tumor_pathologic_pt $i
    run_for_data_all $1 TCGA-KIRC ajcc_nodes_pathologic_pn $i
    run_for_data_all $1 TCGA-KIRC vital_status $i

    run_for_data_all $1 TCGA-LIHC ajcc_pathologic_tumor_stage $i
    run_for_data_all $1 TCGA-LIHC ajcc_tumor_pathologic_pt $i
    run_for_data_all $1 TCGA-LIHC ajcc_nodes_pathologic_pn $i
    run_for_data_all $1 TCGA-LIHC vital_status $i
    run_for_data_all $1 TCGA-LIHC tumor_grade $i
    wait
done
