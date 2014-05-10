#!/bin/sh

#usage: submit_job root_data_dir

cd tmp


#submit_job usage: submit_job working_dir data target method
submit_job(){
qsub -cwd -V -N $3-$2-$4 -t 1:100 -j y -o /scratch/TL/pool0/ajalali/ratboost_log -l h_rt=20:: -l mem_free=$5,h_vmem=$5,h_stack=256M ../runjob.sh $1/$2/$3 $4 $6
echo $1 $2 $3 $4 $5 $6
}

run_for_data_others(){
    submit_job $1 $2 $3 'others' 5G -1
}

#run_for_data usage: run_for_data working_dir data target regularizer_index
run_for_data(){
    #submit_job $1 $2 $3 'ratboost_logistic_regression' 10G $4
    submit_job $1 $2 $3 'ratboost_linear_svc' $5 $4
    #submit_job $1 $2 $3 'ratboost_nu_svc' 10G $4
}

run_for_data_others $1 vantveer prognosis

run_for_data_others $1 TCGA-LAML vital_status
run_for_data_others $1 TCGA-LAML risk_group

run_for_data_others $1 TCGA-LAML-GeneExpression vital_status
run_for_data_others $1 TCGA-LAML-GeneExpression risk_group

run_for_data_others $1 TCGA-BRCA ER
run_for_data_others $1 TCGA-BRCA T
run_for_data_others $1 TCGA-BRCA N
run_for_data_others $1 TCGA-BRCA stage


for RI in 2 4 6 8 10 12 14 16 18
do

    run_for_data $1 vantveer prognosis $RI 5G

    run_for_data $1 TCGA-LAML vital_status $RI 5G
    run_for_data $1 TCGA-LAML risk_group $RI 5G

    run_for_data $1 TCGA-LAML-GeneExpression vital_status $RI 5G
    run_for_data $1 TCGA-LAML-GeneExpression risk_group $RI 5G

    run_for_data $1 TCGA-BRCA ER $RI 10G
    run_for_data $1 TCGA-BRCA T $RI 10G
    run_for_data $1 TCGA-BRCA N $RI 10G
    run_for_data $1 TCGA-BRCA stage $RI 10G

    #run_for_data $1 nordlund TvsB $RI
    #run_for_data $1 nordlund HeHvst1221 $RI
done

