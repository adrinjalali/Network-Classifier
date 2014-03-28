#!/bin/sh

#usage: submit_job root_data_dir

cd tmp


#submit_job usage: submit_job working_dir data target method
submit_job(){
qsub -cwd -V -N $2-$3-$4 -t 1:100 -j y -o /scratch/TL/pool0/ajalali/ratboost_log -l h_rt=20:: -l mem_free=11G,h_vmem=11G,h_stack=256M ../runjob.sh $1/$2/$3  $4
echo $1 $2 $3 $4
}

#run_for_data usage: run_for_data working_dir data target
run_for_data(){
    #submit_job $1 $2 $3 'others'
    submit_job $1 $2 $3 'ratboost_logistic_regression'
    #submit_job $1 $2 $3 'ratboost_linear_svc'
    #submit_job $1 $2 $3 'ratboost_nu_svc'
}

#run_for_data $1 vantveer prognosis

#run_for_data $1 TCGA-LAML vital_status
#run_for_data $1 TCGA-LAML risk_group

#run_for_data $1 TCGA-BRCA ER
#run_for_data $1 TCGA-BRCA T
#run_for_data $1 TCGA-BRCA N
#run_for_data $1 TCGA-BRCA stage

run_for_data $1 nordlund TvsB
#run_for_data $1 nordlund HeHvst1221

