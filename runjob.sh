#!/bin/bash
#python3.3 /TL/stat_learn/work/ajalali/Network-Classifier/run.py --working-dir $1 --method $2 --cv-index $SGE_TASK_ID --regularizer-index $3 --batch-based
env
echo "date-time-begin"
date

python3.5 run.py --input-dir $1 --working-dir $2 --method $3 --cv-index $SGE_TASK_ID --batch-based
#ipython3 -i run.py -- --input-dir $1 --working-dir $2 --method $3 --cv-index $SGE_TASK_ID --batch-based --cpu-count 50

echo "date-time-finish"
date
