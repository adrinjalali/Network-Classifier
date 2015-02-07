#python3.3 /TL/stat_learn/work/ajalali/Network-Classifier/run.py --working-dir $1 --method $2 --cv-index $SGE_TASK_ID --regularizer-index $3 --batch-based
python3.3 /TL/stat_learn/work/ajalali/Network-Classifier/run.py --working-dir $1 --method $2 --cv-index $SGE_TASK_ID --batch-based
