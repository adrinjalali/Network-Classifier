#to check which files don't exist:

#i variable goes through cross validation indices (0 .. len(cvs - 1)), 
#j is the regularizer.

root_dir='/scratch/TL/pool0/ajalali/ratboost/data_29_sep_2014/'
working_dir='TCGA-UCEC/tumor_status/'
for i in {0..21}; 
do 
   for j in {2..18..2}; 
   do 
      if ! ls $root_dir/$working_dir/results/ratboost_linear_svc-$i-rat-$j* &>/dev/null; 
      then 
	   let "cv = $i+1"
	   echo $cv $j
	   python3.3 /TL/stat_learn/work/ajalali/Network-Classifier/run.py --working-dir $root_dir/$working_dir --method ratboost_linear_svc --cv-index $cv --regularizer-index $j --batch-based --cpu-count 40
      fi;
   done; 
done;



