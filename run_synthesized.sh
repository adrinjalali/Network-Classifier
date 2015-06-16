#!/bin/bash

for bnet_count in 5 10 15 20
do
    for noise in 0.1 0.3 0.5 0.7 0.9
    do
	#ipython3 run_synthesized.py -- --bnet_count $bnet_count --feature_noise $noise &
	python3.3 run_synthesized.py --bnet_count $bnet_count --feature_noise $noise --working_dir $1 &
    done
done
