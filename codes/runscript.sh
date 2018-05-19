#!/bin/bash
# script for generating and analyzing many datasets
# to modify, change the numbers 1 and 9 to correspond to the numbers 
# of the first and last set you want to generate

for i in $(seq 1 1 9)
do
    python optstat_v3.py --Agw 5e-15 --datasetname dataset_nano_$i --nreal 10000 --computeMonopole --computeDipole
done
