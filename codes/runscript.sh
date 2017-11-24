#!/bin/bash
# script for generating and analyzing many datasets
# to modify, change the numbers 30 and 33 to correspond to the numbers 
# of the first and last set you want to generate

for i in $(seq 0 1 9)
do
	python optstat_v2.py --datasetname dataset_nano_$i --computeMonopole --computeDipole --computeSkyScrambles
done
