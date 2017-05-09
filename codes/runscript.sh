#!/bin/bash
# script for generating and analyzing many datasets
# to modify, change the numbers 30 and 33 to correspond to the numbers 
# of the first and last set you want to generate

for i in $(seq 30 1 33)
do
	python optstat.py --datasetname dataset$i --computeMonopole --computeDipole --computeSkyScrambles
done
