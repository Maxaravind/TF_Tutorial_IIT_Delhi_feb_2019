#!/bin/sh
#PBS -N run1
#PBS -P <your dept code>
#PBS -e err
#PBS -o out
#PBS -l select=1:ngpus=1
#PBS -l walltime=40:00:00


cd /<path to your directory containing the files filename.py >/
source activate <your environment>
module load apps/pythonpackages/3.6.0/tensorflow/1.9.0/gpu
unbuffer python filename.py | tee ./log.txt
source deactivate
