#!/bin/bash

date=20170819 

# some script to process HARMONIE data
#
#
# run python script to create DALES input data

python create_input.py $date

# copy input folder to BULL

pc_path="/nobackup/users/theeuwes/DALES_runs/$date"
bull_path="theeuwes@bxshnr02:/nfs/home/users/theeuwes/work/DALES_runs/"

rsync -vau $pc_path $bull_path
