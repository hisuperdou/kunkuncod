#!/bin/bash
export LD_LIBRARY_PATH=/home/user2/anaconda3/envs/jojo/lib:$LD_LIBRARY_PATH
export PATH=/home/user2/anaconda3/envs/jojo/bin:${PATH}
which python


echo 'train model'
date_time=$(date '+%Y-%m-%d %H:%M:%S')
log="mylogs/GPN0512flow_${date_time}.log"
nohup python train_test_eval.py > "$log" 2>&1 &
