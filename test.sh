#!/bin/bash
set -e

name=${1}
echo $name
net=$(echo $name | sed 's/models\/\([^\/:]*\).*/\1/g')
dataset=$(echo $name | sed 's/.*target\_\(.*\)\_session.*/\1/g')
echo "net:"$net
echo "dataset"$dataset
for i in {1..2}
do    
    model_name=${name%epoch*}epoch_${i}_step${name#*step}
    srun -p optimal --gres=gpu:1 --quotatype=auto -J bz8 python -u umt_test.py --dataset $dataset --net $net --load_name $model_name 2>&1 | tee log/$(basename $model_name).log
done
