#!/bin/bash
# 10 machines
# ./knn_fagin_dict_old.sh 5 15 2 10 10 /home/jiangjia/datasets/HIGGS-parts-label/ bach11:25000

workers="bach07 bach08 bach09 bach10 bach11 bach12 bach13 bach14 bach15 bach16"

world_size=$1
n_features=$2
n_classes=$3
k=$4
n_test=$5
root=$6
master_ip=$7

home_path="/home/jiangjia/"
env_path="${home_path}/virtualenvs/pytorch-bach/bin/activate"
code_path="${home_path}/code/vfl-knn/script/"

for i in $workers; do
  dir_path="${home_path}/logs/vfl-knn/higgs_fagin_dict_old_w${world_size}_k${k}_t${n_test}/"
  log_path=$dir_path$i"_"$world_size".log"
  if [[ $i == "bach07" ]]
	then
	  source $env_path
	  cd $code_path
	  rm -rf $dir_path
	  mkdir -p $dir_path
    rm -f $log_path
	  nohup python -u knn_fagin_dict_old.py --init-method tcp://$master_ip --rank 0 --world-size $world_size \
	  --root $root --n-features $n_features --n-classes $n_classes --k $k \
	  --n-test $n_test --no-cuda > $log_path 2>&1 &
	else
	  node_name=$i
	  # master and slaves share the same volume, do not need to rm and mkdir.
    ssh $node_name "source $env_path; cd $code_path; nohup python -u higgs_knn_fagin_dict_old.py --init-method tcp://$master_ip --rank $i --world-size $world_size --root $root --n-features $n_features --n-classes $n_classes --k $k --n-test $n_test --no-cuda > $log_path 2>&1 &"
  fi
done