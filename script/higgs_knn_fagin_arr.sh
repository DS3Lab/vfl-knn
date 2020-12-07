#!/bin/bash
# 10 machines
# ./higgs_knn_fagin_arr.sh 10 28 2 10 10 /bigdata/dataset/higgs-vertical/ 172.31.38.0:24000 t2.medium-10
# ./higgs_knn_fagin_arr.sh 10 10 28 2 2 ../../data/ bach08.ethz.ch:8898 bach
#TODO clean + same in CLUSTER
world_size=$1
n_features=$2
n_classes=$3
k=$4
n_test=$5
root=$6
master_ip=$7
cluster_name=$8
rank=1

home_path="/mnt/scratch/ldiego"
env_path="/mnt/scratch/ldiego/tutorial-env/bin/activate"
code_path="${home_path}/vfl-knn/script/"

machines=( 08 02 03 04 06 07 09 10 11 12 13 )

for i in "${machines[@]}"; do
  dir_path="${home_path}/logs/vfl-knn/higgs_fagin_arr_w${world_size}_k${k}_t${n_test}/"
  log_path=$dir_path$i"_"$world_size".log"
  if [[ $i == 08 ]]
	then
	  source $env_path
	  cd $code_path
	  rm -rf $dir_path
	  mkdir -p $dir_path
    rm -f $log_path
	  nohup python3 -u higgs_knn_fagin_arr.py --init-method tcp://$master_ip --rank 0 --world-size $world_size \
	  --root $root --n-features $n_features --n-classes $n_classes --k $k \
	  --n-test $n_test --no-cuda > $log_path 2>&1 &
	else
	  node_name=${cluster_name}$i
	  # master and slaves share the same volume, do not need to rm and mkdir.
    ssh $node_name "source $env_path; cd $code_path; nohup python3 -u higgs_knn_fagin_arr.py --init-method tcp://$master_ip --rank $rank --world-size $world_size --root $root --n-features $n_features --n-classes $n_classes --k $k --n-test $n_test --no-cuda > $log_path 2>&1 &"
    rank=$(($rank+1))
  fi
done