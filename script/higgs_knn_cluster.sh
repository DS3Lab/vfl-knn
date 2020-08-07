#!/bin/bash
# 10 machines
# ./higgs_knn_cluster.sh 10 28 2 10 10 1000 /bigdata/dataset/higgs-vertical/ 172.31.38.0:24000 t2.medium-10

world_size=$1
n_features=$2
n_classes=$3
k=$4
n_test=$5
n_cluster=$6
root=$7
master_ip=$8
cluster_name=$9

home_path="/bigdata/"
env_path="/home/ubuntu/envs/pytorch/bin/activate"
code_path="${home_path}/code/vfl-knn/script/"


for ((i=0; i<$world_size; i++)); do
  dir_path="${home_path}/logs/vfl-knn/higgs_cluster_w${world_size}_k${k}_t${n_test}_c${n_cluster}/"
  log_path=$dir_path$i"_"$world_size".log"
  if [[ $i == 0 ]]
	then
	  source $env_path
	  cd $code_path
	  rm -rf $dir_path
	  mkdir -p $dir_path
    rm -f $log_path
	  nohup python -u higgs_knn_cluster.py --init-method tcp://$master_ip --rank 0 --world-size $world_size \
	  --root $root --n-features $n_features --n-classes $n_classes --k $k \
	  --n-test $n_test --n-cluster $n_cluster --no-cuda > $log_path 2>&1 &
	else
	  node_name=""
	  if [[ $i -lt 10 ]]
	  then
	    node_name=${cluster_name}-node00$i
	  elif [[ $i -lt 100 ]]
	  then
	    node_name=${cluster_name}-node0$i
	  else
	    node_name=${cluster_name}-node$i
	  fi
	  # master and slaves share the same volume, do not need to rm and mkdir.
    ssh $node_name "source $env_path; cd $code_path; nohup python -u higgs_knn_cluster.py --init-method tcp://$master_ip --rank $i --world-size $world_size --root $root --n-features $n_features --n-classes $n_classes --k $k --n-test $n_test --n-cluster $n_cluster --no-cuda > $log_path 2>&1 &"
  fi
done