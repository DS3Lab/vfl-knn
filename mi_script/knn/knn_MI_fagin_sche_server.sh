#!/bin/bash

# ./knn_MI_fagin_sche_server.sh world_size n_features n_classes k n_test root master_ip cluster_name

# bank, N=3200, D=8, C=2
# ./knn_MI_fagin_sche_server.sh 5 8 2 3 320 /home/jiangjia/datasets/Bank-parts-label/ bach03:24000 bach bank

world_size=$1
n_features=$2
n_classes=$3
k=$4
n_test=$5
root=$6
master_ip=$7
cluster_name=$8
dataset=$9

home_path="/home/jiangjia/"
env_path="${home_path}/virtualenvs/pytorch-euler/bin/activate"
code_path="${home_path}/code/vfl-knn/mi_script/knn/"
config="../ts_ckks.config"

machines=( 03 04 06 07 08 )

rank=0

for i in "${machines[@]}"; do
  dir_path="${home_path}/logs/vfl-knn/${dataset}_mi_fagin_schedule_w${world_size}_k${k}_t${n_test}/"
  node_name=${cluster_name}$i
  log_path="${dir_path}${node_name}_${world_size}.log"
  if [[ $i == ${machines[0]} ]]
	then
	  rm -rf $dir_path
	  mkdir -p $dir_path
	fi
	# master and slaves share the same volume, do not need to rm and mkdir.
	echo $node_name
  ssh $node_name "source $env_path; cd $code_path; nohup python3 -u knn_MI_fagin_sche_server.py --init-method tcp://$master_ip --rank $rank --world-size $world_size --root $root --config $config --n-features $n_features --n-classes $n_classes --k $k --n-test $n_test --no-cuda > $log_path 2>&1 &"
  rank=$(($rank+1))
  [ "$rank" -ge "$world_size" ] && echo "enough clients" && exit 1
done
