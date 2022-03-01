#!/bin/bash

# ./allreduce_test.sh world_size n_test tensor_size master_ip cluster_name

# ./allreduce_test.sh 2 100 1000 bach03:24000 bach

world_size=$1
n_test=$2
tensor_size=$3
master_ip=$4
cluster_name=$5

home_path="/home/jiangjia/"
env_path="${home_path}/virtualenvs/pytorch-euler/bin/activate"
code_path="${home_path}/code/vfl-knn/crypten_script/"

machines=( 03 04 06 07 08 )

rank=0

for i in "${machines[@]}"; do
  dir_path="${home_path}/logs/CrypTen/allreduce_w${world_size}_t${n_test}_size${tensor_size}/"
  log_path=${dir_path}${cluster_name}$i"_"$world_size".log"
  node_name=${cluster_name}$i
  if [[ $i == ${machines[0]} ]]
	then
	  rm -rf $dir_path
	  mkdir -p $dir_path
	fi
	# master and slaves share the same volume, do not need to rm and mkdir.
	echo $node_name
  ssh $node_name "source $env_path; cd $code_path; nohup python3 -u allreduce_test.py --init-method tcp://$master_ip --rank $rank --world-size $world_size --n-test $n_test --tensor-size $tensor_size --no-cuda > $log_path 2>&1 &"
  rank=$(($rank+1))
  [ "$rank" -ge "$world_size" ] && echo "enough clients" && exit 1
done
