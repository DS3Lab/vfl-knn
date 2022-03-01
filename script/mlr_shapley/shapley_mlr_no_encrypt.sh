#!/bin/bash

# ./shapley_mlr_no_encrypt.sh world_size n_features n_classes n_epochs valid_ratio lr lam root master_ip cluster_name dataset

# birch1, N=100K, D=2, C=100
# ./shapley_mlr_no_encrypt.sh 2 2 100 1 9000 0.1 0.1 0.01 /home/jiangjia/datasets/birch1-vertical/ bach03:24000 bach birch1
# ./shapley_mlr_no_encrypt.sh 2 2 100 100 256 0.1 0.1 0.01 /home/jiangjia/datasets/birch1-vertical/ bach03:24000 bach birch1

# birch2, N=100K, D=2, C=100
# ./shapley_mlr_no_encrypt.sh 2 2 100 1 9000 0.1 0.1 0.01 /home/jiangjia/datasets/birch2-vertical/ bach03:24000 bach birch2
# ./shapley_mlr_no_encrypt.sh 2 2 100 100 256 0.1 0.1 0.01 /home/jiangjia/datasets/birch2-vertical/ bach03:24000 bach birch2

# letter, N=20K, D=16, C=26
# ./shapley_mlr_no_encrypt.sh 4 16 26 1 1800 0.1 0.1 0.01 /home/jiangjia/datasets/letter-vertical/ bach03:24000 bach letter
# ./shapley_mlr_no_encrypt.sh 4 16 26 100 256 0.1 0.1 0.01 /home/jiangjia/datasets/letter-vertical/ bach03:24000 bach letter
# ./shapley_mlr_no_encrypt.sh 8 16 26 100 256 0.1 0.1 0.01 /home/jiangjia/datasets/letter-vertical-8/ bach03:24000 bach letter

# unbalance, N=6500, D=2, C=8
# ./shapley_mlr_no_encrypt.sh 2 2 8 100 585 0.1 0.1 0.01 /home/jiangjia/datasets/unbalance-vertical/ bach03:24000 bach unbalance
# ./shapley_mlr_no_encrypt.sh 2 2 8 100 256 0.1 0.1 0.01 /home/jiangjia/datasets/unbalance-vertical/ bach03:24000 bach unbalance


world_size=$1
n_features=$2
n_classes=$3
n_epochs=$4
batch_size=$5
valid_ratio=$6
lr=$7
lam=$8
root=$9
master_ip=${10}
cluster_name=${11}
dataset=${12}

home_path="/home/jiangjia/"
env_path="${home_path}/virtualenvs/pytorch-euler/bin/activate"
code_path="${home_path}/code/vfl-knn/script/mlr_shapley/"

machines=( 03 04 06 07 08 )

rank=0

for i in "${machines[@]}"; do
  dir_path="${home_path}/logs/vfl-lr/${dataset}_shapley_no_encrypt_w${world_size}_epoch${n_epochs}_b${batch_size}_lr${lr}/"
  log_path=${dir_path}${cluster_name}$i"_"$world_size".log"
  node_name=${cluster_name}$i
  if [[ $i == ${machines[0]} ]]
	then
	  rm -rf $dir_path
	  mkdir -p $dir_path
	fi
	# master and slaves share the same volume, do not need to rm and mkdir.
	echo $node_name
  ssh $node_name "source $env_path; cd $code_path; nohup python3 -u shapley_mlr_no_encrypt.py --init-method tcp://$master_ip --rank $rank --world-size $world_size --root $root --n-features $n_features --n-classes $n_classes --n-epochs $n_epochs --batch-size $batch_size --valid-ratio $valid_ratio --lr $lr --lam $lam --no-cuda > $log_path 2>&1 &"
  rank=$(($rank+1))
  [ "$rank" -ge "$world_size" ] && echo "enough clients" && exit 1
done
