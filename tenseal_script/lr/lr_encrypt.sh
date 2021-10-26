#!/bin/bash

# ./lr_encrypt.sh world_size n_features n_classes n_epochs valid_ratio lr lam root config master_ip cluster_name

# Bank, N=3200, D=8, C=2
# ./lr_encrypt.sh 4 8 2 100 287 0.1 0.1 0.01 /home/jiangjia/datasets/Bank-parts-label/ ts_ckks.config bach03:24000 bach bank
# ./lr_encrypt.sh 4 8 2 100 256 0.1 0.1 0.01 /home/jiangjia/datasets/Bank-parts-label/ ts_ckks.config bach03:24000 bach bank

# synthesis, N=1K, D=50, C=2
# ./lr_encrypt.sh 5 50 2 100 256 0.1 0.1 0.01 /home/jiangjia/datasets/n1k_d50_c2_vertical/ ts_ckks.config bach03:24000 bach synthesis1K

# G2, N=2048, D=4, C=2
# ./lr_encrypt.sh 5 128 2 100 256 0.1 0.1 0.01 /home/jiangjia/datasets/g2-128-100-vertical/ ts_ckks.config bach03:24000 bach G2-128
# ./lr_encrypt.sh 4 4 2 100 256 0.1 0.1 0.01 /home/jiangjia/datasets/g2-4-100-vertical/ ts_ckks.config bach03:24000 bach G2-4


world_size=$1
n_features=$2
n_classes=$3
n_epochs=$4
batch_size=$5
valid_ratio=$6
lr=$7
lam=$8
root=$9
config=${10}
master_ip=${11}
cluster_name=${12}
dataset=${13}

home_path="/home/jiangjia/"
env_path="${home_path}/virtualenvs/pytorch-euler/bin/activate"
code_path="${home_path}/code/vfl-knn/tenseal_script/lr/"

machines=( 03 04 06 07 08 )

rank=0

for i in "${machines[@]}"; do
  dir_path="${home_path}/logs/vfl-lr/tenseal_${dataset}_encrypt_adam_w${world_size}_epoch${n_epochs}_b${batch_size}_lr${lr}/"
  node_name=${cluster_name}$i
  log_path="${dir_path}${node_name}_${world_size}.log"
  if [[ $i == ${machines[0]} ]]
	then
	  rm -rf $dir_path
	  mkdir -p $dir_path
	fi
	# master and slaves share the same volume, do not need to rm and mkdir.
	echo $node_name
  ssh $node_name "source $env_path; cd $code_path; nohup python3 -u lr_encrypt.py --init-method tcp://$master_ip --rank $rank --world-size $world_size --root $root --config ${config} --n-features $n_features --n-classes $n_classes --n-epochs $n_epochs --batch-size $batch_size --valid-ratio $valid_ratio --lr $lr --lam $lam --no-cuda > $log_path 2>&1 &"
  rank=$(($rank+1))
  [ "$rank" -ge "$world_size" ] && echo "enough clients" && exit 1
done
