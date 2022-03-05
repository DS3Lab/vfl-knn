#!/bin/bash

# ./lr_no_encrypt.sh world_size n_features n_classes n_epochs valid_ratio lr lam root master_ip cluster_name

# synthesis, N=1K, D=50, C=2
# ./lasso_no_encrypt.sh 5 50 2 100 256 0.1 0.1 0.01 /home/ubuntu/datasets/synthesis-vertical-5/ 172.31.5.19:24000 slave synthesis
# ./lasso_no_encrypt.sh 8 50 2 100 256 0.1 0.1 0.01 /home/ubuntu/datasets/synthesis-vertical-8/ 172.31.5.19:24000 slave synthesis

# Bank, N=3200, D=8, C=2
# ./lasso_no_encrypt.sh 4 8 2 100 256 0.1 0.1 0.01 /home/ubuntu/datasets/Bank-vertical-4/ 172.31.5.19:24000 slave bank
# ./lasso_no_encrypt.sh 8 8 2 100 256 0.1 0.1 0.01 /home/ubuntu/datasets/Bank-vertical-8/ 172.31.5.19:24000 slave bank

# G2, N=2048, D=4, C=2
# ./lasso_no_encrypt.sh 4 4 2 100 256 0.1 0.1 0.01 /home/ubuntu/datasets/g2-4-100-vertical-4/ 172.31.5.19:24000 slave g2-4
# ./lasso_no_encrypt.sh 5 128 2 100 256 0.1 0.1 0.01 /home/ubuntu/datasets/g2-128-100-vertical-5/ 172.31.5.19:24000 slave g2-128
# ./lasso_no_encrypt.sh 8 128 2 100 256 0.1 0.1 0.01 /home/ubuntu/datasets/g2-128-100-vertical-8/ 172.31.5.19:24000 slave g2-128


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
code_path="${home_path}/code/vfl-knn/mi_script/lasso/"

machines=( 03 04 06 07 08 )

rank=0

for i in "${machines[@]}"; do
  dir_path="${home_path}/logs/vfl-lasso/${dataset}_no_encrypt_w${world_size}_epoch${n_epochs}_b${batch_size}_lr${lr}/"
  log_path=${dir_path}${cluster_name}$i"_"$world_size".log"
  node_name=${cluster_name}$i
  if [[ $i == ${machines[0]} ]]
	then
	  rm -rf $dir_path
	  mkdir -p $dir_path
	fi
	# master and slaves share the same volume, do not need to rm and mkdir.
	echo $node_name
  ssh -i ~/jiawei_ds3_aws.pem $node_name "source $env_path; cd $code_path; nohup python3 -u lasso_no_encrypt.py --init-method tcp://$master_ip --rank $rank --world-size $world_size --root $root --n-features $n_features --n-classes $n_classes --n-epochs $n_epochs --batch-size $batch_size --valid-ratio $valid_ratio --lr $lr --lam $lam --no-cuda > $log_path 2>&1 &"
  rank=$(($rank+1))
  [ "$rank" -ge "$world_size" ] && echo "enough clients" && exit 1
done
