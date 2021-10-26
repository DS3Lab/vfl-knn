#!/bin/bash

# ./shapley_knn_cluster_no_encrypt.sh world_size n_features n_classes k n_test root master_ip cluster_name

# for higgs
# ./shapley_knn_cluster_no_encrypt.sh 5 15 2 5 10 10 /home/jiangjia/datasets/HIGGS-parts-label/ bach03:24000 bach
# ./shapley_knn_cluster_no_encrypt.sh 5 15 2 10 10 10 /home/jiangjia/datasets/HIGGS-parts-label-small/ bach03:24000 bach

# for a1a/a6a
# ./shapley_knn_cluster_no_encrypt.sh 5 125 2 5 10 10 /home/jiangjia/datasets/a6a-parts-label/ bach03:24000 bach
# ./shapley_knn_cluster_no_encrypt.sh 5 125 2 5 10 10 /home/jiangjia/datasets/a1a-parts-label/ bach03:24000 bach

# for synthesis
# ./shapley_knn_cluster_no_encrypt.sh 5 50 2 5 10 10 /home/jiangjia/datasets/n100k_d50_c2_vertical/ bach03:24000 bach
# ./shapley_knn_cluster_no_encrypt.sh 5 50 2 5 100 400 /home/jiangjia/datasets/n1k_d50_c2_vertical/ bach03:24000 bach

# for Bank
# ./shapley_knn_cluster_no_encrypt.sh 4 50 2 5 320 100 /home/jiangjia/datasets/Bank-parts-label/ bach03:24000 bach

# for G2
# ./shapley_knn_cluster_no_encrypt.sh 4 4 2 5 205 100 /home/jiangjia/datasets/g2-4-100-vertical/ bach03:24000 bach

# Birch1
# ./shapley_knn_cluster_no_encrypt.sh 2 2 100 5 10000 100 /home/jiangjia/datasets/birch1-vertical/ bach03:24000 bach

# letter
# ./shapley_knn_cluster_no_encrypt.sh 4 16 26 5 2000 100 /home/jiangjia/datasets/letter-vertical/ bach03:24000 bach

# unbalance
# ./shapley_knn_cluster_no_encrypt.sh 2 2 8 5 650 100 /home/jiangjia/datasets/unbalance-vertical/ bach03:24000 bach


world_size=$1
n_features=$2
n_classes=$3
k=$4
n_test=$5
n_cluster=$6
root=$7
master_ip=$8
cluster_name=$9

home_path="/home/jiangjia/"
env_path="${home_path}/virtualenvs/pytorch-euler/bin/activate"
code_path="${home_path}/code/vfl-knn/script/"

machines=( 03 04 06 07 08 )

rank=0

for i in "${machines[@]}"; do
  dir_path="${home_path}/logs/vfl-knn/shapley_cluster_no_encrypt_w${world_size}_k${k}_t${n_test}/"
  log_path=${dir_path}${cluster_name}$i"_"$world_size".log"
  node_name=${cluster_name}$i
  if [[ $i == ${machines[0]} ]]
	then
	  rm -rf $dir_path
	  mkdir -p $dir_path
	fi
	# master and slaves share the same volume, do not need to rm and mkdir.
	echo $node_name
  ssh $node_name "source $env_path; cd $code_path; nohup python3 -u shapley_knn_cluster_no_encrypt.py --init-method tcp://$master_ip --rank $rank --world-size $world_size --root $root --n-features $n_features --n-classes $n_classes --k $k --n-test $n_test --n-cluster $n_cluster --no-cuda > $log_path 2>&1 &"
  rank=$(($rank+1))
done
