#!/bin/bash

# ./knn_fagin_arr_no_encrypt.sh world_size n_features n_classes k n_test n_cluster root master_ip cluster_name dataset

# higgs, N=11 million
# ./knn_fagin_arr_no_encrypt.sh 6 15 2 5 10 /home/jiangjia/datasets/HIGGS-parts-label/ bach11:24000 bach
# ./knn_fagin_arr_no_encrypt.sh 6 15 2 5 10 /home/jiangjia/datasets/HIGGS-parts-label-small/ bach11:24000 bach

# synthesis, N=1K, D=50, C=2
# ./knn_fagin_arr_no_encrypt.sh 6 50 2 5 10 /home/jiangjia/datasets/n100k_d50_c2_vertical/ bach11:24000 bach synthesis1K
# ./knn_fagin_arr_no_encrypt.sh 6 50 2 5 10 /home/jiangjia/datasets/n1k_d50_c2_vertical/ bach11:24000 bach synthesis1K

# bank, N=3200, D=8, C=2
# ./knn_fagin_arr_no_encrypt.sh 4 8 2 5 320 /home/jiangjia/datasets/Bank-parts-label/ bach03:24000 bach bank
# ./knn_fagin_arr_no_encrypt.sh 4 8 2 5 320 /home/jiangjia/datasets/Bank-parts-label/ bach03:24000 bach bank

# G2, N=2048, D=128/4, C=2
# ./knn_fagin_arr_no_encrypt.sh 5 128 2 5 205 /home/jiangjia/datasets/g2-128-10-vertical/ bach03:24000 bach G2-148
# ./knn_fagin_arr_no_encrypt.sh 4 4 2 5 205 /home/jiangjia/datasets/g2-4-100-vertical/ bach03:24000 bach G2-4

# birch1/birch2, N=100K, D=2, C=100
# ./knn_fagin_arr_no_encrypt.sh 2 2 100 5 10000 /home/jiangjia/datasets/birch1-vertical/ bach03:24000 bach birch1
# ./knn_fagin_arr_no_encrypt.sh 2 2 100 5 10000 /home/jiangjia/datasets/birch2-vertical/ bach03:24000 bach birch2

# letter, N=20K, D=16, C=26
# ./knn_fagin_arr_no_encrypt.sh 4 16 26 3 2000 /home/jiangjia/datasets/letter-vertical/ bach03:24000 bach letter
# ./knn_fagin_arr_no_encrypt.sh 2 16 26 3 2000 /home/jiangjia/datasets/letter-vertical-2/ bach03:24000 bach letter
# ./knn_fagin_arr_no_encrypt.sh 8 16 26 3 2000 /home/jiangjia/datasets/letter-vertical-8/ bach03:24000 bach letter
# ./knn_fagin_arr_no_encrypt.sh 16 16 26 3 2000 /home/jiangjia/datasets/letter-vertical-16/ bach02:24000 bach letter

# unbalance, N=6500, D=2, C=8
# ./knn_fagin_arr_no_encrypt.sh 2 2 8 5 650 /home/jiangjia/datasets/unbalance-vertical/ bach03:24000 bach unbalance
# ./knn_fagin_arr_no_encrypt.sh 2 2 8 3 650 /home/jiangjia/datasets/unbalance-vertical/ bach03:24000 bach unbalance


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
code_path="${home_path}/code/vfl-knn/script/knn/"

machines=( 03 04 06 07 08 )

rank=0

for i in "${machines[@]}"; do
  dir_path="${home_path}/logs/vfl-knn/${dataset}_fagin_arr_no_encrypt_w${world_size}_k${k}_t${n_test}/"
  node_name=${cluster_name}$i
  log_path="${dir_path}${node_name}_${world_size}.log"
  if [[ $i == ${machines[0]} ]]
	then
	  rm -rf $dir_path
	  mkdir -p $dir_path
	fi
	# master and slaves share the same volume, do not need to rm and mkdir.
	echo $node_name
  ssh $node_name "source $env_path; cd $code_path; nohup python3 -u knn_fagin_arr_no_encrypt.py --init-method tcp://$master_ip --rank $rank --world-size $world_size --root $root --n-features $n_features --n-classes $n_classes --k $k --n-test $n_test --no-cuda > $log_path 2>&1 &"
  rank=$(($rank+1))
  [ "$rank" -ge "$world_size" ] && echo "enough clients" && exit 1
done