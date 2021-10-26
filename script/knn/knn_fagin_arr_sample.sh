#!/bin/bash

# ./knn_fagin_arr_sample.sh world_size n_features n_classes k n_test n_cluster root master_ip cluster_name

# for higgs
# ./knn_fagin_arr_sample.sh 6 15 2 5 10 /home/jiangjia/datasets/HIGGS-parts-label/ bach11:24000 bach
# ./knn_fagin_arr_sample.sh 6 15 2 5 10 /home/jiangjia/datasets/HIGGS-parts-label-small/ bach11:24000 bach

# for a1a/a6a
# ./knn_fagin_arr_sample.sh 6 125 2 5 10 /home/jiangjia/datasets/a6a-parts-label/ bach11:24000 bach
# ./knn_fagin_arr_sample.sh 6 125 2 5 10 /home/jiangjia/datasets/a1a-parts-label/ bach11:24000 bach

# for synthesis
# ./knn_fagin_arr_sample.sh 6 50 2 5 10 /home/jiangjia/datasets/n100k_d50_c2_vertical/ bach11:24000 bach
# ./knn_fagin_arr_sample.sh 6 50 2 5 10 /home/jiangjia/datasets/n1k_d50_c2_vertical/ bach11:24000 bach

# for Bank
# ./knn_fagin_arr_sample.sh 5 8 2 5 320 /home/jiangjia/datasets/Bank-parts-label/ bach11:24000 bach
# ./knn_fagin_arr_sample.sh 5 8 2 5 320 /home/jiangjia/datasets/Bank-parts-label/ bach11:24000 bach

# for G2
# ./knn_fagin_arr_sample.sh 6 128 2 5 205 /home/jiangjia/datasets/g2-128-10-vertical/ bach11:24000 bach
# ./knn_fagin_arr_sample.sh 5 4 2 5 205 /home/jiangjia/datasets/g2-4-100-vertical/ bach11:24000 bach

# Birch1
# ./knn_fagin_arr_sample.sh 3 2 100 5 10000 /home/jiangjia/datasets/birch1-vertical/ bach11:24000 bach

# letter
# ./knn_fagin_arr_sample.sh 5 16 26 5 2000 /home/jiangjia/datasets/letter-vertical/ bach11:24000 bach

# unbalance
# ./knn_fagin_arr_sample.sh 3 2 8 5 650 /home/jiangjia/datasets/unbalance-vertical/ bach11:24000 bach


world_size=$1
n_features=$2
n_classes=$3
k=$4
n_test=$5
root=$6
master_ip=$7
cluster_name=$8

home_path="/home/jiangjia/"
env_path="${home_path}/virtualenvs/pytorch-euler/bin/activate"
code_path="${home_path}/code/vfl-knn/script/"

machines=( 11 03 04 06 07 08 )

rank=0

for i in "${machines[@]}"; do
  dir_path="${home_path}/logs/vfl-knn/fagin_arr_sample_w${world_size}_k${k}_t${n_test}/"
  log_path=${dir_path}${cluster_name}$i"_"$world_size".log"
  node_name=${cluster_name}$i
  if [[ $i == ${machines[0]} ]]
	then
	  rm -rf $dir_path
	  mkdir -p $dir_path
	fi
	# master and slaves share the same volume, do not need to rm and mkdir.
	echo $node_name
  ssh $node_name "source $env_path; cd $code_path; nohup python3 -u knn_fagin_arr_sample.py --init-method tcp://$master_ip --rank $rank --world-size $world_size --root $root --n-features $n_features --n-classes $n_classes --k $k --n-test $n_test --no-cuda > $log_path 2>&1 &"
  rank=$(($rank+1))
done