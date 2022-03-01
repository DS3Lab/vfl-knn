#!/bin/bash

# ./knn_MI_fagin_no_encrypt.sh world_size n_features n_classes k n_test root master_ip cluster_name

# for higgs
# ./knn_MI_fagin_no_encrypt.sh 5 15 2 5 10 /home/jiangjia/datasets/HIGGS-parts-label/ bach03:24000 bach
# ./knn_MI_fagin_no_encrypt.sh 5 15 2 10 10 /home/jiangjia/datasets/HIGGS-parts-label-small/ bach03:24000 bach

# for a1a/a6a
# ./knn_MI_fagin_no_encrypt.sh 5 125 2 5 10 /home/jiangjia/datasets/a6a-parts-label/ bach03:24000 bach
# ./knn_MI_fagin_no_encrypt.sh 5 125 2 5 10 /home/jiangjia/datasets/a1a-parts-label/ bach03:24000 bach

# for synthesis
# ./knn_MI_fagin_no_encrypt.sh 5 50 2 5 10 /home/jiangjia/datasets/n100k_d50_c2_vertical/ bach03:24000 bach synthesis
# ./knn_MI_fagin_no_encrypt.sh 5 50 2 5 10 /home/jiangjia/datasets/n1k_d50_c2_vertical/ bach03:24000 bach synthesis
# ./knn_MI_fagin_no_encrypt.sh 5 50 2 3 100 /home/jiangjia/datasets/n1k_d50_c2_vertical/ bach03:24000 bach synthesis
# ./knn_MI_fagin_no_encrypt.sh 8 50 2 3 100 /home/jiangjia/datasets/synthesis-vertical-8/ bach03:24000 bach synthesis

# for Bank
# ./knn_MI_fagin_no_encrypt.sh 4 8 2 3 320 /home/jiangjia/datasets/Bank-parts-label/ bach03:24000 bach Bank
# ./knn_MI_fagin_no_encrypt.sh 8 8 2 3 320 /home/jiangjia/datasets/Bank-vertical-8/ bach03:24000 bach Bank

# for G2
# ./knn_MI_fagin_no_encrypt.sh 4 4 2 3 205 /home/jiangjia/datasets/g2-4-100-vertical/ bach03:24000 bach g2-4
# ./knn_MI_fagin_no_encrypt.sh 5 128 2 3 205 /home/jiangjia/datasets/g2-128-100-vertical/ bach03:24000 bach g2-128
# ./knn_MI_fagin_no_encrypt.sh 8 128 2 3 205 /home/jiangjia/datasets/g2-128-100-vertical-8/ bach03:24000 bach g2-128

# birch1/birch2
# ./knn_MI_fagin_no_encrypt.sh 2 2 100 3 10000 /home/jiangjia/datasets/birch1-vertical/ bach03:24000 bach birch1
# ./knn_MI_fagin_no_encrypt.sh 2 2 100 3 10000 /home/jiangjia/datasets/birch2-vertical/ bach03:24000 bach birch2

# letter
# ./knn_MI_fagin_no_encrypt.sh 4 16 26 3 2000 /home/jiangjia/datasets/letter-vertical/ bach03:24000 bach letter
# ./knn_MI_fagin_no_encrypt.sh 8 16 26 3 2000 /home/jiangjia/datasets/letter-vertical-8/ bach03:24000 bach letter
# ./knn_MI_fagin_no_encrypt.sh 16 16 26 3 2000 /home/jiangjia/datasets/letter-vertical-16/ bach03:24000 bach letter

# unbalance
# ./knn_MI_fagin_no_encrypt.sh 2 2 8 3 650 /home/jiangjia/datasets/unbalance-vertical/ bach03:24000 bach unbalance

ETHds3900506

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

machines=( 03 04 06 07 08 )

rank=0

for i in "${machines[@]}"; do
  dir_path="${home_path}/logs/vfl-knn/MI/${dataset}_fagin_no_encrypt_w${world_size}_k${k}_t${n_test}/"
  log_path=${dir_path}${cluster_name}$i"_"$world_size".log"
  node_name=${cluster_name}$i
  if [[ $i == ${machines[0]} ]]
	then
	  rm -rf $dir_path
	  mkdir -p $dir_path
	fi
	# master and slaves share the same volume, do not need to rm and mkdir.
	echo $node_name
  ssh $node_name "source $env_path; cd $code_path; nohup python3 -u knn_MI_fagin_no_encrypt.py --init-method tcp://$master_ip --rank $rank --world-size $world_size --root $root --n-features $n_features --n-classes $n_classes --k $k --n-test $n_test --no-cuda > $log_path 2>&1 &"
  rank=$(($rank+1))
  [ "$rank" -ge "$world_size" ] && echo "enough clients" && exit 1
done
