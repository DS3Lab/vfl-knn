#!/bin/bash

# ./shapley_knn_fagin_key_server.sh world_size n_features n_classes k n_test root config master_ip cluster_name

# higgs, N=11 million
# ./shapley_knn_fagin_key_server.sh 5 15 2 5 10000 /home/jiangjia/datasets/HIGGS-parts-label/ ts_ckks.config bach03:24000 bach higgs
# ./shapley_knn_fagin_key_server.sh 5 15 2 10 10000 /home/jiangjia/datasets/HIGGS-parts-label-small/ ts_ckks.config bach03:24000 bach higgs

# synthesis, N=1K, D=50, C=2
# ./shapley_knn_fagin_key_server.sh 5 50 2 5 100 /home/jiangjia/datasets/n100k_d50_c2_vertical/ ts_ckks.config bach03:24000 bach synthesis1K
# ./shapley_knn_fagin_key_server.sh 5 50 2 5 100 /home/jiangjia/datasets/n1k_d50_c2_vertical/ ts_ckks.config bach03:24000 bach synthesis1K
# ./shapley_knn_fagin_key_server.sh 5 50 2 3 100 /home/jiangjia/datasets/n1k_d50_c2_vertical/ ts_ckks.config bach03:24000 bach synthesis1K

# bank, N=3200, D=8, C=2
# ./shapley_knn_fagin_key_server.sh 4 8 2 5 320 /home/jiangjia/datasets/Bank-parts-label/ ts_ckks.config bach03:24000 bach bank
# ./shapley_knn_fagin_key_server.sh 4 8 2 3 320 /home/jiangjia/datasets/Bank-parts-label/ ts_ckks.config bach03:24000 bach bank

# G2-4/G2-128, N=2048, D=4, C=2
# ./shapley_knn_fagin_key_server.sh 4 4 2 5 205 /home/jiangjia/datasets/g2-4-100-vertical/ ts_ckks.config bach03:24000 bach G2-4
# ./shapley_knn_fagin_key_server.sh 4 4 2 3 205 /home/jiangjia/datasets/g2-4-100-vertical/ ts_ckks.config bach03:24000 bach G2-4
# ./shapley_knn_fagin_key_server.sh 5 128 2 5 205 /home/jiangjia/datasets/g2-128-100-vertical/ ts_ckks.config bach03:24000 bach G2-128
# ./shapley_knn_fagin_key_server.sh 5 128 2 3 205 /home/jiangjia/datasets/g2-128-100-vertical/ ts_ckks.config bach03:24000 bach G2-128

# birch1, N=100K, D=2, C=100
# birch2, N=100K, D=2, C=100
# ./shapley_knn_fagin_key_server.sh 2 2 100 5 10000 /home/jiangjia/datasets/birch1-vertical/ ts_ckks.config bach03:24000 bach birch1
# ./shapley_knn_fagin_key_server.sh 2 2 100 3 10000 /home/jiangjia/datasets/birch1-vertical/ ts_ckks.config bach03:24000 bach birch1
# ./shapley_knn_fagin_key_server.sh 2 2 100 5 10000 /home/jiangjia/datasets/birch2-vertical/ ts_ckks.config bach03:24000 bach birch2

# letter, N=20K, D=16, C=26
# ./shapley_knn_fagin_key_server.sh 4 16 26 5 2000 /home/jiangjia/datasets/letter-vertical/ ts_ckks.config bach03:24000 bach letter
# ./shapley_knn_fagin_key_server.sh 4 16 26 3 2000 /home/jiangjia/datasets/letter-vertical/ ts_ckks.config bach03:24000 bach letter

# unbalance, N=6500, D=2, C=8
# ./shapley_knn_fagin_key_server.sh 2 2 8 5 650 /home/jiangjia/datasets/unbalance-vertical/ ts_ckks.config bach03:24000 bach unbalance
# ./shapley_knn_fagin_key_server.sh 2 2 8 3 650 /home/jiangjia/datasets/unbalance-vertical/ ts_ckks.config bach03:24000 bach unbalance


world_size=$1
n_features=$2
n_classes=$3
k=$4
n_test=$5
root=$6
config=$7
master_ip=$8
cluster_name=$9
dataset=${10}

home_path="/home/jiangjia/"
env_path="${home_path}/virtualenvs/pytorch-euler/bin/activate"
code_path="${home_path}/code/vfl-knn/tenseal_script/knn_shapley/"

machines=( 03 04 06 07 08 )

rank=0

for i in "${machines[@]}"; do
  dir_path="${home_path}/logs/vfl-knn/${dataset}_shapley_fagin_key_tenseal_w${world_size}_k${k}_t${n_test}/"
  node_name=${cluster_name}$i
  log_path="${dir_path}${node_name}_${world_size}.log"
  if [[ $i == ${machines[0]} ]]
	then
	  rm -rf $dir_path
	  mkdir -p $dir_path
	fi
	# master and slaves share the same volume, do not need to rm and mkdir.
	echo $node_name
  ssh $node_name "source $env_path; cd $code_path; nohup python3 -u shapley_knn_fagin_key_server.py --init-method tcp://$master_ip --rank $rank --world-size $world_size --root $root --config $config --n-features $n_features --n-classes $n_classes --k $k --n-test $n_test --no-cuda > $log_path 2>&1 &"
  rank=$(($rank+1))
  [ "$rank" -ge "$world_size" ] && echo "enough clients" && exit 1
done
