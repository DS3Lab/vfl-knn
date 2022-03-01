#!/bin/bash

# This script will initialize n Servers to received encrypted data and apply addition
# python3 server.py size split_size n_clients k n_test method(e.g., allreduce) n_servers cluter_name

# higgs, N=11 million
# ./servers_launcher.sh 30 3 5 5 10 allreduce 5 bach

# synthesis, N=1K, D=50, C=2
# ./servers_launcher.sh 5 /home/jiangjia/code/vfl-knn/tenseal_script/ts_ckks.config synthesis1K 5 1000 1 bach
# ./servers_launcher.sh 5 /home/jiangjia/code/vfl-knn/tenseal_script/ts_ckks.config synthesis1K 3 1000 1 bach

# Bank, N=3200, D=8, C=2
# ./servers_launcher.sh 4 /home/jiangjia/code/vfl-knn/tenseal_script/ts_ckks.config bank 5 3200 1 bach
# ./servers_launcher.sh 4 /home/jiangjia/code/vfl-knn/tenseal_script/ts_ckks.config bank 3 3200 1 bach

# G2, N=2048, D=128/4, C=2
# ./servers_launcher.sh 5 /home/jiangjia/code/vfl-knn/tenseal_script/ts_ckks.config G2-128 5 2048 1 bach
# ./servers_launcher.sh 4 /home/jiangjia/code/vfl-knn/tenseal_script/ts_ckks.config G2-4 5 2048 1 bach

# birch1/birch1, N=100K, D=2, C=100
# ./servers_launcher.sh 2 /home/jiangjia/code/vfl-knn/tenseal_script/ts_ckks.config birch1 5 10000 1 bach
# ./servers_launcher.sh 2 /home/jiangjia/code/vfl-knn/tenseal_script/ts_ckks.config birch2 5 10000 1 bach

# letter, N=20K, D=16, C=26
# ./servers_launcher.sh 4 /home/jiangjia/code/vfl-knn/tenseal_script/ts_ckks.config letter 3 20000 1 bach
# ./servers_launcher.sh 2 /home/jiangjia/code/vfl-knn/tenseal_script/ts_ckks.config letter 3 20000 1 bach
# ./servers_launcher.sh 8 /home/jiangjia/code/vfl-knn/tenseal_script/ts_ckks.config letter 3 20000 1 bach
# ./servers_launcher.sh 16 /home/jiangjia/code/vfl-knn/tenseal_script/ts_ckks.config letter 3 20000 1 bach

# unbalance, N=6500, D=2, C=8
# ./servers_launcher.sh 2 /home/jiangjia/code/vfl-knn/tenseal_script/ts_ckks.config unbalance 5 6500 1 bach

# server parameters
n_clients=$1
config=$2

# log dir
dataset=$3
k=$4
n_test=$5

n_servers=$6
cluster_name=$7

[ -z "$n_clients" ] && echo "var1 n_clients is empty" && exit 1
[ -z "$config" ] && echo "var2 config is empty" && exit 1
[ -z "$dataset" ] && echo "var3 method is empty" && exit 1
[ -z "$k" ] && echo "var4 k is empty" && exit 1
[ -z "$n_test" ] && echo "var5 n_test is empty" && exit 1
[ -z "$n_servers" ] && echo "var6 n_servers is empty" && exit 1
[ -z "$cluster_name" ] && echo "var7 cluster_name is empty" && exit 1

home_path="/home/jiangjia/"
code_path="${home_path}/code/vfl-knn/transmission/tenseal/"
env_path="${home_path}/virtualenvs/pytorch-euler/bin/activate"

machines=( 11 )
ports=( 8991 )
#ports=( 8991 8992 8993 8994 )

server_count=0

for port in "${ports[@]}"; do
  for i in "${machines[@]}"; do
    node_name=${cluster_name}$i
    #address="${node_name}.ethz.ch:${port}"
    address="${node_name}:${port}"
    dir_path="${home_path}/logs/vfl-knn/server/tenseal_${dataset}_w${n_clients}_k${k}_t${n_test}/"
    log_path="${dir_path}${i}_${port}.log"
    [ "$server_count" -eq 0 ] && echo "creating log dir" && rm -rf $dir_path && mkdir -p $dir_path && rm -rf $log_path
    echo $node_name "source $env_path; cd $code_path; nohup python3 -u tenseal_server.py $address $n_clients $config > $log_path 2>&1 &"
    ssh $node_name "source $env_path; cd $code_path; nohup python3 -u tenseal_server.py $address $n_clients $config > $log_path 2>&1 &"
    server_count=$((server_count+1))
    [ "$server_count" -ge "$n_servers" ] && echo "enough servers" && exit 1
  done
done
