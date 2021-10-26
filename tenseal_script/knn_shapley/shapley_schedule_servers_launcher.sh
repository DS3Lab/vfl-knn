#!/bin/bash

# ./shapley_schedule_servers_launcher.sh n_clients config dataset k n_test n_server cluster_name

# synthesis, N=1K, D=50, C=2
# ./shapley_schedule_servers_launcher.sh 5 /home/jiangjia/code/vfl-knn/tenseal_script/ts_ckks.config synthesis1K 5 100 1 bach
# ./shapley_schedule_servers_launcher.sh 5 /home/jiangjia/code/vfl-knn/tenseal_script/ts_ckks.config synthesis1K 3 100 1 bach

# bank, N=3200, D=8, C=2
# ./shapley_schedule_servers_launcher.sh 4 /home/jiangjia/code/vfl-knn/tenseal_script/ts_ckks.config bank 5 320 1 bach
# ./shapley_schedule_servers_launcher.sh 4 /home/jiangjia/code/vfl-knn/tenseal_script/ts_ckks.config bank 3 320 1 bach

# G2-4/G2-128, N=2048, D=4, C=2
# ./shapley_schedule_servers_launcher.sh 4 /home/jiangjia/code/vfl-knn/tenseal_script/ts_ckks.config G2-4 5 205 1 bach
# ./shapley_schedule_servers_launcher.sh 4 /home/jiangjia/code/vfl-knn/tenseal_script/ts_ckks.config G2-4 3 205 1 bach
# ./shapley_schedule_servers_launcher.sh 5 /home/jiangjia/code/vfl-knn/tenseal_script/ts_ckks.config G2-128 5 205 1 bach
# ./shapley_schedule_servers_launcher.sh 5 /home/jiangjia/code/vfl-knn/tenseal_script/ts_ckks.config G2-128 3 205 1 bach

# birch1, N=100K, D=2, C=100
# birch2, N=100K, D=2, C=100
# ./shapley_schedule_servers_launcher.sh 2 /home/jiangjia/code/vfl-knn/tenseal_script/ts_ckks.config birch1 5 100000 1 bach
# ./shapley_schedule_servers_launcher.sh 2 /home/jiangjia/code/vfl-knn/tenseal_script/ts_ckks.config birch1 3 100000 1 bach
# ./shapley_schedule_servers_launcher.sh 2 /home/jiangjia/code/vfl-knn/tenseal_script/ts_ckks.config birch2 5 100000 1 bach
# ./shapley_schedule_servers_launcher.sh 2 /home/jiangjia/code/vfl-knn/tenseal_script/ts_ckks.config birch2 3 100000 1 bach

# letter, N=20K, D=16, C=26
# ./shapley_schedule_servers_launcher.sh 4 /home/jiangjia/code/vfl-knn/tenseal_script/ts_ckks.config letter 5 2000 1 bach
# ./shapley_schedule_servers_launcher.sh 4 /home/jiangjia/code/vfl-knn/tenseal_script/ts_ckks.config letter 3 2000 1 bach

# unbalance, N=6500, D=2, C=8
# ./shapley_schedule_servers_launcher.sh 2 /home/jiangjia/code/vfl-knn/tenseal_script/ts_ckks.config unbalance 5 650 1 bach
# ./shapley_schedule_servers_launcher.sh 2 /home/jiangjia/code/vfl-knn/tenseal_script/ts_ckks.config unbalance 3 650 1 bach

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

aggr_servers=( 11 )
key_sever=12
ports=( 8991 )
#ports=( 8991 8992 8993 8994 )

server_count=0

for port in "${ports[@]}"; do
  for i in "${aggr_servers[@]}"; do
    aggr_node=${cluster_name}$i
    key_node=${cluster_name}${key_sever}
    #address="${node_name}.ethz.ch:${port}"
    aggr_address="${aggr_node}:${port}"
    key_address="${key_node}:${port}"
    dir_path="${home_path}/logs/vfl-knn/server/tenseal_shapley_schedule_${dataset}_k${k}_t${n_test}/"
    log_path="${dir_path}${i}_${port}.log"
    [ "$server_count" -eq 0 ] && echo "creating log dir" && rm -rf $dir_path && mkdir -p $dir_path && rm -rf $log_path
    echo $aggr_node "source $env_path; cd $code_path; nohup python3 -u tenseal_shapley_schedule_server.py $aggr_address $key_address $n_clients $k $config > $log_path 2>&1 &"
    ssh $aggr_node "source $env_path; cd $code_path; nohup python3 -u tenseal_shapley_schedule_server.py $aggr_address $key_address $n_clients $k $config > $log_path 2>&1 &"
    server_count=$((server_count+1))
    [ "$server_count" -ge "$n_servers" ] && echo "enough servers" && exit 1
  done
done
