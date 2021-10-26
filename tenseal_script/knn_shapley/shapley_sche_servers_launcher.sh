#!/bin/bash

# ./shapley_sche_servers_launcher.sh cluster_name n_servers config dataset

# unbalance
# ./shapley_sche_servers_launcher.sh bach 2 /home/jiangjia/code/vfl-knn/tenseal_script/ts_ckks.config unbalance

# server parameters
cluster_name=$1
n_servers=$2
config=$3
dataset=$4

[ -z "$cluster_name" ] && echo "var7 cluster_name is empty" && exit 1
[ -z "$config" ] && echo "var2 config is empty" && exit 1

home_path="/home/jiangjia/"
code_path="${home_path}/code/vfl-knn/transmission/tenseal/"
env_path="${home_path}/virtualenvs/pytorch-euler/bin/activate"

nodes=( 03 04 06 07 08 )
ports=( 8992 )
#ports=( 8991 8992 8993 8994 )

server_count=0

for port in "${ports[@]}"; do
  for i in "${nodes[@]}"; do
    node_name=${cluster_name}$i
    #address="${node_name}.ethz.ch:${port}"
    address="${node_name}:${port}"
    dir_path="${home_path}/logs/vfl-knn/server/tenseal_shapley_schedule_server_${dataset}/"
    log_path="${dir_path}${i}_${port}.log"
    [ "$server_count" -eq 0 ] && echo "creating log dir" && rm -rf $dir_path && mkdir -p $dir_path && rm -rf $log_path
    echo $node_name "source $env_path; cd $code_path; nohup python3 -u tenseal_key_server.py $address $config > $log_path 2>&1 &"
    ssh $node_name "source $env_path; cd $code_path; nohup python3 -u tenseal_key_server.py $address $config > $log_path 2>&1 &"
    server_count=$((server_count+1))
    [ "$server_count" -ge "$n_servers" ] && echo "enough servers" && exit 1
  done
done
