#!/bin/bash

# ./shapley_schedule_key_server_launcher.sh cluster_name config

# ./shapley_schedule_key_server_launcher.sh bach /home/jiangjia/code/vfl-knn/tenseal_script/ts_ckks.config

# server parameters
cluster_name=$1
config=$2

[ -z "$cluster_name" ] && echo "var7 cluster_name is empty" && exit 1
[ -z "$config" ] && echo "var2 config is empty" && exit 1

home_path="/home/jiangjia/"
code_path="${home_path}/code/vfl-knn/transmission/tenseal/"
env_path="${home_path}/virtualenvs/pytorch-euler/bin/activate"

nodes=( 12 )
n_servers=1
ports=( 8991 )
#ports=( 8991 8992 8993 8994 )

server_count=0

for port in "${ports[@]}"; do
  for i in "${nodes[@]}"; do
    node_name=${cluster_name}$i
    #address="${node_name}.ethz.ch:${port}"
    address="${node_name}:${port}"
    dir_path="${home_path}/logs/vfl-knn/server/tenseal_shapley_key_server/"
    log_path="${dir_path}${i}_${port}.log"
    [ "$server_count" -eq 0 ] && echo "creating log dir" && rm -rf $dir_path && mkdir -p $dir_path && rm -rf $log_path
    echo $node_name "source $env_path; cd $code_path; nohup python3 -u tenseal_key_server.py $address $config > $log_path 2>&1 &"
    ssh $node_name "source $env_path; cd $code_path; nohup python3 -u tenseal_key_server.py $address $config > $log_path 2>&1 &"
    server_count=$((server_count+1))
    [ "$server_count" -ge "$n_servers" ] && echo "enough servers" && exit 1
  done
done
