#!/bin/bash

# This script will initialize n Servers to received encrypted data and apply addition
# cluster_servers_launcher.sh n_clients k n_test method(e.g., allreduce) n_servers cluter_name

# for higgs
# ./cluster_servers_launcher.sh 5 5 10 allreduce 1 bach

# for a6a/a1a
# ./cluster_servers_launcher.sh 5 3 3 allreduce 1 bach
# ./cluster_servers_launcher.sh 5 5 10 allreduce 1 bach
# ./cluster_servers_launcher.sh 5 3 10 allreduce 1 bach

# for synthesis
# ./cluster_servers_launcher.sh 5 5 10 allreduce 1 bach

# for Bank
# ./cluster_servers_launcher.sh 4 5 320 allreduce 1 bach

# for G2
# ./cluster_servers_launcher.sh 4 5 205 allreduce 1 bach

# Birch1
# ./cluster_servers_launcher.sh 2 5 10000 allreduce 5 bach

# letter
# ./cluster_servers_launcher.sh 4 5 2000 allreduce 5 bach

# unbalance
# ./cluster_servers_launcher.sh 2 5 650 allreduce 5 bach


# server parameters
# should be n_worker - 1, the first worker is the coordinator
n_clients=$1
k=$2
n_test=$3
method=$4
n_servers=$5
cluster_name=$6


[ -z "$n_clients" ] && echo "var3 n_clients is empty" && exit 1
[ -z "$k" ] && echo "var4 k is empty" && exit 1
[ -z "$n_test" ] && echo "var5 n_test is empty" && exit 1
[ -z "$method" ] && echo "var6 method is empty" && exit 1
[ -z "$n_servers" ] && echo "var7 n_servers is empty" && exit 1

home_path="/home/jiangjia/"
code_path="${home_path}/code/vfl-knn/transmission/"
env_path="${home_path}/virtualenvs/pytorch-euler/bin/activate"

machines=( 11 )
ports=( 8991 )
#ports=( 8991 8992 8993 8994 )

server_count=0

for port in "${ports[@]}"; do
  for i in "${machines[@]}"; do
    node_name=${cluster_name}$i
    address="${node_name}:${port}"
    dir_path="${home_path}/logs/vfl-knn/cluster_server/${method}_k${k}_t${n_test}/"
    log_path=$dir_path$i"_"$port".log"
    [ "$server_count" -eq 0 ] && echo "creating log dir" && rm -rf $dir_path && mkdir -p $dir_path && rm -rf $log_path
    echo $node_name "source $env_path; cd $code_path; nohup python3 -u cluster_server.py $address $n_clients $n_test > $log_path 2>&1 &"
    ssh $node_name "source $env_path; cd $code_path; nohup python3 -u cluster_server.py $address $n_clients $n_test > $log_path 2>&1 &"
    server_count=$((server_count+1))
    [ "$server_count" -ge "$n_servers" ] && echo "enough servers" && exit 1
  done
done
