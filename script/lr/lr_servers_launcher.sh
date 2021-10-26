#!/bin/bash

# This script will initialize n Servers to received encrypted data and apply addition
# python3 server.py size split_size n_clients k n_test method(e.g., allreduce) n_servers cluter_name

# for higgs
# ./servers_launcher.sh 30 3 5 5 10 allreduce 5 bach

# for a6a/a1a
# ./servers_launcher.sh 125 3 5 3 3 allreduce 5 bach

# for synthesis
# ./lr_servers_launcher.sh 500000 5 5 1 bach synthesis

# for Bank
# ./lr_servers_launcher.sh 400000 4 4 1 bach bank

# for G2
# ./lr_servers_launcher.sh 200000 2 4 1 bach G2

# birch1/birch2
# ./lr_servers_launcher.sh 10000 1 2 1 bach birch1
# ./lr_servers_launcher.sh 10000 1 2 1 bach birch2

# letter
# ./lr_servers_launcher.sh 40000 4 4 1 bach unbalance

# unbalance
# ./lr_servers_launcher.sh 10000 1 2 1 bach unbalance

# server parameters
size=$1
split_size=$2
# should be n_worker - 1, the first worker is the coordinator
n_clients=$3
n_servers=$4
cluster_name=$5
dataset=$6

[ -z "$size" ] && echo "var1 size is empty" && exit 1
[ -z "$split_size" ] && echo "var2 split_size is empty" && exit 1
[ -z "$n_clients" ] && echo "var3 n_clients is empty" && exit 1
[ -z "$n_servers" ] && echo "var7 n_servers is empty" && exit 1

home_path="/home/jiangjia/"
code_path="${home_path}/code/vfl-knn/transmission/pallier"
env_path="${home_path}/virtualenvs/pytorch-euler/bin/activate"

machines=( 11 12 13 14 15 )
ports=( 8991 )
#ports=( 8991 8992 8993 8994 )

server_count=0

for port in "${ports[@]}"; do
  for i in "${machines[@]}"; do
    # Connection to machines
    node_name=${cluster_name}$i
    #address="${node_name}.ethz.ch:${port}"
    address="${node_name}:${port}"
    dir_path="${home_path}/logs/vfl-lr/server/${dataset}_size${size}_client${n_clients}_server${n_servers}/"
    log_path=$dir_path$i"_"$port".log"
    [ "$server_count" -eq 0 ] && echo "creating log dir" && rm -rf $dir_path && mkdir -p $dir_path && rm -rf $log_path
    echo $node_name "source $env_path; cd $code_path; nohup python3 -u server.py $size $split_size $address $n_clients > $log_path 2>&1 &"
    ssh $node_name "source $env_path; cd $code_path; nohup python3 -u server.py $size $split_size $address $n_clients > $log_path 2>&1 &"
    server_count=$((server_count+1))
    [ "$server_count" -ge "$n_servers" ] && echo "enough servers" && exit 1
  done
done




