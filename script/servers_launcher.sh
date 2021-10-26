#!/bin/bash

# This script will initialize n Servers to received encrypted data and apply addition
# python3 server.py size split_size n_clients k n_test method(e.g., allreduce) n_servers cluter_name

# for higgs
# ./servers_launcher.sh 30 3 5 5 10 allreduce 5 bach

# for a6a/a1a
# ./servers_launcher.sh 125 3 5 3 3 allreduce 5 bach

# for synthesis
# ./servers_launcher.sh 50 5 5 5 10 allreduce 5 bach

# for Bank
# ./servers_launcher.sh 10000 5 4 5 3200 allreduce 5 bach
# ./servers_launcher.sh 10000 5 4 3 3200 allreduce 5 bach
# ./servers_launcher.sh 10000 1 4 5 10000 allreduce 1 bach

# for G2
# ./servers_launcher.sh 1000 2 4 5 205 allreduce 5 bach

# Birch1/Birch2
# ./servers_launcher.sh 10000 1 2 5 10000 allreduce 5 bach

# letter
# ./servers_launcher.sh 10000 4 4 5 2000 allreduce 5 bach

# unbalance
# ./servers_launcher.sh 650 1 2 5 650 allreduce 5 bach

# server parameters
size=$1
split_size=$2
# should be n_worker - 1, the first worker is the coordinator
n_clients=$3

# Log info
k=$4
n_test=$5
method=$6

n_servers=$7
cluster_name=$8

[ -z "$size" ] && echo "var1 size is empty" && exit 1
[ -z "$split_size" ] && echo "var2 split_size is empty" && exit 1
[ -z "$n_clients" ] && echo "var3 n_clients is empty" && exit 1
[ -z "$k" ] && echo "var4 k is empty" && exit 1
[ -z "$n_test" ] && echo "var5 n_test is empty" && exit 1
[ -z "$method" ] && echo "var6 method is empty" && exit 1
[ -z "$n_servers" ] && echo "var7 n_servers is empty" && exit 1

home_path="/home/jiangjia/"
code_path="${home_path}/code/vfl-knn/transmission/pallier/"
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
    dir_path="${home_path}/logs/vfl-knn/server/${method}_k${k}_t${n_test}_size${size}/"
    log_path=$dir_path$i"_"$port".log"
    [ "$server_count" -eq 0 ] && echo "creating log dir" && rm -rf $dir_path && mkdir -p $dir_path && rm -rf $log_path
    echo $node_name "source $env_path; cd $code_path; nohup python3 -u server.py $size $split_size $address $n_clients > $log_path 2>&1 &"
    ssh $node_name "source $env_path; cd $code_path; nohup python3 -u server.py $size $split_size $address $n_clients > $log_path 2>&1 &"
    server_count=$((server_count+1))
    [ "$server_count" -ge "$n_servers" ] && echo "enough servers" && exit 1
  done
done




