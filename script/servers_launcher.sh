#!/bin/bash

# This script will initialize n Servers to received encrypted data and apply addition
# python3 server.py size splitSize address nClients

# Server parameters
size=$1
split_size=$2
n_clients=$3

# Log info
k=$4
n_test=$5
method=$6

n_servers=$7


[ -z "$size" ] && echo "var1 size is empty" && exit 1
[ -z "$split_size" ] && echo "var2 split_size is empty" && exit 1
[ -z "$n_clients" ] && echo "var3 n_clients is empty" && exit 1
[ -z "$k" ] && echo "var4 k is empty" && exit 1
[ -z "$n_test" ] && echo "var5 n_test is empty" && exit 1
[ -z "$method" ] && echo "var6 method is empty" && exit 1
[ -z "$n_servers" ] && echo "var7 n_servers is empty" && exit 1

home_path="/mnt/scratch/ldiego"
code_path="${home_path}/transmission_framework/"
env_path="/mnt/scratch/ldiego/tutorial-env/bin/activate"

machines=( 13 14 15 16 )
ports=( 8991 8992 8993 8994 )
cluster_name="bach"
server_count=0
for port in "${ports[@]}"; do
  for i in "${machines[@]}"; do

    # Connection to machines
    node_name=${cluster_name}$i
    address="${node_name}.ethz.ch:${port}"
    dir_path="${home_path}/logs/server/${method}_k${k}_t${n_test}_size${size}/"
    log_path=$dir_path$i"_"$port".log"
    [ "$server_count" -eq 0 ] && echo "creating log dir" && rm -rf $dir_path && mkdir -p $dir_path && rm -f $log_path
    echo $node_name "source $env_path; cd $code_path; nohup python3 -u server.py $size $split_size $address $n_clients > $log_path 2>&1 &"
    ssh $node_name "source $env_path; cd $code_path; nohup python3 -u server.py $size $split_size $address $n_clients > $log_path 2>&1 &"
    server_count=$((server_count+1))
    [ "$server_count" -ge "$n_servers" ] && echo "enough servers" && exit 1
  done
done




