#!/bin/bash

# ./shapley_aggr_sche_servers_launcher.sh n_clients config dataset k n_test n_server cluster_name

# bank, N=3200, D=8, C=2
# ./shapley_aggr_sche_servers_launcher.sh 4 bank 3 320 4 bach

# parameters
n_clients=$1
dataset=$2
k=$3
n_test=$4
n_sche_servers=$5
cluster_name=$6

[ -z "$n_clients" ] && echo "var1 n_clients is empty" && exit 1
[ -z "$dataset" ] && echo "var3 method is empty" && exit 1
[ -z "$k" ] && echo "var4 k is empty" && exit 1
[ -z "$n_test" ] && echo "var5 n_test is empty" && exit 1
[ -z "$n_sche_servers" ] && echo "var6 n_servers is empty" && exit 1
[ -z "$cluster_name" ] && echo "var7 cluster_name is empty" && exit 1

home_path="/home/jiangjia/"
env_path="${home_path}/virtualenvs/pytorch-euler/bin/activate"
code_path="${home_path}/code/vfl-knn/transmission/tenseal_mi/"
config="ts_ckks.config"

aggr_servers=( 11 )
aggr_port=8991

sche_servers=( 04 06 07 08 )
sche_port=8992
sche_servers_str=""

sche_server_count=0
for sche in "${sche_servers[@]}"; do
  sche_servers_str="${sche_servers_str},${cluster_name}${sche}:${sche_port}"
  sche_server_count=$((sche_server_count+1))
  [ "$sche_server_count" -ge "$n_clients" ] && echo "enough schedule servers" && break
done
echo $sche_servers_str

aggr_node=${cluster_name}${aggr_servers[0]}
aggr_address="${aggr_node}:${aggr_port}"
dir_path="${home_path}/logs/vfl-knn/mi_server/tenseal_mi_aggr_server_${dataset}_k${k}_t${n_test}/"
log_path="${dir_path}${i}_${aggr_port}.log"
rm -rf $dir_path && mkdir -p $dir_path && rm -rf $log_path
source $env_path
cd $code_path
nohup python3 -u tenseal_mi_aggr_server.py $aggr_address $sche_servers_str $n_clients $k $config > $log_path 2>&1 &


sche_server_count=0
for i in "${sche_servers[@]}"; do
  sche_node=${cluster_name}$i
  sche_address="${sche_node}:${sche_port}"
  dir_path="${home_path}/logs/vfl-knn/mi_server/tenseal_mi_sche_server_${dataset}_k${k}_t${n_test}/"
  log_path="${dir_path}${i}_${sche_port}.log"
  [ "$sche_server_count" -eq 0 ] && echo "creating log dir" && rm -rf $dir_path && mkdir -p $dir_path && rm -rf $log_path
  echo $sche_node "source $env_path; cd $code_path; nohup python3 -u tenseal_mi_sche_server.py $sche_address $config > $log_path 2>&1 &"
  ssh $sche_node "source $env_path; cd $code_path; nohup python3 -u tenseal_mi_sche_server.py $sche_address $config > $log_path 2>&1 &"
  sche_server_count=$((sche_server_count+1))
  [ "$sche_server_count" -ge "$n_sche_servers" ] && echo "enough sched servers" && exit 1
  done
done
