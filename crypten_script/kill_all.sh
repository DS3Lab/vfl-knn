#!/bin/bash
#TODO clean + this on is the same in CLUSTER DONE
# 10 machines
# ./knn_allreduce.sh 10 28 2 10 10 /bigdata/dataset/higgs-vertical/ 172.31.38.0:24000 t2.medium-10
#./knn_allreduce.sh 10 10 28 2 10 ../../data/ 10.111.1.18:24000 bach
#10.111.1.18 172.17.0.1


#machines=( 11 12 13 14 16 )
machines=( 11 03 04 06 07 08 )

rank=1
for i in "${machines[@]}"; do
  node_name=bach$i
  if [[ $i == ${machines[0]} ]]
	then
	  echo $node_name
	  ssh $node_name "pkill python3" 
	else
	  echo $node_name
	  ssh $node_name "pkill python3"
    rank=$(($rank+1))
  fi
done
