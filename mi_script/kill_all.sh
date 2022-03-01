#!/bin/bash

# 10 machines
# ./kill_all.sh num

n_machines=$1

machines=( 11 03 04 06 07 08 )
machines=( 03 02 04 06 07 08 11 12 13 14 15 16 20 26 09 )

rank=0

for i in "${machines[@]}"; do
  node_name=bach$i
  if [[ $i == ${machines[0]} ]]
        then
          echo $node_name
          ssh $node_name "pkill python3"
        else
          echo $node_name
          ssh $node_name "pkill python3"
  fi
  rank=$(($rank+1))
  [ "$rank" -ge "$n_machines" ] && echo "enough machines" && exit 1
done
