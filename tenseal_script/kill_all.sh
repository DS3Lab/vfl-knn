#!/bin/bash

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
