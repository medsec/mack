#!/bin/bash

#usage:
#more sha256_lower_3_pre | ./hash_line.sh > sha256_lower_3_hash
#change hash algo accordingly

while read line
do
echo -n $line | sha256sum | awk '{print $1}'
done
