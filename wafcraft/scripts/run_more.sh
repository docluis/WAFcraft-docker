#!/bin/bash

cd /app/wafcraft

configs_more=( "Surrogate_Data_V11" "Surrogate_Data_V12" "Surrogate_Data_V13" "Surrogate_Data_V14")

for config in ${configs_more[@]}; do
    for i in {1..4}; do
        python main.py --data --new --config $config
    done
done
