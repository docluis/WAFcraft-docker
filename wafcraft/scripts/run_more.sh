#!/bin/bash

cd /app/wafcraft

configs=("NoAdv_Surrogate_Data_V1")

for config in ${configs[@]}; do
    for i in {1..3}; do
        curl -d "$i/3 starting with $config" ntfy.sh/luis-info-buysvauy12iiq
        python main.py --data --new --config $config || curl -d "something went wrong with $config" ntfy.sh/luis-info-buysvauy12iiq
    done
done


configs_more=( "NoAdv_Surrogate_Data_V2" "NoAdv_Surrogate_Data_V3" "NoAdv_Surrogate_Data_V4")

for config in ${configs_more[@]}; do
    for i in {1..6}; do
        curl -d "$i/6 starting with $config" ntfy.sh/luis-info-buysvauy12iiq
        python main.py --data --new --config $config || curl -d "something went wrong with $config" ntfy.sh/luis-info-buysvauy12iiq
    done
done

curl -d "done with all all sets" ntfy.sh/luis-info-buysvauy12iiq