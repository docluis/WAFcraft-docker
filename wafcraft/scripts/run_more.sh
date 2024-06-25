#!/bin/bash

cd /app/wafcraft

configs_more=( "Surrogate_Data_V11" "Surrogate_Data_V12" "Surrogate_Data_V13" "Surrogate_Data_V14")

for config in ${configs_more[@]}; do
    for i in {1..4}; do
        curl -d "$i/4 starting with $config" ntfy.sh/luis-info-buysvauy12iiq
        python main.py --data --new --config $config || curl -d "something went wrong with $config" ntfy.sh/luis-info-buysvauy12iiq
    done
done

curl -d "done with all all sets" ntfy.sh/luis-info-buysvauy12iiq