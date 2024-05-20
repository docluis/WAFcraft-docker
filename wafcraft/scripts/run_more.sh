#!/bin/bash

cd /app/wafcraft

configs=("Surrogate_Data_V1" "Surrogate_Data_V2" "Surrogate_Data_V3" "Surrogate_Data_V4" "Surrogate_Data_V5")

for config in ${configs[@]}; do
    for i in {1..4}; do
        python main.py --data --new --config $config || curl -d "something went wrong with $config" ntfy.sh/luis-info-buysvauy12iiq
        curl -d "$i done with $config" ntfy.sh/luis-info-buysvauy12iiq
    done
done

curl -d "done with all sets" ntfy.sh/luis-info-buysvauy12iiq
