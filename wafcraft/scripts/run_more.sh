#!/bin/bash

cd /app/wafcraft

configs=("NoAdv_Surrogate_Data_V1" "NoAdv_Surrogate_Data_V2" "NoAdv_Surrogate_Data_V3" "NoAdv_Surrogate_Data_V4" "NoAdv_Surrogate_Data_V5")

for config in ${configs[@]}; do
    for i in {1..6}; do
        python main.py --data --new --config $config || curl -d "something went wrong with $config" ntfy.sh/luis-info-buysvauy12iiq
        curl -d "$i done with $config" ntfy.sh/luis-info-buysvauy12iiq
    done
done

curl -d "done with all sets" ntfy.sh/luis-info-buysvauy12iiq
