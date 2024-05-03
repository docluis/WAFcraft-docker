#!/bin/bash

cd /app/wafcraft



for i in {1..4}
do
    python main.py --data --new --config Surrogate_Data_V1 || curl -d "something went wrong with 0% overlaps" ntfy.sh/luis-info-buysvauy12iiq
    curl -d "$i done with 0% overlaps" ntfy.sh/luis-info-buysvauy12iiq
done


configs=("Surrogate_Data_V2" "Surrogate_Data_V3" "Surrogate_Data_V4" "Surrogate_Data_V5")

for config in ${configs[@]}; do
    for i in {1..5}
    do
        python main.py --data --new --config $config || curl -d "something went wrong with $config" ntfy.sh/luis-info-buysvauy12iiq
        curl -d "$i done with $config" ntfy.sh/luis-info-buysvauy12iiq
    done
done

curl -d "done with all sets" ntfy.sh/luis-info-buysvauy12iiq;