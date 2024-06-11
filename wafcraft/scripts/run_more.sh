#!/bin/bash

cd /app/wafcraft

configs=("Surrogate_Data_V12" "Surrogate_Data_V13" "Surrogate_Data_V14")

for config in ${configs[@]}; do
    for i in {1..6}; do
        curl -d "$i/6 starting with $config" ntfy.sh/luis-info-buysvauy12iiq
        python main.py --data --new --config $config || curl -d "something went wrong with $config" ntfy.sh/luis-info-buysvauy12iiq
    done
done

curl -d "done with all adv sets" ntfy.sh/luis-info-buysvauy12iiq


configs_nonadv=("NoAdv_Surrogate_Data_V1" "NoAdv_Surrogate_Data_V2" "NoAdv_Surrogate_Data_V3" "NoAdv_Surrogate_Data_V4")

for config in ${configs_nonadv[@]}; do
    for i in {1..6}; do
        curl -d "$i/6 starting with $config" ntfy.sh/luis-info-buysvauy12iiq
        python main.py --data --new --config $config || curl -d "something went wrong with $config" ntfy.sh/luis-info-buysvauy12iiq
    done
done

curl -d "done with all all sets" ntfy.sh/luis-info-buysvauy12iiq