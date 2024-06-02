#!/bin/bash

cd /app/wafcraft

configs=("Surrogate_Paranoia_V1" "Surrogate_Paranoia_V2" "Surrogate_Paranoia_V3")

for config in ${configs[@]}; do
    for i in {1..6}; do
        curl -d "$i/6 starting with $config" ntfy.sh/luis-info-buysvauy12iiq
        python main.py --data --new --config $config || curl -d "something went wrong with $config" ntfy.sh/luis-info-buysvauy12iiq
        curl -d "$i/6 done with $config" ntfy.sh/luis-info-buysvauy12iiq
    done
done

curl -d "done with all sets" ntfy.sh/luis-info-buysvauy12iiq
