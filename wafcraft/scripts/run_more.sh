#!/bin/bash

cd /app/wafcraft

configs=("NoAdv40k_Target")

for config in ${configs[@]}; do
    for i in {1..7}; do
        curl -d "$i/7 starting with $config" ntfy.sh/luis-info-buysvauy12iiq
        python main.py --data --new --config $config || curl -d "something went wrong with $config" ntfy.sh/luis-info-buysvauy12iiq
        curl -d "$i/7 done with $config" ntfy.sh/luis-info-buysvauy12iiq
    done
done

curl -d "done with all sets" ntfy.sh/luis-info-buysvauy12iiq
