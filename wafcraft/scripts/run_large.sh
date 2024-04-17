#!/bin/bash

configs=("Large_Surrogate_SVM_V1" "Large_Surrogate_Data_V1" "Large_Surrogate_Data_V4")

# for each do python main.py --data --new --config $config || curl -d "something went wrong :/" ntfy.sh/luis-info-buysvauy12iiq

for config in ${configs[@]}; do
    python main.py --data --new --config $config || curl -d "something went wrong :/" ntfy.sh/luis-info-buysvauy12iiq
done

curl -d "`hostname` done with all large sets" ntfy.sh/luis-info-buysvauy12iiq;