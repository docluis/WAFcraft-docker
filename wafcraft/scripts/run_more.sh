#!/bin/bash

cd /app/wafcraft


for i in {1..5}; do
        python main.py --data --new --config "Surrogate_SVM_V1" || curl -d "something went wrong with Surrogate_SVM_V1" ntfy.sh/luis-info-buysvauy12iiq
        curl -d "$i/5 done with Surrogate_SVM_V1" ntfy.sh/luis-info-buysvauy12iiq
done

configs=("Surrogate_GBoost_V1" "Surrogate_NaiveBayes_V1" "Surrogate_LogReg_V1" "Surrogate_KNN_V1")

for config in ${configs[@]}; do
    for i in {1..6}; do
        python main.py --data --new --config $config || curl -d "something went wrong with $config" ntfy.sh/luis-info-buysvauy12iiq
        curl -d "$i/6 done with $config" ntfy.sh/luis-info-buysvauy12iiq
    done
done

curl -d "done with all sets" ntfy.sh/luis-info-buysvauy12iiq
