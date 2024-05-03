#!/bin/bash

cd /app/wafcraft

target="2024-04-07_18-15-53_brown-lot"
surrogates=("2024-04-10_13-36-12_darkcyan-why" "2024-04-28_07-18-23_lightcoral-pretty" "2024-04-28_15-07-53_aliceblue-suddenly" "2024-04-28_22-59-40_blueviolet-physical")

for surrogate in ${surrogates[@]}; do
    python main.py --transfer --config Target --target $target --surrogate $surrogate --samples 400
done

curl -d "`hostname` done transfers" ntfy.sh/luis-info-buysvauy12iiq;