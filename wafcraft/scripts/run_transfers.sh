#!/bin/bash

cd /app/wafcraft

target="2024-04-15_10-04-17_turquoise-community"
surrogates=("2024-04-15_18-10-24_indigo-long" "2024-04-16_15-06-58_chartreuse-good")

for surrogate in ${surrogates[@]}; do
    python main.py --transfer --config Large_Target --target $target --surrogate $surrogate
done