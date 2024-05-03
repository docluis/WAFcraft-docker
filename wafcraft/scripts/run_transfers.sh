#!/bin/bash

cd /app/wafcraft

target="2024-04-07_18-15-53_brown-lot"

surrogates=( "2024-04-29_15-29-39_darkkhaki-quickly" "2024-04-29_07-05-06_darkslategray-approach" )

for surrogate in ${surrogates[@]}; do
    python main.py --transfer --config Target --target $target --surrogate $surrogate
done

curl -d "`hostname` done transfers" ntfy.sh/luis-info-buysvauy12iiq;