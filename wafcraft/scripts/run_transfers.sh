#!/bin/bash

cd /app/wafcraft

target="2024-05-06_11-25-19_honeydew-tough"

surrogates=("2024-05-07_02-30-05_darkblue-imagine" "2024-05-06_14-54-00_orangered-skill" "2024-05-06_11-52-12_peachpuff-despite" "2024-05-06_23-23-04_limegreen-prove" "2024-05-06_17-43-19_aqua-value" "2024-05-06_20-28-46_lime-star" "2024-05-07_05-33-27_chocolate-sit" "2024-05-07_11-44-50_cornflowerblue-mouth" "2024-05-07_20-56-24_tan-choice" "2024-05-07_14-54-59_olive-east" "2024-05-07_18-06-34_burlywood-prove" "2024-05-07_08-44-08_blanchedalmond-tv" "2024-05-08_14-18-20_sandybrown-low" "2024-05-08_11-20-14_silver-them" "2024-05-07_23-36-27_lightblue-investment" "2024-05-08_05-35-56_floralwhite-with" "2024-05-08_08-35-13_yellowgreen-throughout" "2024-05-08_02-32-03_gold-news" "2024-05-08_23-13-40_moccasin-buy" "2024-05-08_20-18-01_darkslategray-technology" "2024-05-09_05-18-07_red-group" "2024-05-09_08-29-03_lawngreen-clear" "2024-05-09_02-23-55_turquoise-part" "2024-05-08_17-12-06_blueviolet-all" "2024-05-10_01-41-38_lightslategray-operation" "2024-05-09_11-28-00_lemonchiffon-carry" "2024-05-09_14-13-30_orangered-collection" "2024-05-09_17-17-19_lightsalmon-manage" "2024-05-09_23-02-23_lightcoral-owner" "2024-05-09_20-00-28_mediumpurple-class")
for surrogate in ${surrogates[@]}; do
    python main.py --transfer --config NoAdv_Target --target $target --surrogate $surrogate --target_use_adv 0 --surrogate_use_adv 0
done

curl -d "$(hostname) done transfers" ntfy.sh/luis-info-buysvauy12iiq
