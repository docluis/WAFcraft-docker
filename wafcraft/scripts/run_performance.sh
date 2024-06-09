#!/bin/bash

cd /app/wafcraft

# trained on full
workspaces=("2024-04-18_14-12-51_lightblue-around" "2024-05-10_15-03-09_darkred-number" "2024-04-22_11-20-36_yellow-majority" "2024-05-10_23-07-35_beige-western" "2024-04-23_05-02-11_cadetblue-right" "2024-04-08_21-57-36_greenyellow-fear" "2024-05-11_07-05-20_honeydew-check" "2024-04-23_02-58-14_blanchedalmond-table" "2024-04-22_18-57-13_darkslateblue-air" "2024-05-11_14-57-56_darkgray-general")

# trained on 40k
# workspaces=("2024-06-09_16-46-14_darkorchid-later" "2024-06-09_16-47-28_lavenderblush-agent" "2024-06-09_16-48-40_mediumslateblue-agreement" "2024-06-09_16-58-58_lawngreen-loss" "2024-06-09_17-00-14_chartreuse-every" "2024-06-09_17-01-29_darkgoldenrod-focus" "2024-06-09_17-02-43_darkturquoise-another" "2024-06-09_17-03-56_deeppink-front" "2024-06-09_17-05-11_chartreuse-set" "2024-06-09_17-06-25_darkslateblue-present")


for workspace in ${workspaces[@]}; do
    python check_model_performance.py --workspace $workspace
done

curl -d "done with all performance checks" ntfy.sh/luis-info-buysvauy12iiq
