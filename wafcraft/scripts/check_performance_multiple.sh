#!/bin/bash

cd /app/wafcraft

workspaces_onfull=("2024-04-18_14-12-51_lightblue-around" "2024-05-10_15-03-09_darkred-number" "2024-04-22_11-20-36_yellow-majority" "2024-05-10_23-07-35_beige-western" "2024-04-23_05-02-11_cadetblue-right")

workspaces_on40k=("2024-06-09_16-46-14_darkorchid-later" "2024-06-09_16-47-28_lavenderblush-agent" "2024-06-09_16-48-40_mediumslateblue-agreement" "2024-06-09_16-58-58_lawngreen-loss" "2024-06-09_17-00-14_chartreuse-every")


for workspace in ${workspaces_on40k[@]}; do
    python check_model_performance.py --workspace $workspace --base_data "40k" --note "train from 40k";
done

for workspace in ${workspaces_onfull[@]}; do
    python check_model_performance.py --workspace $workspace --base_data "full" --note "train from full";
done

printf "done with all workspaces\n"