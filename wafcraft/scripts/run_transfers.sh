#!/bin/bash

# This will run transfers with workspaces after a certain timestamp

from_ts="2024-05-20_12-27-00" # CHANGE THIS
target_workspace="2024-04-07_18-15-53_brown-lot" # CHANGE THIS
target_config="Target" # CHANGE THIS

cd /app/wafcraft/data/prepared

dirs=($(ls -d */))
dirs=($(for dir in ${dirs[@]}; do if [[ $dir > $from_ts ]]; then echo $dir; fi; done))
dirs=($(for dir in ${dirs[@]}; do echo ${dir::-1}; done))

configs=()
for dir in ${dirs[@]}; do
    config=$(grep -oP 'NAME: \K.*' $dir/config.txt)
    configs+=($config)
done

echo
echo "Running Transfers from these to :" $target_workspace
echo

for i in ${!dirs[@]}; do
    echo ${configs[$i]}: ${dirs[$i]}
done

echo
echo "Press enter to continue or CTRL+C to cancel"
read

cd /app/wafcraft
for i in ${!dirs[@]}; do
    echo  ${dirs[$i]} "->" $target_workspace
    python main.py --transfer --config $target_config --target $target_workspace --surrogate ${dirs[$i]}
done
