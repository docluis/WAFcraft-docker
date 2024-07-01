#!/bin/bash

# This will rerun the crashed runs
# It will look for directories with tmp_* in them
# Next it will parse out the configuration
# It will then rerun the directories

from_ts="2024-05-10_15-00-00" # CHANGE THIS

cd /app/wafcraft/data/prepared

dirs=($(ls -d */))
dirs=($(for dir in ${dirs[@]}; do if [[ $dir > $from_ts ]]; then echo $dir; fi; done))
dirs=($(for dir in ${dirs[@]}; do echo ${dir::-1}; done))

rerun_dirs=()
for dir in ${dirs[@]}; do
    if find "$dir" -maxdepth 1 -type d -name 'tmp_*' | grep -q .; then
        rerun_dirs+=($dir)
    fi
done

configs=()
for dir in ${rerun_dirs[@]}; do
    config=$(grep -oP 'NAME: \K.*' $dir/config.txt)
    configs+=($config)
done

echo
echo "Rerunning the following directories:"
echo

for i in ${!rerun_dirs[@]}; do
    echo ${configs[$i]}: ${rerun_dirs[$i]}
done

echo
echo "Press enter to continue or CTRL+C to cancel"
read

cd /app/wafcraft
for i in ${!rerun_dirs[@]}; do
    echo "python main.py --data --config ${configs[$i]} --workspace ${rerun_dirs[$i]}"
    python main.py --data --config ${configs[$i]} --workspace ${rerun_dirs[$i]}
done

