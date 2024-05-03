#!/bin/bash

cd /app/wafcraft;

python main.py --data --config Surrogate_Data_V1 --workspace 2024-04-23_02-58-14_blanchedalmond-table;
python main.py --data --config Surrogate_Data_V2 --workspace 2024-04-23_12-52-13_blue-cover;
python main.py --data --config Surrogate_Data_V3 --workspace 2024-04-25_06-58-57_lightpink-simple;
python main.py --data --config Surrogate_Data_V3 --workspace 2024-04-26_07-01-34_royalblue-machine;