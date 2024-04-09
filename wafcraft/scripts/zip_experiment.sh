#!/bin/bash

cd /app/wafcraft/data/prepared

date_time=$(date +"%Y-%m-%d_%H-%M-%S")

zip -r /app/wafcraft/data/zips/prepared_$date_time.zip *