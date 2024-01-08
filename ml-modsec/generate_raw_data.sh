#!/bin/bash

cd /app/ml-modsec;
mkdir data;
cd data;
cat /app/wafamole_dataset/attacks.sql.* > attacks_full.sql;
cat /app/wafamole_dataset/sane.sql.* > sanes_full.sql;

head -n 5000 attacks_full.sql > attacks_5k.sql;
head -n 5000 sane_full.sql > sanes_5k.sql;

echo "Files created successfully!";