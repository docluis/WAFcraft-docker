#!/bin/bash

cd /app/wafcraft;
mkdir data;
cd data;
mkdir raw;
cd raw;

cat /app/wafamole_dataset/attacks.sql.* > attacks_full.sql;
cat /app/wafamole_dataset/sane.sql.* > sanes_full.sql;

head -n 5000 attacks_full.sql > attacks_5k.sql;
head -n 5000 sanes_full.sql > sanes_5k.sql;

head -n 20000 attacks_full.sql > attacks_20k.sql;
head -n 20000 sanes_full.sql > sanes_20k.sql;

echo "Files created successfully!";