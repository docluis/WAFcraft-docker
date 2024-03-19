#!/bin/bash
./app/wafcraft/scripts/generate_raw_data.sh;

cd /app/wafcraft;

# Start Jupyter Notebook
exec jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root --NotebookApp.token='aC9Zsec4kHLAcYndnYoUsaZbM52LrT' --NotebookApp.password=''