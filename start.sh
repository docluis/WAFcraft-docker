#!/bin/bash
cd /app/ml-modsec && \
./generate_raw_data.sh;

# Start Jupyter Notebook
exec jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root --NotebookApp.token='aC9Zsec4kHLAcYndnYoUsaZbM52LrT' --NotebookApp.password=''