

from os import listdir


prepared_dir = "/app/wafcraft/data/prepared"
workspace_names = listdir(prepared_dir)

search_str = "NAME: Surrogate_Data_V1"

for workspace in workspace_names:
    with open(f"{prepared_dir}/{workspace}/config.txt", "r") as f:
        config = f.read()
    if search_str in config:
        print(workspace)

