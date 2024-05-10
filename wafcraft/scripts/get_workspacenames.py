from os import listdir


prepared_dir = "/app/wafcraft/data/prepared"
workspace_names = listdir(prepared_dir)

search_str = "NoAdv_Surrogate_Data_V5"

target = "2024-05-06_11-25-19_honeydew-tough"
results_target_dir = "/app/wafcraft/results/" + target

matching_workspaces = []
for workspace in workspace_names:
    with open(f"{prepared_dir}/{workspace}/config.txt", "r") as f:
        config = f.read()
    if search_str in config:
        matching_workspaces.append(workspace)

# open the transferability.csv file
try:
    with open(f"{results_target_dir}/transferability.csv", "r") as f:
        transferability = f.read()
except FileNotFoundError:
    transferability = ""

# print the matching workspaces that are not in the transferability.csv file
for workspace in matching_workspaces:
    if workspace not in transferability:
        print(workspace)