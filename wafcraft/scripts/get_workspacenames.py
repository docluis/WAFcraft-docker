from os import listdir

# This script is used to get the names of the workspaces that
# contain a specific search string in their config.txt file.
#
# Optionally, the script can filter out the workspaces that are
# already in the transferability.csv file of the target.

search_str = ": NoAdv_Surrogate_Data_V5\n"  # CHANGE THIS
target = "2024-06-09_20-11-04_lightslategray-them"  # CHANGE THIS
filter_for_non_transfered = False  # CHANGE THIS

prepared_dir = "/app/wafcraft/data/prepared"
workspace_names = listdir(prepared_dir)


results_target_dir = "/app/wafcraft/results/" + target

matching_workspaces = []
for workspace in workspace_names:
    with open(f"{prepared_dir}/{workspace}/config.txt", "r") as f:
        config = f.read()
    if search_str in config:
        matching_workspaces.append(workspace)

if filter_for_non_transfered:
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
else:
    for workspace in matching_workspaces:
        print(workspace)
    # print all in one line in quotes space separated
    print()
    print(" ".join([f'"{workspace}"' for workspace in matching_workspaces]))
