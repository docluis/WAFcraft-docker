import os

# architecutres
# configs = [
#     "Surrogate_Data_V5", # this is RF, also with 100% overlap
#     "Surrogate_SVM_V1",
#     "Surrogate_GBoost_V1",
#     "Surrogate_NaiveBayes_V1",
#     "Surrogate_LogReg_V1",
#     "Surrogate_KNN_V1",
# ]

# paranoia levels
configs = [
    "Surrogate_Paranoia_V1",
    "Surrogate_Paranoia_V2",
    "Surrogate_Paranoia_V3",
    "Surrogate_Data_V5",
]

ignore = [
    "2024-04-29_07-05-06_darkslategray-approach",
    "2024-05-15_14-55-23_red-resource",
    "2024-05-15_06-47-54_mediumpurple-nothing",
    "2024-05-15_22-43-02_lightseagreen-first",
    "2024-04-12_18-12-33_crimson-clear",  # old PL1
    "2024-04-13_13-56-32_linen-generation",  # old PL2
    "2024-04-13_21-57-35_seagreen-in",  # old PL3
]


def find_workspaces(config):
    prepared_dir = "/app/wafcraft/data/prepared"
    workspace_names = os.listdir(prepared_dir)
    search_str = f": {config}\n"
    matching_workspaces = []
    for workspace in workspace_names:
        with open(f"{prepared_dir}/{workspace}/config.txt", "r") as f:
            config = f.read()
        if search_str in config:
            if workspace not in ignore:
                matching_workspaces.append(workspace)
    return matching_workspaces


def run_transfer(target_workspace, surrogate_workspace):
    os.system(
        f"python main.py --transfer --config Target --target {target_workspace} --surrogate {surrogate_workspace} --samples 400"
    )


def run_transfers_nxn(workspaces):
    for i, target in enumerate(workspaces):
        print(f"Running transfers for target {i+1}/{len(workspaces)}: {target}")
        # filter out target from workspaces
        surrogates = [workspace for workspace in workspaces if workspace != target]
        for j, surrogate in enumerate(surrogates):
            print(
                f"Running transfer {j+1}/{len(surrogates)} of target {i+1}/{len(workspaces)}: {surrogate}"
            )
            print(f"{surrogate} -> {target}")
            run_transfer(target, surrogate)


workspaces = []
for config in configs:
    workspaces.extend(find_workspaces(config))
print(workspaces)

run_transfers_nxn(workspaces)
# ntfy