import os

configs = [
    "Surrogate_Data_V5", # this is RF, also with 100% overlap
    "Surrogate_SVM_V1",
    "Surrogate_GBoost_V1",
    "Surrogate_NaiveBayes_V1",
    "Surrogate_LogReg_V1",
    "Surrogate_KNN_V1",
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
            matching_workspaces.append(workspace)
    return matching_workspaces


def run_transfer(target_workspace, surrogate_workspace):
    os.system(
        f"python main.py --transfer --config Target --target {target_workspace} --surrogate {surrogate_workspace}"
    )


def run_transfers_nxn(workspaces):
    for i, target in enumerate(workspaces):
        print(f"Running transfers for target {i+1}/{len(workspaces)}: {target}")
        # filter out target from workspaces
        surrogates = [workspace for workspace in workspaces if workspace != target]
        for i, surrogate in enumerate(surrogates):
            print(f"Running transfer {i+1}/{len(surrogates)} of {i+1}/{len(workspaces)}: {surrogate}")
            print(f"{surrogate} -> {target}")
            run_transfer(target, surrogate)


workspaces = []
for config in configs:
    workspaces.extend(find_workspaces(config))
print(workspaces)

run_transfers_nxn(workspaces)
# ntfy
message = "Transfers completed"
os.system(
    f'curl -d "`hostname`: {message}" -H "Tags: hedgehog" ntfy.sh/luis-info-buysvauy12iiq -s -o /dev/null'
)
