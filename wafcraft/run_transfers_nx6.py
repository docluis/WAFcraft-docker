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
surrogate_configs = [
    "NoAdv_Surrogate_Data_V1",
    "NoAdv_Surrogate_Data_V2",
    "NoAdv_Surrogate_Data_V3",
    "NoAdv_Surrogate_Data_V4",
    "NoAdv_Surrogate_Data_V5",
]

target_configs = [
    "NoAdv_Surrogate_Data_V5",
]

ignore = [
    "2024-04-29_07-05-06_darkslategray-approach",
    "2024-05-15_14-55-23_red-resource",
    "2024-05-15_06-47-54_mediumpurple-nothing",
    "2024-05-15_22-43-02_lightseagreen-first",
    "2024-04-12_18-12-33_crimson-clear",  # old PL1
    "2024-04-13_13-56-32_linen-generation",  # old PL2
    "2024-04-13_21-57-35_seagreen-in",  # old PL3
    "2024-05-07_02-30-05_darkblue-imagine",  # old NoAdv_Surrogate_Data_V1
    "2024-05-06_14-54-00_orangered-skill",  # old NoAdv_Surrogate_Data_V1
    "2024-05-06_11-52-12_peachpuff-despite",  # old NoAdv_Surrogate_Data_V1
    "2024-05-06_23-23-04_limegreen-prove",  # old NoAdv_Surrogate_Data_V1
    "2024-05-06_17-43-19_aqua-value",  # old NoAdv_Surrogate_Data_V1
    "2024-05-06_20-28-46_lime-star",  # old NoAdv_Surrogate_Data_V1
    "2024-05-07_05-33-27_chocolate-sit", # old NoAdv_Surrogate_Data_V2
    "2024-05-07_11-44-50_cornflowerblue-mouth", # old NoAdv_Surrogate_Data_V2
    "2024-05-07_20-56-24_tan-choice", # old NoAdv_Surrogate_Data_V2
    "2024-05-07_14-54-59_olive-east", # old NoAdv_Surrogate_Data_V2
    "2024-05-07_18-06-34_burlywood-prove", # old NoAdv_Surrogate_Data_V2
    "2024-05-07_08-44-08_blanchedalmond-tv", # old NoAdv_Surrogate_Data_V2
    "2024-05-08_14-18-20_sandybrown-low", # old NoAdv_Surrogate_Data_V3
    "2024-05-08_11-20-14_silver-them", # old NoAdv_Surrogate_Data_V3
    "2024-05-07_23-36-27_lightblue-investment", # old NoAdv_Surrogate_Data_V3
    "2024-05-08_05-35-56_floralwhite-with", # old NoAdv_Surrogate_Data_V3
    "2024-05-08_08-35-13_yellowgreen-throughout", # old NoAdv_Surrogate_Data_V3
    "2024-05-08_02-32-03_gold-news", # old NoAdv_Surrogate_Data_V3
    "2024-05-08_23-13-40_moccasin-buy", # old NoAdv_Surrogate_Data_V4
    "2024-05-08_20-18-01_darkslategray-technology", # old NoAdv_Surrogate_Data_V4
    "2024-05-09_05-18-07_red-group", # old NoAdv_Surrogate_Data_V4
    "2024-05-09_08-29-03_lawngreen-clear", # old NoAdv_Surrogate_Data_V4
    "2024-05-09_02-23-55_turquoise-part", # old NoAdv_Surrogate_Data_V4
    "2024-05-08_17-12-06_blueviolet-all", # old NoAdv_Surrogate_Data_V4
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
        f"python main.py --transfer --config Target --target {target_workspace} --surrogate {surrogate_workspace} --target_use_nonadv --surrogate_use_nonadv --samples 1000"
    )


def run_transfers_nxn(surrogate_workspaces, target_workspaces):
    for i, target in enumerate(target_workspaces):
        print()
        print(f"Running transfers for target {i+1}/{len(target_workspaces)}: {target}")
        # filter out target from workspaces
        surrogates = [x for x in surrogate_workspaces if x != target]
        for j, surrogate in enumerate(surrogates):
            print(
                f"Running transfer {j+1}/{len(surrogates)} of target {i+1}/{len(target_workspaces)}"
            )
            print(f"{surrogate} -> {target}")
            run_transfer(target, surrogate)


surrogate_workspaces = []
target_workspaces = []
for config in surrogate_configs:
    surrogate_workspaces.extend(find_workspaces(config))
print(surrogate_workspaces)
for config in target_configs:
    target_workspaces.extend(find_workspaces(config))
print(target_workspaces)

run_transfers_nxn(surrogate_workspaces, target_workspaces)
# # ntfy
message = "Transfers completed"
os.system(
    f'curl -d "`hostname`: {message}" -H "Tags: hedgehog" ntfy.sh/luis-info-buysvauy12iiq -s -o /dev/null'
)
