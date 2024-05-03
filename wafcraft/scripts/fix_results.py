

import pandas as pd


results_file = "/app/wafcraft/results/target_2024-04-15_10-04-17_turquoise-community/transferability.csv"

results = pd.read_csv(results_file)

# make a list of surrogate_workspace collumn
surrogate_workspaces = results["surrogate_workspace"].tolist()

# new dataframe with additional name column
results["config"] = ""

for surrogate in surrogate_workspaces:
    # read config.txt file
    with open(f"/app/wafcraft/data/prepared/{surrogate}/config.txt", "r") as f:
        config = f.read()
    # parse the NAME: (format is NAME: Surrogate_Data_V3)
    name = config.split("NAME: ")[1].split("\n")[0]
    # add the name to a new column
    results.loc[results["surrogate_workspace"] == surrogate, "config"] = name

# move the config column to the front
cols = results.columns.tolist()
cols = cols[-1:] + cols[:-1]
results = results[cols]
# save the new dataframe
results.to_csv(results_file, index=False)