import numpy as np
from src.utils import load_data_label_vector

rule_ids = ['942011', '942012', '942013', '942014', '942015', '942016', '942017', '942018', '942100', '942101', '942110', '942120', '942130', '942131', '942140', '942150', '942151', '942152', '942160', '942170', '942180', '942190', '942200', '942210', '942220', '942230', '942240', '942250', '942251', '942260', '942270', '942280', '942290', '942300', '942310', '942320', '942321', '942330', '942340', '942350', '942360', '942361', '942362', '942370', '942380', '942390', '942400', '942410', '942420', '942421', '942430', '942431', '942432', '942440', '942450', '942460', '942470', '942480', '942490', '942500', '942510', '942511', '942520', '942521', '942522', '942530', '942540', '942550', '942560']

workspaces = [
    "2024-04-18_14-12-51_lightblue-around",
    "2024-05-10_15-03-09_darkred-number",
    "2024-04-22_11-20-36_yellow-majority",
    "2024-05-10_23-07-35_beige-western",
    "2024-04-23_05-02-11_cadetblue-right",
    "2024-04-08_21-57-36_greenyellow-fear",
    "2024-05-11_07-05-20_honeydew-check",
    "2024-04-23_02-58-14_blanchedalmond-table",
    "2024-04-22_18-57-13_darkslateblue-air",
    "2024-05-11_14-57-56_darkgray-general",
]

def get_activated(workspace, dset):
    data = load_data_label_vector(f"/app/wafcraft/data/prepared/{workspace}/{dset}.csv")
    activated = data['vector'][0]
    for i in range(len(data)):
        vec = data['vector'][i]
        # update the activated vector by or-ing it with the current vector
        activated = [a or b for a, b in zip(activated, vec)]
    return activated

def get_activated_per_set(dset):
    all_activated = np.zeros(len(rule_ids)) 
    for workspace in workspaces:
        print(workspace)
        activated = get_activated(workspace, dset)
        # now or over all activated vectors
        all_activated = [a or b for a, b in zip(all_activated, activated)]
    return all_activated

all_all_activated = np.zeros(len(rule_ids))
for dset in ['train', 'test', 'train_adv', 'test_adv']:
    all_activated = get_activated_per_set(dset)
    all_all_activated = [a or b for a, b in zip(all_all_activated, all_activated)]
    

print(all_all_activated)
non_activated = [not a for a in all_all_activated]
# now print the rule_ids that are not activated
print([rule_ids[i] for i in range(len(rule_ids)) if non_activated[i]])

print(f"Total number non activated rules: {sum(non_activated)}")
print(f"Total number activated rules: {sum(all_all_activated)}")