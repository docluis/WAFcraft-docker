import joblib
import pandas as pd
from src.utils import load_data_label_vector
from sklearn.metrics import confusion_matrix


def check_performance(workspace, adv, testing):
    print()
    print(f"Workspace: {workspace}")
    workspace_fullpath = f"/app/wafcraft/data/prepared/{workspace}"
    model_dir = "model_adv" if adv else "model"
    model = joblib.load(f"{workspace_fullpath}/{model_dir}/{model_dir}.joblib")
    threshold = float(
        open(
            f'{workspace_fullpath}/{model_dir}/threshold{"_adv" if adv else ""}.txt',
            "r",
        ).read()
    )
    print(f"Threshold: {threshold}")
    print(f"Model: {model}")

    my_test = load_data_label_vector(f"{workspace_fullpath}/test.csv")
    other_test = load_data_label_vector(
        f"/app/wafcraft/data/prepared/2024-04-07_18-15-53_brown-lot/test.csv"
    )
    other_test_adv = load_data_label_vector(
        f"/app/wafcraft/data/prepared/2024-04-07_18-15-53_brown-lot/test_adv.csv"
    )
    my_test_adv = (
        load_data_label_vector(f"{workspace_fullpath}/test_adv.csv") if adv else None
    )

    # drop columns that are not vector or label
    my_test = my_test[["vector", "label"]]
    other_test = other_test[["vector", "label"]]
    my_test_adv = my_test_adv[["vector", "label"]] if adv else None
    other_test_adv = other_test_adv[["vector", "label"]]

    if testing == "my_all":
        test = pd.concat([my_test, my_test_adv])
    elif testing == "my_test":
        test = my_test
    elif testing == "my_test-adv":
        my_test_benign = my_test[my_test["label"] == 0]
        test = pd.concat([my_test_adv, my_test_benign]) if adv else None
        # test = my_test_adv
    elif testing == "other_test":
        test = other_test
    elif testing == "other_test-adv":
        other_test_benign = other_test[other_test["label"] == 0]
        test = pd.concat([other_test_adv, other_test_benign])
        # test = other_test_adv
    else:
        raise ValueError("Invalid testing option")

    print(f"Testing: {testing}")
    print(f"Test shape: {test.shape}")

    X_test, y_test = list(test["vector"]), test["label"]
    probabilities = model.predict_proba(X_test)[:, 1]
    adjusted_predictions = (probabilities >= threshold).astype(int)
    cm = confusion_matrix(y_test, adjusted_predictions)
    return cm


# pl 4, 0 overlap
# workspaces = [
#     "2024-04-18_14-12-51_lightblue-around",
#     "2024-05-10_15-03-09_darkred-number",
#     "2024-04-22_11-20-36_yellow-majority",
#     "2024-05-10_23-07-35_beige-western",
#     "2024-04-23_05-02-11_cadetblue-right",
#     "2024-04-08_21-57-36_greenyellow-fear",
#     "2024-05-11_07-05-20_honeydew-check",
#     "2024-04-23_02-58-14_blanchedalmond-table",
#     "2024-04-22_18-57-13_darkslateblue-air",
#     "2024-05-11_14-57-56_darkgray-general",
# ]
# PL 1
# workspaces = [
#     "2024-06-01_19-29-44_forestgreen-history",
#     "2024-04-12_18-12-33_crimson-clear",
#     "2024-06-03_03-24-49_azure-tax",
#     "2024-06-02_19-01-47_mintcream-hotel",
#     "2024-06-01_11-41-52_royalblue-stay",
#     "2024-06-02_03-06-38_navajowhite-fund",
#     "2024-06-02_10-45-26_plum-early",
# ]
# PL 2
# workspaces = [
#     # "2024-06-03_21-29-45_lightslategray-feel",
#     "2024-04-13_13-56-32_linen-generation",
#     # "2024-06-04_10-20-22_springgreen-pattern",
#     "2024-06-03_11-16-37_steelblue-wrong",
#     # "2024-06-04_08-05-08_green-might",
#     "2024-06-03_23-40-39_aquamarine-less",
#     # "2024-06-03_19-18-06_lavender-final",
# ]
# PL 3
# workspaces = [
#     "2024-06-04_20-54-46_cadetblue-want",
#     # "2024-06-05_13-50-21_olivedrab-marriage",
#     # "2024-06-05_04-30-29_coral-million",
#     "2024-06-05_06-32-02_blueviolet-rather",
#     "2024-04-13_21-57-35_seagreen-in",
#     "2024-06-04_12-34-40_sienna-education",
#     "2024-06-05_15-58-09_gold-administration",
# ]
# new random one, no adv
# workspaces = [
#     "2024-06-06_16-50-23_olivedrab-wrong",
# ]
# svm models
workspaces = [
    "2024-05-20_21-11-36_linen-civil",
    "2024-04-08_10-49-51_paleturquoise-nor",
    "2024-05-20_12-27-17_plum-television",
    "2024-05-21_13-37-58_mediumvioletred-worker",
    "2024-05-21_04-59-27_peachpuff-perform",
    "2024-05-21_21-37-36_seagreen-together",
]


all_fprs = []
all_tprs = []

for workspace in workspaces:
    cm = check_performance(workspace, adv=True, testing="other_test")
    print(f"CM: {cm}")
    fpr = cm[0][1] / (cm[0][1] + cm[0][0])
    tpr = cm[1][1] / (cm[1][1] + cm[1][0])
    print(f"FPR: {fpr}")
    print(f"TPR: {tpr}")
    all_fprs.append(fpr)
    all_tprs.append(tpr)

print(f"Average FPR: {sum(all_fprs) / len(all_fprs)}")
print(f"Average TPR: {sum(all_tprs) / len(all_tprs)}")
