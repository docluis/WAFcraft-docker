import pandas as pd
import numpy as np
import joblib
import base64
import os
import argparse

from tqdm import tqdm
from sklearn.metrics import confusion_matrix, roc_curve

from src.model import create_wafamole_model
from src.modsec import init_modsec
from src.utils import load_data_label_vector, read_and_parse_sql

# This script evaluates the performance of the created models.

# read arguments


parser = argparse.ArgumentParser()
parser.add_argument(
    "--use_adv_model", action="store_true", help="Whether to use the adversarial model"
)
parser.add_argument(
    "--use_adv_test",
    action="store_true",
    help="Whether to use the adversarial test set",
)
parser.add_argument("--workspace", type=str, required=True, help="Workspace directory")

args = parser.parse_args()

workspace = args.workspace

use_adv_model = args.use_adv_model  # Whether to use the adversarial model
use_adv_test = args.use_adv_test  # Whether to use the adversarial test set

print(
    f"Workspace: {workspace} - Use adv model: {use_adv_model} - Use adv test: {use_adv_test}"
)

base_data = "full"  # 20k or 40k or full
test_description = f"trained on full; test from {base_data}"
# test_description = "fresh sampled from full data, extra shuffled"

rule_ids = [
    "942011",
    "942012",
    "942013",
    "942014",
    "942015",
    "942016",
    "942017",
    "942018",
    "942100",
    "942101",
    "942110",
    "942120",
    "942130",
    "942131",
    "942140",
    "942150",
    "942151",
    "942152",
    "942160",
    "942170",
    "942180",
    "942190",
    "942200",
    "942210",
    "942220",
    "942230",
    "942240",
    "942250",
    "942251",
    "942260",
    "942270",
    "942280",
    "942290",
    "942300",
    "942310",
    "942320",
    "942321",
    "942330",
    "942340",
    "942350",
    "942360",
    "942361",
    "942362",
    "942370",
    "942380",
    "942390",
    "942400",
    "942410",
    "942420",
    "942421",
    "942430",
    "942431",
    "942432",
    "942440",
    "942450",
    "942460",
    "942470",
    "942480",
    "942490",
    "942500",
    "942510",
    "942511",
    "942520",
    "942521",
    "942522",
    "942530",
    "942540",
    "942550",
    "942560",
]


def new_test_set(workspace):
    print("Creating new test set")
    attacks = read_and_parse_sql(f"data/raw/attacks_{base_data}.sql")
    attacks["label"] = 1
    print("- attacks read")
    sanes = read_and_parse_sql(f"data/raw/sanes_{base_data}.sql")
    sanes["label"] = 0
    print("- sanes read")

    train = load_data_label_vector(f"/app/wafcraft/data/prepared/{workspace}/train.csv")
    train = train.drop(columns=["vector"])

    attacks_candidates = (
        attacks[~attacks["data"].isin(train["data"])]
        .sample(frac=1)
        .reset_index(drop=True)
    )
    sanes_candidates = (
        sanes[~sanes["data"].isin(train["data"])].sample(frac=1).reset_index(drop=True)
    )
    print("- candidates selected")

    test_attacks = attacks_candidates.sample(n=2000)
    test_sanes = sanes_candidates.sample(n=2000)

    test = pd.concat([test_attacks, test_sanes]).sample(frac=1).reset_index(drop=True)
    return test


def load_model(use_adv_model):
    model_type = "model_adv" if use_adv_model else "model"

    workspace = "/app/wafcraft/data/prepared/2024-04-07_18-15-53_brown-lot"

    model = joblib.load(f"{workspace}/{model_type}/{model_type}.joblib")
    return model


def evaluate_model(wafamole_model, test):
    preds = []
    for i, row in tqdm(test.iterrows(), total=len(test)):
        payload_b64 = row["data"]
        payload = base64.b64decode(payload_b64)
        label = row["label"]
        confidence_is_attack = wafamole_model.classify(payload.decode("utf-8"))
        preds.append((label, confidence_is_attack))
    labels, confidences = zip(*preds)
    labels = np.array(labels)
    confidences = np.array(confidences)

    desired_fpr = 0.01
    fpr, tpr, thresholds = roc_curve(labels, confidences)
    closest_idx = np.argmin(np.abs(fpr - desired_fpr))
    threshold = thresholds[closest_idx]

    predictions = (confidences >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(labels, predictions).ravel()
    tpr_actuall = tp / (tp + fn)
    fpr_actuall = fp / (fp + tn)

    return tpr_actuall, fpr_actuall


def main():
    print()
    print(f"Evaluating workspace {workspace}")
    test = new_test_set(workspace)
    model = load_model(use_adv_model)
    modsec = init_modsec()
    wafamole_model = create_wafamole_model(model, modsec, rule_ids, 4)

    tpr, fpr = evaluate_model(wafamole_model, test)
    print(f"TPR: {tpr}, FPR: {fpr}")

    if not os.path.exists("results/model_performance.csv"):
        with open("results/model_performance.csv", "w") as f:
            f.write("workspace,use_model_adv,use_test_adv,test_description,fpr,tpr\n")

    with open("results/model_performance.csv", "a") as f:
        f.write(
            f"{workspace},{use_adv_model},{use_adv_test},{test_description},{fpr},{tpr}\n"
        )


if __name__ == "__main__":
    main()
