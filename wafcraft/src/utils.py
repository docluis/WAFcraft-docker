import base64
import re
import os

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import pandas as pd

from sklearn.metrics import precision_recall_curve

rules_path = "/app/wafcraft/rules"
log_path = "/app/wafcraft/logs/log.txt"


def log(message, notify=False):
    print(message)
    time = pd.Timestamp.now()
    with open(log_path, "a") as log_file:
        log_file.write(f"{time}: {message}\n")
    if notify:
        try:
            os.system(
                f'curl -d "`hostname`: {message}" -H "Tags: hedgehog" ntfy.sh/luis-info-buysvauy12iiq -s -o /dev/null'
            )
        except Exception as e:
            print(f"Not able to notify: {e}")


# TODO: improve this function
def get_rules_list():
    # read rules from each file in the rules directory
    all_rules = []
    # check if rules exist
    for rule_path in sorted(glob.glob(f"{rules_path}/*.conf")):
        with open(rule_path, "r") as rule_file:
            rules = rule_file.read()
            matches = re.findall(r"id:(\d+),", rules)
            # append matches to rules list
            all_rules.extend(matches)
    # return sorted list of unique rule IDs
    return sorted(set(all_rules))



def load_data_label_vector(file_path):
    """
    Reads a csv file and returns a dataframe with vectorized payloads

    Parameters:
        file_path (str): Path to the file containing payloads

    Returns:
        pd.DataFrame: Dataframe with vectorized payloads
    """
    data = pd.read_csv(file_path)
    # convert string in vector to numpy array
    data["vector"] = data["vector"].apply(lambda x: np.fromstring(x[1:-1], sep=" "))
    return data


def plot_cm(cm):
    plt.figure(figsize=(4, 3))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        xticklabels=["Sane", "Attack"],
        yticklabels=["Sane", "Attack"],
        cmap="Blues",
    )
    plt.gca().invert_yaxis()
    plt.gca().invert_xaxis()
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.title("Confusion Matrix")
    plt.show()


def plot_roc(fpr, tpr, closest_idx, desired_fpr):
    plt.figure(figsize=(4, 3))
    plt.plot(fpr, tpr, label="ROC curve")
    plt.ylabel("True Positive Rate (TPR)")
    plt.xlabel("False Positive Rate (FPR)")
    plt.title("ROC Curve")
    # plot closest point to desired FPR and add label and annotation, make sure its in foreground
    plt.scatter(
        fpr[closest_idx],
        tpr[closest_idx],
        color="red",
        label=f"Closest to FPR of {desired_fpr}",
        zorder=5,
    )
    plt.annotate(
        f"({round(fpr[closest_idx], 4)}, {round(tpr[closest_idx], 4)})",
        (fpr[closest_idx], tpr[closest_idx]),
        textcoords="offset points",
        xytext=(50, 0),
        ha="center",
    )
    plt.legend()
    plt.show()


def plot_precision_recall_curve(y_test, probabilities):
    precision, recall, thresholds = precision_recall_curve(y_test, probabilities)
    thresholds = np.append(thresholds, 1)
    plt.figure(figsize=(4, 3))
    plt.plot(recall, precision, marker=".", label="Precision-Recall Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.grid(True)
    plt.show()
