import base64
import datetime
import re
import os
import shutil

import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import pandas as pd

from faker import Faker
from sklearn.metrics import precision_recall_curve
import sqlparse

rules_path = "/app/wafcraft/rules"
log_path = "/app/wafcraft/logs/log.txt"

# create logs directory directory if it does not exist
os.makedirs("/app/wafcraft/logs", exist_ok=True)


def log(message, level=1):
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(log_path, "a") as log_file:
        log_file.write(f"[{ts}] {message}\n")
    if level >= 2:
        print(message)
        if level >= 3:
            try:
                os.system(
                    f'curl -d "`hostname`: {message}" -H "Tags: hedgehog" ntfy.sh/luis-info-buysvauy12iiq -s -o /dev/null'
                )
            except Exception as e:
                print(f"Not able to notify: {e}")


def generate_codename():
    """
    Generates a codename using the faker library. Used to distinguish different data sets.

    Returns:
        str: Codename
    """
    faker = Faker()
    return f"{faker.color_name().lower()}-{faker.word().lower()}"


def generate_workspace_path():
    ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    codename = generate_codename()
    data_path = f"data/prepared/{ts}_{codename}"
    return data_path


def get_config_string(Config):
    config = ""
    for key, value in Config.__dict__.items():
        # shorten value if it is too long
        config += f"{key: >20}: {value}\n"
    return config


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


def get_most_recent_data_path():
    """
    Returns the path to the most recent prepared data

    Returns:
        str: Path to the most recent prepared data
    """
    try:
        data_path = max(  # use the most recent prepared data
            [os.path.join("data/prepared/", d) for d in os.listdir("data/prepared/")],
            key=os.path.getmtime,
        )
        return data_path
    except ValueError:
        return None


def read_and_parse_sql(file_path):
    content = open(file_path, "r").read()
    statements = sqlparse.split(content)

    parsed_data = []
    for statement in statements:
        base64_statement = base64.b64encode(statement.encode("utf-8")).decode("utf-8")
        parsed_data.append({"data": base64_statement})
    df = pd.DataFrame(parsed_data)
    df = df.drop_duplicates()
    return df


def split_in_batches(data, batch_size, data_path_tmp, label):
    data_batches = [data[i : i + batch_size] for i in range(0, len(data), batch_size)]
    for i, batch in enumerate(data_batches):
        batch.to_csv(f"{data_path_tmp}/todo/{label}_{i}.csv", index=False, header=True)


def load_and_concat_batches(directory, files):
    if not files:
        return None
    else:
        data = pd.concat([pd.read_csv(f"{directory}/{file}") for file in files])
        return data


def plot_cm(cm, path=None):
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
    # save as image
    if path:
        plt.savefig(path)
    plt.show()


def plot_roc(fpr, tpr, closest_idx, desired_fpr, path=None):
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
    if path:
        plt.savefig(path)
    plt.show()


def plot_precision_recall_curve(y_test, probabilities, path=None):
    precision, recall, thresholds = precision_recall_curve(y_test, probabilities)
    thresholds = np.append(thresholds, 1)
    plt.figure(figsize=(4, 3))
    plt.plot(recall, precision, marker=".", label="Precision-Recall Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.grid(True)
    if path:
        plt.savefig(path)
    plt.show()


def remove_tmp_dirs(workspace):
    # remove all temporary directories, okay if they do not exist
    shutil.rmtree(f"{workspace}/tmp_addvec", ignore_errors=True)
    shutil.rmtree(f"{workspace}/tmp_optimize", ignore_errors=True)
    shutil.rmtree(f"{workspace}/tmp_addvec_after_optimize", ignore_errors=True)
    shutil.rmtree(f"{workspace}/tmp_samples", ignore_errors=True)


def save_settings(Config, workspace):
    with open(f"{workspace}/config.txt", "w") as f:
        f.write(get_config_string(Config))


def save_model(workspace, model_dir, model, threshold, is_adv):
    joblib.dump(
        model, f'{workspace}/{model_dir}/model{"_adv" if is_adv else ""}.joblib'
    )
    with open(
        f'{workspace}/{model_dir}/threshold{"_adv" if is_adv else ""}.txt', "w"
    ) as f:
        f.write(str(threshold))


def determine_progress(workspace):
    has_model = os.path.exists(f"{workspace}/model/model.joblib")
    has_tmp_optimize_todos = os.path.exists(f"{workspace}/tmp_optimize/todo")
    if has_tmp_optimize_todos:
        has_tmp_optimize_todos = len(os.listdir(f"{workspace}/tmp_optimize/todo")) > 0
    has_adv_model = os.path.exists(f"{workspace}/model_adv/model_adv.joblib")
    has_tmp_samples_todos = os.path.exists(f"{workspace}/tmp_samples/todo")
    if has_tmp_samples_todos:
        has_tmp_samples_todos = len(os.listdir(f"{workspace}/tmp_samples/todo")) > 0
    has_sample_adv_csv = os.path.exists(f"{workspace}/sample_adv.csv")
    return (
        has_model,
        has_tmp_optimize_todos,
        has_adv_model,
        has_tmp_samples_todos,
        has_sample_adv_csv,
    )
