import base64
import re
import contextlib
import io
import os
import time

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sqlparse
import glob
import pandas as pd

from tqdm import tqdm
from modsec import get_activated_rules, init_modsec
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    roc_curve,
    confusion_matrix,
    precision_recall_curve,
)
from sklearn.preprocessing import LabelEncoder

from concurrent.futures import ProcessPoolExecutor

from wafamole.models import Model  # type: ignore
from wafamole.evasion import EvasionEngine  # type: ignore

rules_path = "/app/wafcraft/rules"
log_path = "/app/wafcraft/logs/log.txt"

f = io.StringIO()


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


def payload_to_vec(payload_base64, rule_ids, modsec, paranoia_level):
    """
    Returns a vectorized representation of a payload based on the activated rules

    Parameters:
        payload_base64 (str): Base64-encoded payload
        rule_ids (list): List of rule IDs
        modsec (modsecurity.ModSecurity): ModSecurity instance
        paranoia_level (int): Paranoia level

    Returns:
        numpy.ndarray: Vectorized payload
    """

    matches = get_activated_rules(
        payloads_base64=[payload_base64], modsec=modsec, paranoia_level=paranoia_level
    )
    # rule_array as numpy array of 0s and 1s
    rule_array = [1 if int(rule_id) in set(matches) else 0 for rule_id in rule_ids]
    return np.array(rule_array)


def add_vec(data, rule_ids, modsec, paranoia_level):
    """
    Returns a dataframe with vectorized payloads

    Parameters:
        data (pd.DataFrame): Dataframe containing payloads
        rule_ids (list): List of rule IDs
        modsec (modsecurity.ModSecurity): ModSecurity instance
        paranoia_level (int): Paranoia level

    Returns:
        pd.DataFrame: Dataframe with vectorized payloads
    """
    tqdm.pandas(desc="Processing payloads")
    data["vector"] = data["data"].progress_apply(
        lambda x: payload_to_vec(x, rule_ids, modsec, paranoia_level)
    )
    return data


def create_train_test_split(
    attack_file,
    sane_file,
    train_attacks_size,
    train_sanes_size,
    test_attacks_size,
    test_sanes_size,
    rule_ids,
    paranoia_level,
):
    """
    Returns train and test dataframes with vectorized payloads

    Parameters:
        attack_file (str): Path to the file containing attack payloads
        sane_file (str): Path to the file containing sane payloads
        train_attacks_size (float): Number of attack payloads to use for training
        train_sanes_size (float): Number of sane payloads to use for training
        test_attacks_size (float): Number of attack payloads to use for testing
        test_sanes_size (float): Number of sane payloads to use for testing
        rule_ids (list): List of rule IDs
        paranoia_level (int): Paranoia level
    """

    def read_and_parse(file_path):
        content = open(file_path, "r").read()
        statements = sqlparse.split(content)

        parsed_data = []
        for statement in statements:
            base64_statement = base64.b64encode(statement.encode("utf-8")).decode(
                "utf-8"
            )
            parsed_data.append({"data": base64_statement})

        return pd.DataFrame(parsed_data)

    log("Reading and parsing data...")

    attacks = read_and_parse(attack_file)
    attacks["label"] = 1
    sanes = read_and_parse(sane_file)
    sanes["label"] = 0

    log("Splitting into train and test...")
    train_attacks, test_attacks = train_test_split(
        attacks,
        train_size=train_attacks_size,
        test_size=test_attacks_size,
        stratify=attacks["label"],
    )

    train_sanes, test_sanes = train_test_split(
        sanes,
        train_size=train_sanes_size,
        test_size=test_sanes_size,
        stratify=sanes["label"],
    )

    # Concatenate and shuffle
    train = (
        pd.concat([train_attacks, train_sanes]).sample(frac=1).reset_index(drop=True)
    )
    test = pd.concat([test_attacks, test_sanes]).sample(frac=1).reset_index(drop=True)

    # Add vector for payloads in train and test
    log("Creating vectors...")
    modsec = init_modsec()
    train = add_vec(train, rule_ids, modsec, paranoia_level)
    test = add_vec(test, rule_ids, modsec, paranoia_level)

    log("Done!")
    log(f"Train shape: {train.shape} | Test shape: {test.shape}")
    return train, test


def train_model(train, test, model, desired_fpr):
    """
    Returns a trained model and the threshold for the desired FPR

    Parameters:
        train (pd.DataFrame): Train dataframe
        test (pd.DataFrame): Test dataframe
        model: Model to train
        desired_fpr (float): Desired false positive rate

    Returns:
        model: Trained Model, float: Trained model and threshold
    """

    threshold = 0.5
    # Extract features and labels
    X_train, y_train = list(train["vector"]), train["label"]
    X_test, y_test = list(test["vector"]), test["label"]

    # Train the model
    model.fit(X_train, y_train)
    log("Model trained successfully!")

    # Evaluate the model
    log("Evaluating model...")
    log(f"Default threshold: {threshold}")
    # calculate FPR = FP / (FP + TN)
    current_fpr = confusion_matrix(y_test, model.predict(X_test))[0, 1] / (
        confusion_matrix(y_test, model.predict(X_test))[0, 1]
        + confusion_matrix(y_test, model.predict(X_test))[1, 1]
    )

    log(f"FRP is currently at {round(current_fpr, 4)}")
    predictions = model.predict(X_test)
    log(classification_report(y_test, predictions))

    cm = confusion_matrix(y_test, predictions)
    plot_cm(cm)

    if desired_fpr is not None:
        log(f"Adjusting threshold to match desired FPR of {desired_fpr}")
        # 'attack' is considered the positive class (1) and 'sane' is the negative class (0)
        probabilities = model.predict_proba(X_test)[:, 1]

        fpr, tpr, thresholds = roc_curve(y_test, probabilities)  # plot ROC curve
        closest_idx = np.argmin(np.abs(fpr - desired_fpr))  # threshold closest to FPR
        threshold = thresholds[closest_idx]
        adjusted_predictions = (probabilities >= threshold).astype(int)  #  new preds

        plot_roc(fpr, tpr, closest_idx, desired_fpr)
        plot_precision_recall_curve(y_test, probabilities)

        log(
            f"Adjusted threshold: {round(threshold, 4)} with FPR of {round(fpr[closest_idx], 4)} (closest to desired FPR {desired_fpr})"
        )
        log(classification_report(y_test, adjusted_predictions))

        cm = confusion_matrix(y_test, adjusted_predictions)
        plot_cm(cm)

    return model, threshold


def create_wafamole_model(
    model,
    modsec,
    rule_ids,
    paranoia_level,
):
    """
    Returns a WAFamole model

    Parameters:
        model (sklearn.ensemble.RandomForestClassifier): Trained model
        modsec (modsecurity.ModSecurity): ModSecurity instance
        rule_ids (list): List of rule IDs
        paranoia_level (int): Paranoia level

    Returns:
        wafamole.models.Model: WAFamole model
    """

    def predict_vec(vec, model):
        probs = model.predict_proba([vec])[0]
        attack_index = list(model.classes_).index(1)
        confidence = probs[attack_index]
        return confidence

    class WAFamoleModel(Model):
        def extract_features(self, value: str):
            payload_base64 = base64.b64encode(value.encode("utf-8")).decode("utf-8")
            return payload_to_vec(
                payload_base64=payload_base64,
                rule_ids=rule_ids,
                modsec=modsec,
                paranoia_level=paranoia_level,
            )

        def classify(self, value: str):
            vec = self.extract_features(value)
            return predict_vec(
                vec=vec,
                model=model,
            )

    wafamole_model = WAFamoleModel()
    return wafamole_model


def optimize(
    data_path,
    batch_number,
    label,
    model_trained,
    engine_settings,
    rule_ids,
    paranoia_level,
    tmp_path,
):
    modsec = init_modsec()
    wafamole_model = create_wafamole_model(
        model_trained, modsec, rule_ids, paranoia_level
    )
    data_set = pd.read_csv(data_path)
    engine = EvasionEngine(wafamole_model)

    for i, row in tqdm(data_set.iterrows(), total=len(data_set)):
        try:
            with contextlib.redirect_stdout(f):
                min_confidence, min_payload = engine.evaluate(
                    payload=base64.b64decode(row["data"]).decode("utf-8"),
                    **engine_settings,
                )
            data_set.at[i, "data"] = base64.b64encode(
                min_payload.encode("utf-8")
            ).decode("utf-8")
        except Exception as e:
            if not min_payload == None:
                data_set.at[i, "data"] = base64.b64encode(
                    min_payload.encode("utf-8")
                ).decode("utf-8")
            log(f"Error: {e}")
            continue
    data_set.to_csv(
        f"{tmp_path}/optimized/{label}_adv_{batch_number}.csv",
        mode="a",
        index=False,
        header=False,
    )


def create_adv_train_test_split(
    train,
    test,
    train_adv_size,
    test_adv_size,
    model_trained,
    engine_settings,
    rule_ids,
    paranoia_level,
    batch_size,
    tmp_path,
):
    """
    Returns train and test dataframes with adversarial payloads

    Parameters:
        train (pd.DataFrame): Train dataframe
        test (pd.DataFrame): Test dataframe
        train_adv_size (float): Number of adversarial payloads to generate for training
        test_adv_size (float): Number of adversarial payloads to generate for testing
        model_trained (sklearn.ensemble.RandomForestClassifier): Trained model
        engine_settings (dict): Settings for the model
        rule_ids (list): List of rule IDs
        paranoia_level (int): Paranoia level
        batch_size (int): Batch size for optimization

    Returns:
        pd.DataFrame, pd.DataFrame: Train and test dataframes with adversarial payloads
    """
    # create directories
    os.makedirs(f"{tmp_path}/todo", exist_ok=True)
    os.makedirs(f"{tmp_path}/optimized", exist_ok=True)

    # Sample train and test dataframes, only use attack payloads
    train_adv = (
        train[train["label"] == 1].sample(n=train_adv_size).drop(columns=["vector"])
    )
    test_adv = test[test["label"] == 1].sample(n=test_adv_size).drop(columns=["vector"])

    train_adv_batches = [
        train_adv[i : i + batch_size] for i in range(0, len(train_adv), batch_size)
    ]
    test_adv_batches = [
        test_adv[i : i + batch_size] for i in range(0, len(test_adv), batch_size)
    ]

    # Save each batch to a csv
    for i, batch in enumerate(train_adv_batches):
        batch.to_csv(f"{tmp_path}/todo/train_adv_{i}.csv", index=False, header=True)
    for i, batch in enumerate(test_adv_batches):
        batch.to_csv(f"{tmp_path}/todo/test_adv_{i}.csv", index=False, header=True)

    # Optimize each batch with subproceesses

    # optimize is prone to TimeoutError, so use multiprocessing

    import multiprocessing

    with multiprocessing.Pool() as pool:
        pool.starmap(
            optimize,
            [
                (
                    f"{tmp_path}/todo/train_adv_{i}.csv",
                    i,
                    "train",
                    model_trained,
                    engine_settings,
                    rule_ids,
                    paranoia_level,
                    tmp_path,
                )
                for i in range(len(train_adv_batches))
            ],
        )
        pool.starmap(
            optimize,
            [
                (
                    f"{tmp_path}/todo/test_adv_{i}.csv",
                    i,
                    "test",
                    model_trained,
                    engine_settings,
                    rule_ids,
                    paranoia_level,
                    tmp_path,
                )
                for i in range(len(test_adv_batches))
            ],
        )

    # Read and concatenate optimized batches (keep in mind that there are no names for the columns)
    train_adv = pd.concat(
        [
            pd.read_csv(
                f"{tmp_path}/optimized/train_adv_{i}.csv",
                names=["data", "label"],
                header=None,
            )
            for i in range(len(train_adv_batches))
        ]
    )
    test_adv = pd.concat(
        [
            pd.read_csv(
                f"{tmp_path}/optimized/test_adv_{i}.csv",
                names=["data", "label"],
                header=None,
            )
            for i in range(len(test_adv_batches))
        ]
    )

    log(f"Train_adv shape: {train_adv.shape} | Test_adv shape: {test_adv.shape}")

    # Add vector for payloads in train and test
    log("Creating vectors...")
    modsec = init_modsec()
    train_adv = add_vec(train_adv, rule_ids, modsec, paranoia_level)
    test_adv = add_vec(test_adv, rule_ids, modsec, paranoia_level)

    return train_adv, test_adv


def test_evasion(
    payload,
    threshold,
    engine_eval_settings,
    model,
    rule_ids,
    modsec,
    paranoia_level,
):
    wafamole_model = create_wafamole_model(model, modsec, rule_ids, paranoia_level)
    engine = EvasionEngine(wafamole_model)
    payload_base64 = base64.b64encode(payload.encode("utf-8")).decode("utf-8")
    vec = payload_to_vec(payload_base64, rule_ids, modsec, paranoia_level)
    is_attack = wafamole_model.classify(payload)
    log(f"Payload: {payload}")
    log(f"Vec: {vec}")
    log(f"Confidence: {round(is_attack, 5)}")

    min_confidence, min_payload = engine.evaluate(
        payload=payload,
        **engine_eval_settings,
    )
    log(f"Min payload: {min_payload.encode('utf-8')}")
    log(f"Min confidence: {round(min_confidence, 5)}")
    log(
        f"Reduced confidence from {round(is_attack, 5)} to {round(min_confidence, 5)} (reduction of {round(is_attack - min_confidence, 5)})"
    )

    log("\nEvasion successful" if min_confidence < threshold else "Evasion failed")


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
