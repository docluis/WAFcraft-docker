import base64
import re
import contextlib
import io
import os

import numpy as np

import sqlparse
import glob
import pandas as pd

from tqdm import tqdm
from modsec import get_activated_rules
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_curve
from sklearn.preprocessing import LabelEncoder

from wafamole.models import Model  # type: ignore

rules_path = "/app/ml-modsec/rules"
log_path = "log.txt"

f = io.StringIO()


def log(message, notify=False):
    print(message)
    time = pd.Timestamp.now()
    with open(log_path, "a") as log_file:
        log_file.write(f"{time}: {message}\n")
    if notify:
        os.system(
            f'curl -d "{message}" -H "Tags: hedgehog" ntfy.sh/luis-info-buysvauy12iiq -s -o /dev/null'
        )


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
    modsec,
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
        modsec (modsecurity.ModSecurity): ModSecurity instance
        rule_ids (list): List of rule IDs
        paranoia_level (int): Paranoia level
    """

    def read_and_parse(file_path):
        content = open(file_path, "r").read()
        statements = sqlparse.split(content)  # TODO: slow for large files

        parsed_data = []
        for statement in statements:
            base64_statement = base64.b64encode(statement.encode("utf-8")).decode(
                "utf-8"
            )
            parsed_data.append({"data": base64_statement})

        return pd.DataFrame(parsed_data)

    log("Reading and parsing data...")

    attacks = read_and_parse(attack_file)
    attacks["label"] = "attack"
    sanes = read_and_parse(sane_file)
    sanes["label"] = "sane"

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
    train = add_vec(train, rule_ids, modsec, paranoia_level)
    test = add_vec(test, rule_ids, modsec, paranoia_level)

    log("Done!")
    log(f"Train shape: {train.shape} | Test shape: {test.shape}")
    return train, test


def create_model(train, test, model, desired_fpr, modsec, rule_ids, paranoia_level):
    """
    Returns a trained model and the threshold for the desired FPR

    Parameters:
        train (pd.DataFrame): Train dataframe
        test (pd.DataFrame): Test dataframe
        model (sklearn.ensemble.RandomForestClassifier): Model to train
        desired_fpr (float): Desired false positive rate
        modsec (modsecurity.ModSecurity): ModSecurity instance
        rule_ids (list): List of rule IDs
        paranoia_level (int): Paranoia level

    Returns:
        wafamole.models.Model, float: Trained model and threshold for the desired FPR
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
    log(classification_report(y_test, model.predict(X_test)))

    if desired_fpr is not None:
        # 'attack' is considered the positive class (1) and 'sane' is the negative class (0)
        # Adjust the model prediction threshold based on the desired FPR
        label_encoder = LabelEncoder()
        binary_y_test = label_encoder.fit_transform(y_test)
        probabilities = model.predict_proba(X_test)[:, 1]
        fpr, tpr, thresholds = roc_curve(binary_y_test, probabilities)
        closest_idx = np.argmin(np.abs(fpr - desired_fpr))
        threshold = thresholds[closest_idx]
        adjusted_predictions = (probabilities >= threshold).astype(int)

        log(f"Adjusted threshold: {threshold}")
        # make sure classification report has attack and sane in the right order
        log(
            classification_report(
                binary_y_test, adjusted_predictions, target_names=label_encoder.classes_
            )
        )
        # log(classification_report(binary_y_test, adjusted_predictions))

    def predict_vec(vec, model):
        """
        Returns the probability of a payload being an attack

        Parameters:
            vec (numpy.ndarray): Vectorized payload
            model (sklearn.ensemble.RandomForestClassifier): Trained model
            modsec (modsecurity.ModSecurity): ModSecurity instance
            rule_ids (list): List of rule IDs

        Returns:
            float: Probability of payload being an attack
        """
        probs = model.predict_proba([vec])[0]
        attack_index = list(model.classes_).index("attack")
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

    return wafamole_model, threshold


def create_adv_train_test_split(
    train,
    test,
    train_adv_size,
    test_adv_size,
    engine,
    engine_settings,
    modsec,
    rule_ids,
    paranoia_level,
):
    """
    Returns train and test dataframes with adversarial payloads

    Parameters:
        train (pd.DataFrame): Train dataframe
        test (pd.DataFrame): Test dataframe
        train_adv_size (float): Number of adversarial payloads to generate for training
        test_adv_size (float): Number of adversarial payloads to generate for testing
        engine (wafamole.models.Model): Model to generate adversarial payloads
        engine_settings (dict): Settings for the model
        modsec (modsecurity.ModSecurity): ModSecurity instance
        rule_ids (list): List of rule IDs
        paranoia_level (int): Paranoia level

    Returns:
        pd.DataFrame, pd.DataFrame: Train and test dataframes with adversarial payloads
    """

    def optimize(data_set, data_set_size):
        for i, row in tqdm(data_set.iterrows(), total=data_set_size):
            with contextlib.redirect_stdout(f):
                try:
                    min_confidence, min_payload = engine.evaluate(
                        payload=base64.b64decode(row["data"]).decode("utf-8"),
                        **engine_settings,
                    )
                    data_set.at[i, "data"] = base64.b64encode(
                        min_payload.encode("utf-8")
                    ).decode("utf-8")
                except Exception as e:
                    log(
                        f"Error: {e}, dropping row {i} payload {base64.b64decode(row['data'])}"
                    )
                    data_set.drop(i, inplace=True)
                    continue

    # Sample train and test dataframes, only use attack payloads
    train_adv = (
        train[train["label"] == "attack"]
        .sample(n=train_adv_size)
        .drop(columns=["vector"])
    )
    test_adv = (
        test[test["label"] == "attack"].sample(n=test_adv_size).drop(columns=["vector"])
    )

    log("Optimizing payloads...")
    optimize(train_adv, train_adv_size)
    optimize(test_adv, test_adv_size)
    log(f"Train_adv shape: {train_adv.shape} | Test_adv shape: {test_adv.shape}")

    # Add vector for payloads in train and test
    log("Creating vectors...")
    train_adv = add_vec(train_adv, rule_ids, modsec, paranoia_level)
    test_adv = add_vec(test_adv, rule_ids, modsec, paranoia_level)

    return train_adv, test_adv


def test_evasion(
    payload,
    threshold,
    engine_eval_settings,
    model,
    engine,
    rule_ids,
    modsec,
    paranoia_level,
):
    payload_base64 = base64.b64encode(payload.encode("utf-8")).decode("utf-8")
    vec = payload_to_vec(payload_base64, rule_ids, modsec, paranoia_level)
    is_attack = model.classify(payload)
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
