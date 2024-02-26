import base64
import re

import numpy as np

import sqlparse
import glob
import pandas as pd

from tqdm import tqdm
from modsec import get_activated_rules, init_modsec
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_curve
from sklearn.preprocessing import LabelEncoder

from wafamole.models import Model

rules_path = "/app/ml-modsec/rules"


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


def payload_to_vec(payload_base64, rule_ids, modsec):
    """
    Returns a vectorized representation of a payload based on the activated rules

    Parameters:
        payload_base64 (str): Base64-encoded payload
        rule_ids (list): List of rule IDs
        modsec (modsecurity.ModSecurity): ModSecurity instance

    Returns:
        numpy.ndarray: Vectorized payload
    """

    matches = get_activated_rules([payload_base64], modsec=modsec)
    # rule_array as numpy array of 0s and 1s
    rule_array = [1 if int(rule_id) in set(matches) else 0 for rule_id in rule_ids]
    return np.array(rule_array)


def create_train_test_split(
    attack_file, sane_file, train_size, test_size, modsec, rule_ids
):
    """
    Returns train and test dataframes with vectorized payloads

    Parameters:
        attack_file (str): Path to file with attack payloads
        sane_file (str): Path to file with sane payloads
        train_size (float): Proportion of the dataset to include in the train split
        test_size (float): Proportion of the dataset to include in the test split
        modsec (modsecurity.ModSecurity): ModSecurity instance
        rule_ids (list): List of rule IDs

    Returns:
        pd.DataFrame, pd.DataFrame: Train and test dataframes
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

    def add_payload_to_vec(data, rule_ids, modsec):
        tqdm.pandas(desc="Processing payloads")
        data["vector"] = data["data"].progress_apply(
            lambda x: payload_to_vec(x, rule_ids, modsec)
        )
        return data

    print("Reading and parsing data...")

    attacks = read_and_parse(attack_file)
    attacks["label"] = "attack"
    sanes = read_and_parse(sane_file)
    sanes["label"] = "sane"

    # Concatenate and shuffle
    full_data = pd.concat([attacks, sanes]).sample(frac=1).reset_index(drop=True)
    print("Full data shape:", full_data.shape)

    print("Splitting into train and test...")
    train, test = train_test_split(
        full_data,
        train_size=train_size,
        test_size=test_size,
        stratify=full_data["label"],
    )

    # Add vector for payloads in train and test
    print("Creating vectors...")
    train = add_payload_to_vec(train, rule_ids, modsec)
    test = add_payload_to_vec(test, rule_ids, modsec)

    print("Done!")
    print(f"Train shape: {train.shape} | Test shape: {test.shape}")
    return train, test


def create_model(train, test, model, desired_fpr, modsec, rule_ids):
    threshold = 0.5
    # Extract features and labels
    X_train, y_train = list(train["vector"]), train["label"]
    X_test, y_test = list(test["vector"]), test["label"]

    # Train the model
    model.fit(X_train, y_train)
    print("Model trained successfully!")

    # Evaluate the model
    print("Evaluating model...")
    print(f"Default threshold: {threshold}")
    print(classification_report(y_test, model.predict(X_test)))

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

        print(f"Adjusted threshold: {threshold}")
        # make sure classification report has attack and sane in the right order
        print(classification_report(binary_y_test, adjusted_predictions, target_names=label_encoder.classes_))
        # print(classification_report(binary_y_test, adjusted_predictions))
    class WAFamoleModel(Model):
        def extract_features(self, value: str):
            payload_base64 = base64.b64encode(value.encode("utf-8")).decode("utf-8")
            return payload_to_vec(
                payload_base64=payload_base64, rule_ids=rule_ids, modsec=modsec
            )

        def classify(self, value: str):
            vec = self.extract_features(value)
            return predict_vec(
                vec=vec,
                model=model,
            )
    wafamole_model = WAFamoleModel()
    
    return wafamole_model, threshold


def predict_payload(payload_base64, model, modsec, rule_ids):
    """
    Returns the probability of a payload being an attack

    Parameters:
        payload_base64 (str): Base64-encoded payload
        model (sklearn.ensemble.RandomForestClassifier): Trained model
        modsec (modsecurity.ModSecurity): ModSecurity instance
        rule_ids (list): List of rule IDs

    Returns:
        float: Probability of payload being an attack
    """
    vec = payload_to_vec(payload_base64, rule_ids, modsec)
    probs = model.predict_proba([vec])[0]
    attack_index = list(model.classes_).index("attack")
    confidence = probs[attack_index]
    return confidence


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
