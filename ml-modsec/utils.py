import base64
import re

import numpy as np

import sqlparse
import glob
import pandas as pd

from tqdm import tqdm
from modsec import get_activated_rules, init_modsec
from sklearn.model_selection import train_test_split

rules_path = "/app/ml-modsec/rules"


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
    matches = get_activated_rules([payload_base64], modsec=modsec)
    # rule_array as numpy array of 0s and 1s
    rule_array = [1 if int(rule_id) in set(matches) else 0 for rule_id in rule_ids]
    return np.array(rule_array)


def create_train_test_split(
    attack_file, sane_file, train_size, test_size, modsec, rule_ids
):
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
    attack_index = list(model.classes_).index('attack')
    confidence = probs[attack_index]
    return confidence

def predict_vec(vec, model, modsec, rule_ids):
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
    attack_index = list(model.classes_).index('attack')
    confidence = probs[attack_index]
    return confidence