import base64
import re
import sqlparse
import glob
import pandas as pd

from modsec import get_activated_rules, init_modsec
from sklearn.model_selection import train_test_split

rules_path = "/app/ml-modsec/rules"
# TODO: handle large files
attack_data_path = "data/attacks_5k.sql"
sane_data_path = "data/sanes_5k.sql"


def get_rules_list():
    # read rules from each file in the rules directory
    for rule_path in sorted(glob.glob(f"{rules_path}/*.conf")):
        with open(rule_path, "r") as rule_file:
            rules = rule_file.read()
            matches = re.findall(r"id:(\d+),", rules)
    # return sorted list of unique rule IDs
    return sorted(set(matches))


def payload_to_vec(payload, rule_ids, modsec):
    matches = get_activated_rules([payload], modsec=modsec)
    rule_array = [1 if int(rule_id) in set(matches) else 0 for rule_id in rule_ids]

    return rule_array


def create_train_test_split(attack_file, sane_file, train_size, test_size, modsec):
    def read_and_parse(file_path):
        content = open(file_path, "r").read()
        statements = sqlparse.split(content) # TODO: slow for large files

        parsed_data = []
        for statement in statements:
            base64_statement = base64.b64encode(statement.encode("utf-8")).decode(
                "utf-8"
            )
            parsed_data.append({"data": base64_statement})

        return pd.DataFrame(parsed_data)

    def add_payload_to_vec(data, rule_ids, modsec):
        data["vector"] = data["data"].apply(
            lambda x: payload_to_vec(x, rule_ids, modsec)
        )
        return data

    rule_ids = get_rules_list()
    
    print("Reading and parsing attacks data...")
    attacks = read_and_parse(attack_file)
    attacks["label"] = "attack"

    print("Reading and parsing sanes data...")
    sanes = read_and_parse(sane_file)
    sanes["label"] = "sane"

    # Concatenate and shuffle
    full_data = pd.concat([attacks, sanes]).sample(frac=1).reset_index(drop=True)

    print("Splitting into train and test...")
    train, test = train_test_split(
        full_data,
        train_size=train_size,
        test_size=test_size,
        stratify=full_data["label"],
    )

    # Add vector for payloads in train and test
    print("Adding train vectors...")
    train = add_payload_to_vec(train, rule_ids, modsec)
    print("Adding test vectors...")
    test = add_payload_to_vec(test, rule_ids, modsec)

    return train, test


# example usage
modsec = init_modsec()
train, test = create_train_test_split(
    attack_file=attack_data_path,
    sane_file=sane_data_path,
    train_size=5,
    test_size=5,
    modsec=modsec,
)
train.to_csv("train_dataset.csv", index=False)
test.to_csv("test_dataset.csv", index=False)
