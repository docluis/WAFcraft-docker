# Imports
import pandas as pd  # type: ignore
import numpy as np  # type: ignore
import os
import base64
from utils import (
    get_rules_list,
    create_train_test_split,
    create_model,
    create_adv_train_test_split,
    test_evasion,
    log,
)
from modsec import init_modsec

from sklearn.ensemble import RandomForestClassifier  # type: ignore
from wafamole.evasion import EvasionEngine  # type: ignore

log("Starting precalc data")
rule_ids = get_rules_list()
modsec = init_modsec()

# Set up variables

attack_data_path = "data/raw/attacks_full.sql" # raw attack data
sane_data_path = "data/raw/sanes_full.sql" # raw sane data
processed_data_path = "data/preprocessed/3"  # path to store the preprocessed train and test data

paranoia_level = 4

train_attacks_size = 5000  # paper uses 10000
train_sanes_size = 5000  # paper uses 10000
test_attacks_size = 1000  # paper uses 2000
test_sanes_size = 1000  # paper uses 2000

train_adv_size = 2500  # paper uses 5000 (1/4 of total train set size)
test_adv_size = 1000  # paper uses 2000 (1/2 of total test set size)

engine_settings = {
    "max_rounds": 200,
    "round_size": 10,
    "timeout": 5,
}

model = RandomForestClassifier(n_estimators=160, random_state=666)
model_adv = RandomForestClassifier(n_estimators=160, random_state=666)

# Create train and test sets and train model
if not os.path.exists(processed_data_path):
    os.makedirs(processed_data_path)
train, test = create_train_test_split(
    attack_file=attack_data_path,
    sane_file=sane_data_path,
    train_attacks_size=train_attacks_size,
    train_sanes_size=train_sanes_size,
    test_attacks_size=test_attacks_size,
    test_sanes_size=test_sanes_size,
    modsec=modsec,
    rule_ids=rule_ids,
    paranoia_level=paranoia_level,
)
train.to_csv(f"{processed_data_path}/train_{train_attacks_size + train_sanes_size}.csv", index=False)
test.to_csv(f"{processed_data_path}/test_{test_attacks_size + test_sanes_size}.csv", index=False)
log("Train and test sets created", True)

wafamole_model, threshold = create_model(
    train=train,
    test=test,
    model=model,
    desired_fpr=0.01,
    modsec=modsec,
    rule_ids=rule_ids,
    paranoia_level=paranoia_level,
)

# adversarial training

engine = EvasionEngine(wafamole_model)
train_adv, test_adv = create_adv_train_test_split(
    train=train,
    test=test,
    train_adv_size=train_adv_size,
    test_adv_size=test_adv_size,
    engine=engine,
    engine_settings={
        **engine_settings,
        "threshold": threshold,
    },
    modsec=modsec,
    rule_ids=rule_ids,
    paranoia_level=paranoia_level,
)
train_adv.to_csv(f"{processed_data_path}/train_adv_{train_adv_size}.csv", index=False)
test_adv.to_csv(f"{processed_data_path}/test_adv_{test_adv_size}.csv", index=False)
log("Adversarial train and test sets created", True)

# train new model with train + train_adv
wafamole_model_adv, threshold_adv = create_model(
    train=pd.concat([train, train_adv]).sample(frac=1).reset_index(drop=True),
    test=pd.concat([test, test_adv]).sample(frac=1).reset_index(drop=True),
    model=model_adv,
    desired_fpr=0.01,
    modsec=modsec,
    rule_ids=rule_ids,
    paranoia_level=paranoia_level,
)
