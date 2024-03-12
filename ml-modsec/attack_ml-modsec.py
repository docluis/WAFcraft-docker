# Imports
import pandas as pd  # type: ignore
import numpy as np  # type: ignore
import base64
from utils import (
    get_rules_list,
    create_train_test_split,
    create_model,
    payload_to_vec,
    create_adv_train_test_split,
    test_evasion,
    notify,
)
from modsec import init_modsec

from sklearn.ensemble import RandomForestClassifier  # type: ignore
from wafamole.evasion import EvasionEngine  # type: ignore

# Set up variables

attack_data_path = "data/attacks_full.sql"
sane_data_path = "data/sanes_full.sql"

rule_ids = get_rules_list()
modsec = init_modsec()

# Create train and test sets and train model

paranoia_level = 4

# train, test = create_train_test_split(
#     attack_file=attack_data_path,
#     sane_file=sane_data_path,
#     train_attacks_size=5000,  # paper uses 10000
#     train_sanes_size=5000,  # paper uses 10000
#     test_attacks_size=1000,  # paper uses 2000
#     test_sanes_size=1000,  # paper uses 2000
#     modsec=modsec,
#     rule_ids=rule_ids,
#     paranoia_level=paranoia_level,
# )
# train.to_csv("data/train_10k.csv", index=False)
# test.to_csv("data/test_2k.csv", index=False)
# notify("Train and test sets created")

# # load the train and test sets from disk
train = pd.read_csv("data/train_10k.csv")
test = pd.read_csv("data/test_2k.csv")
train['vector'] = train['vector'].apply(lambda x: np.fromstring(x[1:-1], sep=' '))
test['vector'] = test['vector'].apply(lambda x: np.fromstring(x[1:-1], sep=' '))

wafamole_model, threshold = create_model(
    train=train,
    test=test,
    model=RandomForestClassifier(n_estimators=160, random_state=666),
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
    train_adv_size=2500, # paper uses 5000 (1/4 of total train set size)
    test_adv_size=1000, # paper uses 2000 (1/2 of total test set size)
    engine=engine,
    engine_settings={
        "max_rounds": 200,
        "round_size": 20,
        "timeout": 10,
        "threshold": threshold,
    },
    modsec=modsec,
    rule_ids=rule_ids,
    paranoia_level=paranoia_level,
)
train_adv.to_csv("data/train_adv_500.csv", index=False)
test_adv.to_csv("data/test_adv_200.csv", index=False)
notify("Adversarial train and test sets created")

# # load the train_adv and test_adv sets from disk
# train_adv = pd.read_csv("data/train_adv_50.csv")
# test_adv = pd.read_csv("data/test_adv_50.csv")
# train_adv['vector'] = train_adv['vector'].apply(lambda x: np.fromstring(x[1:-1], sep=' '))
# test_adv['vector'] = test_adv['vector'].apply(lambda x: np.fromstring(x[1:-1], sep=' '))

# train new model with train + train_adv
wafamole_model_adv, threshold_adv = create_model(
    train=pd.concat([train, train_adv]).sample(frac=1).reset_index(drop=True),
    test=pd.concat([test, test_adv]).sample(frac=1).reset_index(drop=True),
    model=RandomForestClassifier(n_estimators=160, random_state=666),
    desired_fpr=0.01,
    modsec=modsec,
    rule_ids=rule_ids,
    paranoia_level=paranoia_level,
)


# Test the model (without adversarial training)
test_evasion(
    payload='SELECT SLEEP(5)#";',
    threshold=threshold,
    model=wafamole_model,
    engine=EvasionEngine(wafamole_model),
    engine_eval_settings={
        "max_rounds": 200,
        "round_size": 10,
        "timeout": 60,
        "threshold": 0.0,
    },
    modsec=modsec,
    rule_ids=rule_ids,
    paranoia_level=paranoia_level,
)


# Test the model (with adversarial training)
test_evasion(
    payload='SELECT SLEEP(5)#";',
    threshold=threshold,
    model=wafamole_model_adv,
    engine=EvasionEngine(wafamole_model_adv),
    engine_eval_settings={
        "max_rounds": 200,
        "round_size": 10,
        "timeout": 60,
        "threshold": 0.0,
    },
    modsec=modsec,
    rule_ids=rule_ids,
    paranoia_level=paranoia_level,
)