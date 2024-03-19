# This file holds different Configurations

from sklearn.ensemble import RandomForestClassifier
from utils import get_rules_list


class BaseConfig:
    DESCRIPTION = "base config, small dataset"
    ATTACK_DATA_PATH = "data/raw/attacks_20k.sql"
    SANE_DATA_PATH = "data/raw/sanes_20k.sql"
    TRAIN_ATTACKS_SIZE = 200  # paper uses 10000
    TRAIN_SANES_SIZE = 200  # paper uses 10000
    TEST_ATTACKS_SIZE = 100  # paper uses 2000
    TEST_SANES_SIZE = 100  # paper uses 2000
    TRAIN_ADV_SIZE = 30  # paper uses 5000 (1/4 of total train set size)
    TEST_ADV_SIZE = 20  # paper uses 2000 (1/2 of total test set size)
    ENGINE_SETTINGS = {
        "max_rounds": 200,
        "round_size": 10,
        "timeout": 5,
    }
    PARANOIA_LEVEL = 4
    MODEL = RandomForestClassifier(n_estimators=160, random_state=666)
    MODEL_ADV = RandomForestClassifier(n_estimators=160, random_state=666)
    RULE_IDS = get_rules_list()
    BATCH_SIZE = 10
