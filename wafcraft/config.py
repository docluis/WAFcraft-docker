# This file holds different Configurations

from sklearn.ensemble import RandomForestClassifier
from src.utils import get_rules_list


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
    ENGINE_SETTINGS_SAMPLE_CREATION = {
        "max_rounds": 200,
        "round_size": 10,
        "timeout": 10,
        "threshold": 0.0,  # just go as far as possible
    }
    PARANOIA_LEVEL = 4
    MODEL = RandomForestClassifier(n_estimators=160, random_state=666)
    MODEL_ADV = RandomForestClassifier(n_estimators=160, random_state=666)
    RULE_IDS = get_rules_list()
    BATCH_SIZE = 10
    MAX_PROCESSES = 8
    DESIRED_FPR = 0.01


class Target_Config:
    # General Settings
    NAME = "Target"
    DESCRIPTION = "full data set size from paper"
    BATCH_SIZE = 10
    MAX_PROCESSES = 20
    # Training Settings
    ATTACK_DATA_PATH = "data/raw/attacks_full.sql"
    SANE_DATA_PATH = "data/raw/sanes_full.sql"
    TRAIN_ATTACKS_SIZE = 10000
    TRAIN_SANES_SIZE = 10000
    TEST_ATTACKS_SIZE = 2000
    TEST_SANES_SIZE = 2000
    MODEL = RandomForestClassifier(n_estimators=160, random_state=666)
    PARANOIA_LEVEL = 4
    RULE_IDS = get_rules_list()
    DESIRED_FPR = 0.01
    # Adversarial Training Settings
    TRAIN_ADV_SIZE = 5000
    TEST_ADV_SIZE = 2000
    MODEL_ADV = RandomForestClassifier(n_estimators=160, random_state=666)
    ENGINE_SETTINGS = {  # paper unclear about these settings
        "max_rounds": 200,
        "round_size": 10,
        "timeout": 15,
        "threshold": 0.0,  # just go as far as possible
    }
    # Sample Creation Settings
    ENGINE_SETTINGS_SAMPLE_CREATION = {
        "max_rounds": 300,
        "round_size": 10,
        "timeout": 30,
        "threshold": 0.0,  # just go as far as possible
    }


class Surrogate_SVM_V1_Config(Target_Config):
    NAME = "Surrogate_SVM_V1"
    DESCRIPTION = "Surrogate Model with SVM, 100% data overlap"
    # Overlap Settings
    OVERLAP = 1.0
    OVERLAP_PATH = "/app/wafcraft/data/prepared/"  # TODO
