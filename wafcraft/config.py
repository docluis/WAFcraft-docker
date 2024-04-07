# This file holds different Configurations

from sklearn.ensemble import RandomForestClassifier
from src.utils import get_rules_list


class Test_Config:
    # General Settings
    NAME = "Test"
    DESCRIPTION = "small quick test"
    BATCH_SIZE = 10
    MAX_PROCESSES = 20
    # Training Settings
    ATTACK_DATA_PATH = "data/raw/attacks_20k.sql"
    SANE_DATA_PATH = "data/raw/sanes_20k.sql"
    TRAIN_ATTACKS_SIZE = 500
    TRAIN_SANES_SIZE = 500
    TEST_ATTACKS_SIZE = 250
    TEST_SANES_SIZE = 250
    OVERLAP_SETTINGS = {
        "use_overlap": False,
        "overlap": None,
        "overlap_path": None,
    }
    MODEL = RandomForestClassifier(n_estimators=160, random_state=666)
    PARANOIA_LEVEL = 4
    RULE_IDS = get_rules_list()
    DESIRED_FPR = 0.01
    # Adversarial Training Settings
    ADVERSARIAL_TRAINING = True
    TRAIN_ADV_SIZE = 50
    TEST_ADV_SIZE = 20
    MODEL_ADV = RandomForestClassifier(n_estimators=160, random_state=666)
    ENGINE_SETTINGS = {  # paper unclear about these settings
        "max_rounds": 200,
        "round_size": 10,
        "timeout": 5,
        "threshold": 0.0,  # just go as far as possible
    }
    # Sample Creation Settings
    FIND_SAMPLES = True
    SAMPLE_ATTEMPTS = 100
    ENGINE_SETTINGS_SAMPLE_CREATION = {
        "max_rounds": 300,
        "round_size": 10,
        "timeout": 15,
        "threshold": 0.0,
    }


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
    OVERLAP_SETTINGS = {
        "use_overlap": False,
        "overlap": None,
        "overlap_path": None,
    }
    # Adversarial Training Settings
    ADVERSARIAL_TRAINING = True
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
    FIND_SAMPLES = False
    SAMPLE_ATTEMPTS = 10000
    ENGINE_SETTINGS_SAMPLE_CREATION = {
        "max_rounds": 300,
        "round_size": 10,
        "timeout": 30,
        "threshold": 0.0,
    }


class Surrogate_SVM_V1_Config(Target_Config):
    NAME = "Surrogate_SVM_V1"
