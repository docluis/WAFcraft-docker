# This file holds different Configurations

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from src.utils import get_rules_list


# ----------------- Test Configurations -----------------


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
    FIND_SAMPLES = False
    SAMPLE_ATTEMPTS = 100
    ENGINE_SETTINGS_SAMPLE_CREATION = {
        "max_rounds": 300,
        "round_size": 10,
        "timeout": 15,
        "threshold": 0.0,
    }


class Test_Surrogate_Overlap_V1_Config(Test_Config):
    # General Settings
    NAME = "Test_Surrogate_Overlap_V1"
    DESCRIPTION = "small quick test with surrogate model 0% overlap"
    # Training Settings
    OVERLAP_SETTINGS = {
        "use_overlap": True,
        "overlap": 0,
        "overlap_path": "/app/wafcraft/data/prepared/2024-04-07_18-15-53_brown-lot",
    }
    # Adversarial Training Settings
    # Sample Creation Settings
    FIND_SAMPLES = True

class Test_Surrogate_SVM_V1_Config(Test_Config):
    # General Settings
    NAME = "Test_Surrogate_SVM_V1"
    DESCRIPTION = "surrogate model with SVM and 100% overlap"
    # Training Settings
    MODEL = SVC(random_state=666, probability=True)
    OVERLAP_SETTINGS = {
        "use_overlap": False,
        "overlap": 1,
        "overlap_path": "/app/wafcraft/data/prepared/2024-04-07_18-15-53_brown-lot",
    }
    # Adversarial Training Settings
    MODEL_ADV = SVC(random_state=666, probability=True)
    # Sample Creation Settings
    FIND_SAMPLES = True


# ----------------- Target Configuration -----------------


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

class NoAdv_Target_Config:
    # General Settings
    NAME = "NoAdv_Target"
    DESCRIPTION = "full data set size from paper, no adversarial training"
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
    ADVERSARIAL_TRAINING = False
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
    SAMPLE_ATTEMPTS = 5000
    ENGINE_SETTINGS_SAMPLE_CREATION = {
        "max_rounds": 300,
        "round_size": 10,
        "timeout": 30,
        "threshold": 0.0,
    }

class Large_Target_Config:
    # General Settings
    NAME = "Large_Target"
    DESCRIPTION = "twice the dimensions from the paper"
    BATCH_SIZE = 10
    MAX_PROCESSES = 20
    # Training Settings
    ATTACK_DATA_PATH = "data/raw/attacks_full.sql"
    SANE_DATA_PATH = "data/raw/sanes_full.sql"
    TRAIN_ATTACKS_SIZE = 20000
    TRAIN_SANES_SIZE = 20000
    TEST_ATTACKS_SIZE = 4000
    TEST_SANES_SIZE = 4000
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
    TRAIN_ADV_SIZE = 10000
    TEST_ADV_SIZE = 4000
    MODEL_ADV = RandomForestClassifier(n_estimators=160, random_state=666)
    ENGINE_SETTINGS = {  # paper unclear about these settings
        "max_rounds": 200,
        "round_size": 10,
        "timeout": 15,
        "threshold": 0.0,  # just go as far as possible
    }
    # Sample Creation Settings
    FIND_SAMPLES = False
    SAMPLE_ATTEMPTS = 20000
    ENGINE_SETTINGS_SAMPLE_CREATION = {
        "max_rounds": 300,
        "round_size": 10,
        "timeout": 30,
        "threshold": 0.0,
    }


# ----------------- Surrogate Configuration -----------------


class Surrogate_SVM_V1_Config(Target_Config):
    # General Settings
    NAME = "Surrogate_SVM_V1"
    DESCRIPTION = "surrogate model with SVM and 100% overlap"
    # Training Settings
    MODEL = SVC(random_state=666, probability=True)
    OVERLAP_SETTINGS = {
        "use_overlap": True,
        "overlap": 1,
        "overlap_path": "/app/wafcraft/data/prepared/2024-04-07_18-15-53_brown-lot",
    }
    # Adversarial Training Settings
    MODEL_ADV = SVC(random_state=666, probability=True)
    # Sample Creation Settings
    FIND_SAMPLES = True

class Large_Surrogate_SVM_V1_Config(Large_Target_Config):
    # General Settings
    NAME = "Large_Surrogate_SVM_V1"
    DESCRIPTION = "large surrogate model with SVM and 100% overlap"
    # Training Settings
    MODEL = SVC(random_state=666, probability=True)
    OVERLAP_SETTINGS = {
        "use_overlap": True,
        "overlap": 1,
        "overlap_path": "/app/wafcraft/data/prepared/2024-04-15_10-04-17_turquoise-community",
    }
    # Adversarial Training Settings
    MODEL_ADV = SVC(random_state=666, probability=True)
    # Sample Creation Settings
    FIND_SAMPLES = True

class Surrogate_SVM_V2_Config(Target_Config):
    # General Settings
    NAME = "Surrogate_SVM_V2"
    DESCRIPTION = "surrogate model with SVM and 0% overlap"
    # Training Settings
    MODEL = SVC(random_state=666, probability=True)
    OVERLAP_SETTINGS = {
        "use_overlap": True,
        "overlap": 0,
        "overlap_path": "/app/wafcraft/data/prepared/2024-04-07_18-15-53_brown-lot",
    }
    # Adversarial Training Settings
    MODEL_ADV = SVC(random_state=666, probability=True)
    # Sample Creation Settings
    FIND_SAMPLES = True

class Surrogate_GBoost_V1_Config(Target_Config):
    # General Settings
    NAME = "Surrogate_GBoost_V1"
    DESCRIPTION = "surrogate model with GradientBoosting and 100% overlap"
    # Training Settings
    MODEL = GradientBoostingClassifier(n_estimators=160, random_state=666)
    OVERLAP_SETTINGS = {
        "use_overlap": True,
        "overlap": 1,
        "overlap_path": "/app/wafcraft/data/prepared/2024-04-07_18-15-53_brown-lot",
    }
    # Adversarial Training Settings
    MODEL_ADV = GradientBoostingClassifier(n_estimators=160, random_state=666)
    # Sample Creation Settings
    FIND_SAMPLES = True

class Surrogate_NaiveBayes_V1_Config(Target_Config):
    # General Settings
    NAME = "Surrogate_NaiveBayes_V1"
    DESCRIPTION = "surrogate model with Naive Bayes and 100% overlap"
    # Training Settings
    MODEL = GaussianNB()
    OVERLAP_SETTINGS = {
        "use_overlap": True,
        "overlap": 1,
        "overlap_path": "/app/wafcraft/data/prepared/2024-04-07_18-15-53_brown-lot",
    }
    # Adversarial Training Settings
    MODEL_ADV = GaussianNB()
    # Sample Creation Settings
    FIND_SAMPLES = True

class Surrogate_LogReg_V1_Config(Target_Config):
    # General Settings
    NAME = "Surrogate_LogReg_V1"
    DESCRIPTION = "surrogate model with Logistic Regression and 100% overlap"
    # Training Settings
    MODEL = LogisticRegression(random_state=666)
    OVERLAP_SETTINGS = {
        "use_overlap": True,
        "overlap": 1,
        "overlap_path": "/app/wafcraft/data/prepared/2024-04-07_18-15-53_brown-lot",
    }
    # Adversarial Training Settings
    MODEL_ADV = LogisticRegression(random_state=666)
    # Sample Creation Settings
    FIND_SAMPLES = True

class Surrogate_KNN_V1_Config(Target_Config):
    # General Settings
    NAME = "Surrogate_KNN_V1"
    DESCRIPTION = "surrogate model with K-Nearest Neighbors and 100% overlap"
    # Training Settings
    MODEL = KNeighborsClassifier()
    OVERLAP_SETTINGS = {
        "use_overlap": True,
        "overlap": 1,
        "overlap_path": "/app/wafcraft/data/prepared/2024-04-07_18-15-53_brown-lot",
    }
    # Adversarial Training Settings
    MODEL_ADV = KNeighborsClassifier()
    # Sample Creation Settings
    FIND_SAMPLES = True

class Surrogate_Data_V1_Config(Target_Config):
    # General Settings
    NAME = "Surrogate_Data_V1"
    DESCRIPTION = "surrogate model with 0% train, test overlap"
    # Training Settings
    OVERLAP_SETTINGS = {
        "use_overlap": True,
        "overlap": 0,
        "overlap_path": "/app/wafcraft/data/prepared/2024-04-07_18-15-53_brown-lot",
    }
    # Adversarial Training Settings
    # Sample Creation Settings
    FIND_SAMPLES = True

class Large_Surrogate_Data_V1_Config(Large_Target_Config):
    # General Settings
    NAME = "Large_Surrogate_Data_V1"
    DESCRIPTION = "large surrogate model with 0% train, test overlap"
    # Training Settings
    OVERLAP_SETTINGS = {
        "use_overlap": True,
        "overlap": 0,
        "overlap_path": "/app/wafcraft/data/prepared/2024-04-15_10-04-17_turquoise-community",
    }
    # Adversarial Training Settings
    # Sample Creation Settings
    FIND_SAMPLES = True

class NoAdv_Surrogate_Data_V1_Config(NoAdv_Target_Config):
    # General Settings
    NAME = "NoAdv_Surrogate_Data_V1"
    DESCRIPTION = "surrogate model with 0% train, test overlap"
    # Training Settings
    OVERLAP_SETTINGS = {
        "use_overlap": True,
        "overlap": 0,
        "overlap_path": "/app/wafcraft/data/prepared/2024-05-06_11-25-19_honeydew-tough",
    }
    # Adversarial Training Settings
    # Sample Creation Settings
    FIND_SAMPLES = True
class Surrogate_Data_V2_Config(Target_Config):
    # General Settings
    NAME = "Surrogate_Data_V2"
    DESCRIPTION = "surrogate model with 25% train, test overlap"
    # Training Settings
    OVERLAP_SETTINGS = {
        "use_overlap": True,
        "overlap": 0.25,
        "overlap_path": "/app/wafcraft/data/prepared/2024-04-07_18-15-53_brown-lot",
    }
    # Adversarial Training Settings
    # Sample Creation Settings
    FIND_SAMPLES = True

class NoAdv_Surrogate_Data_V2_Config(NoAdv_Target_Config):
    # General Settings
    NAME = "NoAdv_Surrogate_Data_V2"
    DESCRIPTION = "surrogate model with 25% train, test overlap"
    # Training Settings
    OVERLAP_SETTINGS = {
        "use_overlap": True,
        "overlap": 0.25,
        "overlap_path": "/app/wafcraft/data/prepared/2024-05-06_11-25-19_honeydew-tough",
    }
    # Adversarial Training Settings
    # Sample Creation Settings
    FIND_SAMPLES = True

class Surrogate_Data_V3_Config(Target_Config):
    # General Settings
    NAME = "Surrogate_Data_V3"
    DESCRIPTION = "surrogate model with 50% train, test overlap"
    # Training Settings
    OVERLAP_SETTINGS = {
        "use_overlap": True,
        "overlap": 0.5,
        "overlap_path": "/app/wafcraft/data/prepared/2024-04-07_18-15-53_brown-lot",
    }
    # Adversarial Training Settings
    # Sample Creation Settings
    FIND_SAMPLES = True

class NoAdv_Surrogate_Data_V3_Config(NoAdv_Target_Config):
    # General Settings
    NAME = "NoAdv_Surrogate_Data_V3"
    DESCRIPTION = "surrogate model with 50% train, test overlap"
    # Training Settings
    OVERLAP_SETTINGS = {
        "use_overlap": True,
        "overlap": 0.5,
        "overlap_path": "/app/wafcraft/data/prepared/2024-05-06_11-25-19_honeydew-tough",
    }
    # Adversarial Training Settings
    # Sample Creation Settings
    FIND_SAMPLES = True

class Surrogate_Data_V4_Config(Target_Config):
    # General Settings
    NAME = "Surrogate_Data_V4"
    DESCRIPTION = "surrogate model with 75% train, test overlap"
    # Training Settings
    OVERLAP_SETTINGS = {
        "use_overlap": True,
        "overlap": 0.75,
        "overlap_path": "/app/wafcraft/data/prepared/2024-04-07_18-15-53_brown-lot",
    }
    # Adversarial Training Settings
    # Sample Creation Settings
    FIND_SAMPLES = True

class Large_Surrogate_Data_V4_Config(Large_Target_Config):
    # General Settings
    NAME = "Large_Surrogate_Data_V4"
    DESCRIPTION = "large surrogate model with 75% train, test overlap"
    # Training Settings
    OVERLAP_SETTINGS = {
        "use_overlap": True,
        "overlap": 0,
        "overlap_path": "/app/wafcraft/data/prepared/2024-04-15_10-04-17_turquoise-community",
    }
    # Adversarial Training Settings
    # Sample Creation Settings
    FIND_SAMPLES = True

class NoAdv_Surrogate_Data_V4_Config(NoAdv_Target_Config):
    # General Settings
    NAME = "NoAdv_Surrogate_Data_V4"
    DESCRIPTION = "surrogate model with 75% train, test overlap"
    # Training Settings
    OVERLAP_SETTINGS = {
        "use_overlap": True,
        "overlap": 0.75,
        "overlap_path": "/app/wafcraft/data/prepared/2024-05-06_11-25-19_honeydew-tough",
    }
    # Adversarial Training Settings
    # Sample Creation Settings
    FIND_SAMPLES = True

class Surrogate_Data_V5_Config(Target_Config):
    # General Settings
    NAME = "Surrogate_Data_V5"
    DESCRIPTION = "surrogate model with 100% train, test overlap"
    # Training Settings
    OVERLAP_SETTINGS = {
        "use_overlap": True,
        "overlap": 1,
        "overlap_path": "/app/wafcraft/data/prepared/2024-04-07_18-15-53_brown-lot",
    }
    # Adversarial Training Settings
    # Sample Creation Settings
    FIND_SAMPLES = True

class NoAdv_Surrogate_Data_V5_Config(NoAdv_Target_Config):
    # General Settings
    NAME = "NoAdv_Surrogate_Data_V5"
    DESCRIPTION = "surrogate model with 100% train, test overlap"
    # Training Settings
    OVERLAP_SETTINGS = {
        "use_overlap": True,
        "overlap": 1.0,
        "overlap_path": "/app/wafcraft/data/prepared/2024-05-06_11-25-19_honeydew-tough",
    }
    # Adversarial Training Settings
    # Sample Creation Settings
    FIND_SAMPLES = True

class Surrogate_Paranoia_V1_Config(Target_Config):
    # General Settings
    NAME = "Surrogate_Paranoia_V1"
    DESCRIPTION = "surrogate with PL 1, 100% data/test overlap"
    # Training Settings
    PARANOIA_LEVEL = 1
    OVERLAP_SETTINGS = {
        "use_overlap": True,
        "overlap": 1,
        "overlap_path": "/app/wafcraft/data/prepared/2024-04-07_18-15-53_brown-lot",
    }
    # Adversarial Training Settings
    # Sample Creation Settings
    FIND_SAMPLES = True

class Surrogate_Paranoia_V2_Config(Target_Config):
    # General Settings
    NAME = "Surrogate_Paranoia_V2"
    DESCRIPTION = "surrogate with PL 2, 100% data/test overlap"
    # Training Settings
    PARANOIA_LEVEL = 1
    OVERLAP_SETTINGS = {
        "use_overlap": True,
        "overlap": 1,
        "overlap_path": "/app/wafcraft/data/prepared/2024-04-07_18-15-53_brown-lot",
    }
    # Adversarial Training Settings
    # Sample Creation Settings
    FIND_SAMPLES = True

class Surrogate_Paranoia_V3_Config(Target_Config):
    # General Settings
    NAME = "Surrogate_Paranoia_V3"
    DESCRIPTION = "surrogate with PL 3, 100% data/test overlap"
    # Training Settings
    PARANOIA_LEVEL = 1
    OVERLAP_SETTINGS = {
        "use_overlap": True,
        "overlap": 1,
        "overlap_path": "/app/wafcraft/data/prepared/2024-04-07_18-15-53_brown-lot",
    }
    # Adversarial Training Settings
    # Sample Creation Settings
    FIND_SAMPLES = True