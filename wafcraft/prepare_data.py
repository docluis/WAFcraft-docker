import datetime
import os
import shutil

import joblib
from src.utils import log
from src.data import create_train_test_split, create_adv_train_test_split
from src.model import train_model

from config import BaseConfig

# Choose the configuration
Config = BaseConfig
log(f"Using Configuration: {Config.DESCRIPTION}")

# get timestamp
ts = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
data_path = f"data/prepared/{ts}"
# create directory
os.makedirs(data_path, exist_ok=True)
os.makedirs(f"{data_path}/tmp", exist_ok=True)

# main
if __name__ == "__main__":

    # 1. create train and test sets
    train, test = create_train_test_split(
        attack_file=Config.ATTACK_DATA_PATH,
        sane_file=Config.SANE_DATA_PATH,
        train_attacks_size=Config.TRAIN_ATTACKS_SIZE,
        train_sanes_size=Config.TRAIN_SANES_SIZE,
        test_attacks_size=Config.TEST_ATTACKS_SIZE,
        test_sanes_size=Config.TEST_SANES_SIZE,
        rule_ids=Config.RULE_IDS,
        paranoia_level=Config.PARANOIA_LEVEL,
    )
    train.to_csv(f"{data_path}/train.csv", index=False)
    test.to_csv(f"{data_path}/test.csv", index=False)

    # 2. train Config.MODEL
    model_trained, threshold = train_model(
        train=train, test=test, model=Config.MODEL, desired_fpr=Config.DESIRED_FPR
    )
    joblib.dump(model_trained, f"{data_path}/model.joblib")
    with open(f"{data_path}/threshold.txt", "w") as f:
        f.write(str(threshold))

    # 3. create adversarial examples
    train_adv, test_adv = create_adv_train_test_split(
        train=train,
        test=test,
        train_adv_size=Config.TRAIN_ADV_SIZE,
        test_adv_size=Config.TEST_ADV_SIZE,
        model_trained=model_trained,
        engine_settings=Config.ENGINE_SETTINGS,
        rule_ids=Config.RULE_IDS,
        paranoia_level=Config.PARANOIA_LEVEL,
        batch_size=Config.BATCH_SIZE,
        tmp_path=f"{data_path}/tmp",
    )
    train_adv.to_csv(f"{data_path}/train_adv.csv", index=False)
    test_adv.to_csv(f"{data_path}/test_adv.csv", index=False)
    # delete tmp files
    shutil.rmtree(f"{data_path}/tmp")

    log(f"Data prepared and saved to {data_path}")
