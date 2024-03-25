import datetime
import os
import shutil
import argparse

import joblib
from src.utils import (
    get_config_string,
    get_most_recent_data_path,
    log,
)
from src.data import (
    create_train_test_split,
    optimize_data_in_todo,
    prepare_batches_to_todo,
)
from src.model import train_model

from config import BaseConfig, HalfConfig, StressConfig

# Choose the configuration
Config = HalfConfig

# get timestamp
ts = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
data_path = f"data/prepared/{ts}"


def prepare_and_train():
    os.makedirs(data_path, exist_ok=True)
    os.makedirs(f"{data_path}/tmp", exist_ok=True)
    log("Starting data preparation", 2)
    log(f"Using Config:\n{get_config_string(Config)}", 2)

    # 1. save config
    with open(f"{data_path}/config.txt", "w") as f:
        f.write(get_config_string(Config))

    # 2. create train and test sets and save them
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

    # 3. train model and save it
    model_trained, threshold = train_model(
        train=train, test=test, model=Config.MODEL, desired_fpr=Config.DESIRED_FPR
    )
    joblib.dump(model_trained, f"{data_path}/model.joblib")
    with open(f"{data_path}/threshold.txt", "w") as f:
        f.write(str(threshold))

    # 4. prepare batches for optimization and save them in {data_path}/tmp/todo
    prepare_batches_to_todo(
        train=train,
        test=test,
        train_adv_size=Config.TRAIN_ADV_SIZE,
        test_adv_size=Config.TEST_ADV_SIZE,
        batch_size=Config.BATCH_SIZE,
        data_path=data_path,
    )

    log(
        f"Initial preparation and training completed. train: {train.shape[0]} test: {test.shape[0]}",
        3,
    )
    log(f"Saved to {data_path}", 2)


# main
def optimize_data(data_path):
    # 1. load model
    model_trained = joblib.load(f"{data_path}/model.joblib")

    # 2. optimize
    train_adv, test_adv = optimize_data_in_todo(
        model_trained=model_trained,
        engine_settings=Config.ENGINE_SETTINGS,
        rule_ids=Config.RULE_IDS,
        paranoia_level=Config.PARANOIA_LEVEL,
        max_processes=Config.MAX_PROCESSES,
        data_path=data_path,
    )

    train_adv.to_csv(f"{data_path}/train_adv.csv", index=False)
    test_adv.to_csv(f"{data_path}/test_adv.csv", index=False)
    # delete tmp files
    shutil.rmtree(f"{data_path}/tmp")

    log(
        f"Optimization completed. train_adv: {train_adv.shape[0]} test_adv: {test_adv.shape[0]}",
        3,
    )
    log(f"Saved to {data_path}", 2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run data preparation and model training or optimization"
    )
    parser.add_argument(
        "--mode",
        choices=["prepare", "optimize"],
        required=True,
        help="Choose whether to prepare and train model or to optimize data.",
    )

    most_recent_data_path = get_most_recent_data_path()
    parser.add_argument(
        "--data_path",
        type=str,
        help="Path to the data directory.",
        default=most_recent_data_path,
    )

    args = parser.parse_args()

    if args.mode == "prepare":
        prepare_and_train()
    elif args.mode == "optimize":
        optimize_data(args.data_path)
    else:
        raise ValueError("Invalid mode")
