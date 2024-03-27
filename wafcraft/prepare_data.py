# This script prepares the data for training and testing the model.

# Description:
# The script can be run in two modes:
# - prepare: Prepare the data and train the model
# - optimize: Optimize the data using the trained model

# To prepare train, test, train_adv, and test_adv data both modes are required.

# (Since the optimization process is error-prone and can be interrupted, optimization
# is moved to a separate mode to allow for continuing  optimization process after it is
# interrupted or some error occurs)

# Usage:
# python prepare_data.py --mode prepare
# python prepare_data.py --mode optimize
# python prepare_data.py --mode optimize --data_path data/prepared/2023-12-31-12-00-00

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
    addvec_batches_in_data_path_tmp,
    optimize_batches_in_todo,
    prepare_batches_for_addvec,
    prepare_batches_for_optimization,
)
from src.model import train_model

from config import BaseConfig, HalfConfig, StressConfig, PaperConfig

os.makedirs("data/prepared", exist_ok=True)

# Choose the configuration
Config = PaperConfig


def prepare_and_train():
    ts = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    data_path = f"data/prepared/{ts}"
    os.makedirs(data_path, exist_ok=True)
    os.makedirs(f"{data_path}/tmp_optimize", exist_ok=True)
    os.makedirs(f"{data_path}/tmp_addvec", exist_ok=True)
    os.makedirs(f"{data_path}/tmp_addvec_after_optimize", exist_ok=True)

    os.makedirs(f"{data_path}/tmp_addvec/todo", exist_ok=True)
    os.makedirs(f"{data_path}/tmp_addvec/addedvec", exist_ok=True)
    os.makedirs(f"{data_path}/tmp_optimize/todo", exist_ok=True)
    os.makedirs(f"{data_path}/tmp_optimize/optimized", exist_ok=True)
    os.makedirs(f"{data_path}/tmp_addvec_after_optimize/todo", exist_ok=True)
    os.makedirs(f"{data_path}/tmp_addvec_after_optimize/addedvec", exist_ok=True)

    log("Starting data preparation", 2)
    log(f"Using Config:\n{get_config_string(Config)}", 2)

    # 1. save config
    with open(f"{data_path}/config.txt", "w") as f:
        f.write(get_config_string(Config))

    # 2. read and parse data, split into train and test, split into batches
    prepare_batches_for_addvec(
        attack_file=Config.ATTACK_DATA_PATH,
        sane_file=Config.SANE_DATA_PATH,
        train_attacks_size=Config.TRAIN_ATTACKS_SIZE,
        train_sanes_size=Config.TRAIN_SANES_SIZE,
        test_attacks_size=Config.TEST_ATTACKS_SIZE,
        test_sanes_size=Config.TEST_SANES_SIZE,
        data_path=data_path,
        batch_size=Config.BATCH_SIZE,
    )
    # 3. add vectors to batches, concatenate them and save them to disk
    train, test = addvec_batches_in_data_path_tmp(
        data_path_tmp=f"{data_path}/tmp_addvec",
        rule_ids=Config.RULE_IDS,
        paranoia_level=Config.PARANOIA_LEVEL,
        max_processes=Config.MAX_PROCESSES,
    )
    train.to_csv(f"{data_path}/train.csv", index=False)
    test.to_csv(f"{data_path}/test.csv", index=False)
    shutil.rmtree(f"{data_path}/tmp_addvec")

    # 4. train model and save it
    model_trained, threshold = train_model(
        train=train, test=test, model=Config.MODEL, desired_fpr=Config.DESIRED_FPR
    )
    joblib.dump(model_trained, f"{data_path}/model.joblib")
    with open(f"{data_path}/threshold.txt", "w") as f:
        f.write(str(threshold))

    # 5. choose payloads for train_adv and test_adv and split them into batches
    prepare_batches_for_optimization(
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


def optimize_data(data_path):
    # 1. load model
    model_trained = joblib.load(f"{data_path}/model.joblib")

    # 2. optimize, add vectors and save to disk
    train_adv, test_adv = optimize_batches_in_todo(
        model_trained=model_trained,
        engine_settings=Config.ENGINE_SETTINGS,
        rule_ids=Config.RULE_IDS,
        paranoia_level=Config.PARANOIA_LEVEL,
        max_processes=Config.MAX_PROCESSES,
        data_path=data_path,
        batch_size=Config.BATCH_SIZE,
    )

    train_adv.to_csv(f"{data_path}/train_adv.csv", index=False)
    test_adv.to_csv(f"{data_path}/test_adv.csv", index=False)
    # delete tmp files
    shutil.rmtree(f"{data_path}/tmp_optimize")
    shutil.rmtree(f"{data_path}/tmp_addvec_after_optimize")

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
