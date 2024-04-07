# find adversarial sampes for a given model

import argparse
import os
import shutil
import joblib
import pandas as pd
from src.data import optimize_batches_in_todo
from src.model import train_model
from src.utils import (
    get_most_recent_data_path,
    load_data_label_vector,
    log,
    read_and_parse_sql,
    split_in_batches,
)
from config import HalfConfig, PaperConfig, BaseConfig, StressConfig


def find_samples(Config, data_path, attempts):
    log(f"finding adversarial samples with {attempts} attempts...", 1)
    log(f"using config: {Config.DESCRIPTION}", 2)
    log(f"using data path: {data_path}", 2)
    # create directories
    os.makedirs(f"{data_path}/tmp_optimize", exist_ok=True)
    os.makedirs(f"{data_path}/tmp_optimize/todo", exist_ok=True)
    os.makedirs(f"{data_path}/tmp_optimize/optimized", exist_ok=True)

    # load data
    train = load_data_label_vector(f"{data_path}/train.csv")
    test = load_data_label_vector(f"{data_path}/test.csv")
    train_adv = load_data_label_vector(f"{data_path}/train_adv.csv")
    test_adv = load_data_label_vector(f"{data_path}/test_adv.csv")

    # load model (potentaly not needed)
    model_trained = joblib.load(f"{data_path}/model/model.joblib")
    threshold = float(open(f"{data_path}/model/threshold.txt", "r").read())

    # train adv model
    model_adv_trained, threshold_adv = train_model(
        train=pd.concat([train, train_adv]).sample(frac=1).reset_index(drop=True),
        test=pd.concat([test, test_adv]).sample(frac=1).reset_index(drop=True),
        model=Config.MODEL_ADV,
        desired_fpr=Config.DESIRED_FPR,
        image_path=f"{data_path}/model_adv",
    )

    # save model_adv and threshold_adv
    joblib.dump(model_adv_trained, f"{data_path}/model_adv/model_adv.joblib")
    with open(f"{data_path}/model_adv/threshold_adv.txt", "w") as f:
        f.write(str(threshold_adv))

    # sample attacks
    log(f"reading and parsing sql file...", 2)
    attacks = read_and_parse_sql(Config.ATTACK_DATA_PATH)
    attacks["label"] = 1
    log(f"sampling {attempts} attacks...", 2)
    attacks_sample = attacks.sample(attempts).reset_index(drop=True)

    # split attacks into batches
    log(f"splitting attacks samples into batches...", 2)
    split_in_batches(
        attacks_sample, Config.BATCH_SIZE, f"{data_path}/tmp_optimize", "sample"
    )

    # optimize batches
    log(f"optimizing batches...", 2)
    _, _, sample_adv = optimize_batches_in_todo(
        model_trained=model_adv_trained,
        engine_settings=Config.ENGINE_SETTINGS_SAMPLE_CREATION,
        rule_ids=Config.RULE_IDS,
        paranoia_level=Config.PARANOIA_LEVEL,
        max_processes=Config.MAX_PROCESSES,
        data_path=data_path,
    )

    # only save the samples with min_confidence column < threshold_adv
    sample_adv = sample_adv[sample_adv["min_confidence"] < threshold_adv]
    log(f"found {sample_adv.shape[0]} adversarial samples", 3)

    # append to file if it exists else create it
    if os.path.exists(f"{data_path}/sample_adv.csv"):
        sample_adv.to_csv(
            f"{data_path}/sample_adv.csv", mode="a", header=False, index=False
        )
    else:
        sample_adv.to_csv(f"{data_path}/sample_adv.csv", index=False)

    shutil.rmtree(f"{data_path}/tmp_optimize")


if __name__ == "__main__":
    most_recent_data_path = get_most_recent_data_path()

    parser = argparse.ArgumentParser(
        description="Find adversarial samples for a given model."
    )
    parser.add_argument(
        "--config",
        choices=["base", "half", "stress", "paper"],
        help="Choose the configuration to use.",
        required=True,
    )
    parser.add_argument(
        "--data_path",
        type=str,
        help="Path to the data directory.",
        default=most_recent_data_path,
    )
    parser.add_argument(
        "--attempts",
        type=int,
        help="Number of attempts to find adversarial samples.",
        required=True,
    )

    args = parser.parse_args()

    # Set config based on the argument
    if args.config == "base":
        Config = BaseConfig
    elif args.config == "half":
        Config = HalfConfig
    elif args.config == "stress":
        Config = StressConfig
    elif args.config == "paper":
        Config = PaperConfig
    else:
        raise ValueError("Invalid config")

    find_samples(Config, args.data_path, args.attempts)
