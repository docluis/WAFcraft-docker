import base64
import contextlib
import io
import os
import multiprocessing

import sqlparse
import pandas as pd

from tqdm import tqdm
from sklearn.model_selection import train_test_split
from wafamole.evasion import EvasionEngine  # type: ignore
from src.modsec import init_modsec
from src.model import create_wafamole_model, payload_to_vec
from src.utils import load_and_concat_csv, log, read_and_parse_sql


wafamole_log = "logs/wafamole_log.txt"


def add_vec(data_path, file_name, rule_ids, paranoia_level):
    modsec = init_modsec()
    data_set = pd.read_csv(f"{data_path}/tmp_addvec/todo/{file_name}")

    tqdm.pandas(desc="Processing payloads")
    data_set["vector"] = data_set["data"].progress_apply(
        lambda x: payload_to_vec(x, rule_ids, modsec, paranoia_level)
    )

    data_set.to_csv(
        f"{data_path}/tmp_addvec/optimized/{file_name}",
        mode="a",
        index=False,
        header=False,
    )


def prepare_batches_for_addvec(
    attack_file,
    sane_file,
    train_attacks_size,
    train_sanes_size,
    test_attacks_size,
    test_sanes_size,
    rule_ids,
    paranoia_level,
    data_path,
    batch_size,
):
    attacks = read_and_parse_sql(attack_file)
    attacks["label"] = 1
    sanes = read_and_parse_sql(sane_file)
    sanes["label"] = 0

    train_attacks, test_attacks = train_test_split(
        attacks,
        train_size=train_attacks_size,
        test_size=test_attacks_size,
        stratify=attacks["label"],
    )

    train_sanes, test_sanes = train_test_split(
        sanes,
        train_size=train_sanes_size,
        test_size=test_sanes_size,
        stratify=sanes["label"],
    )

    train = (
        pd.concat([train_attacks, train_sanes]).sample(frac=1).reset_index(drop=True)
    )
    test = pd.concat([test_attacks, test_sanes]).sample(frac=1).reset_index(drop=True)

    os.makedirs(f"{data_path}/tmp_addvec/todo", exist_ok=True)
    os.makedirs(f"{data_path}/tmp_addvec/addedvec", exist_ok=True)

    # split in batches

    train_batches = [
        train[i : i + batch_size] for i in range(0, len(train), batch_size)
    ]
    test_batches = [test[i : i + batch_size] for i in range(0, len(test), batch_size)]
    for i, batch in enumerate(train_batches):
        batch.to_csv(
            f"{data_path}/tmp_addvec/todo/train_{i}.csv", index=False, header=True
        )
    for i, batch in enumerate(test_batches):
        batch.to_csv(
            f"{data_path}/tmp_addvec/todo/test_{i}.csv", index=False, header=True
        )


def addvec_batches_in_todo(data_path, rule_ids, paranoia_level, max_processes):
    # load all files in data_path/tmp_addvec/todo

    todo_files_all = os.listdir(f"{data_path}/tmp/tmp_addvec")
    log(f"addvec {len(todo_files_all)} batches...", 2)

    # addvec for all batches parallel
    with multiprocessing.Pool(processes=max_processes) as pool:
        pool.starmap(
            add_vec,
            [
                (
                    data_path,
                    file,
                    rule_ids,
                    paranoia_level,
                )
                for file in todo_files_all
            ],
        )

    # concat and delete temporary dir
    addvec_files_train = [
        file
        for file in os.listdir(f"{data_path}/tmp_addvec/addedvec")
        if "train" in file
    ]
    addvec_files_test = [
        file
        for file in os.listdir(f"{data_path}/tmp_addvec/addedvec")
        if "test" in file
    ]

    log(
        f"concatenating {len(addvec_files_train)} optimized train baches and {len(addvec_files_test)} optimized test batches...",
        2,
    )

    train = load_and_concat_csv(addvec_files_train)
    test = load_and_concat_csv(addvec_files_test)

    # return test, train
    return train, test


def prepare_batches_for_optimization(
    train,
    test,
    train_adv_size,
    test_adv_size,
    batch_size,
    data_path,
):
    """
    Prepares batches for optimization and saves them in data_path/tmp/todo

    Parameters:
        train (pd.DataFrame): Train dataframe
        test (pd.DataFrame): Test dataframe
        train_adv_size (int): Number of adversarial payloads to use for training
        test_adv_size (int): Number of adversarial payloads to use for testing
        batch_size (int): Number of payloads in each batch
        data_path (str): Path to the data directory

    """
    # create directories
    os.makedirs(f"{data_path}/tmp/todo", exist_ok=True)
    os.makedirs(f"{data_path}/tmp/optimized", exist_ok=True)

    # Sample train and test dataframes, only use attack payloads
    train_adv = (
        train[train["label"] == 1].sample(n=train_adv_size).drop(columns=["vector"])
    )
    test_adv = test[test["label"] == 1].sample(n=test_adv_size).drop(columns=["vector"])

    train_adv_batches = [
        train_adv[i : i + batch_size] for i in range(0, len(train_adv), batch_size)
    ]
    test_adv_batches = [
        test_adv[i : i + batch_size] for i in range(0, len(test_adv), batch_size)
    ]

    # Save each batch to a csv in data_path/tmp/todo
    for i, batch in enumerate(train_adv_batches):
        batch.to_csv(
            f"{data_path}/tmp/todo/train_adv_{i}.csv", index=False, header=True
        )
    for i, batch in enumerate(test_adv_batches):
        batch.to_csv(f"{data_path}/tmp/todo/test_adv_{i}.csv", index=False, header=True)


def optimize(
    data_path,
    file_name,
    model_trained,
    engine_settings,
    rule_ids,
    paranoia_level,
):
    """
    Optimizes payloads in data_path/tmp/todo/{label}_adv_{batch_number}.csv
    Data is saved to data_path/tmp/optimized/{label}_adv_{batch_number}.csv

    Parameters:
        data_path (str): Path to the data directory
        batch_number (int): Number of the batch
        label (str): Label of the batch
        model_trained (sklearn.base.BaseEstimator): Trained model
        engine_settings (dict): Settings for the evasion engine
        rule_ids (list): List of rule IDs
        paranoia_level (int): Paranoia level
    """
    modsec = init_modsec()
    wafamole_model = create_wafamole_model(
        model_trained, modsec, rule_ids, paranoia_level
    )
    data_set = pd.read_csv(f"{data_path}/tmp/todo/{file_name}")
    engine = EvasionEngine(wafamole_model)
    with open(wafamole_log, "a") as f:
        for i, row in tqdm(data_set.iterrows(), total=len(data_set)):
            min_payload = None
            try:
                with contextlib.redirect_stdout(f):
                    min_confidence, min_payload = engine.evaluate(
                        payload=base64.b64decode(row["data"]).decode("utf-8"),
                        **engine_settings,
                    )
                data_set.at[i, "data"] = base64.b64encode(
                    min_payload.encode("utf-8")
                ).decode("utf-8")
            except Exception as e:
                log(f"Error optimizing payload {i}: {e}")
                log(f"Payload: {row['data']}")
                if min_payload is not None:
                    data_set.at[i, "data"] = base64.b64encode(
                        min_payload.encode("utf-8")
                    ).decode("utf-8")
                    log(f"min_payload not None: {min_payload}")
                continue
    data_set.to_csv(
        f"{data_path}/tmp/optimized/{file_name}",
        mode="a",
        index=False,
        header=False,
    )
    os.remove(f"{data_path}/tmp/todo/{file_name}")
    log(f"{file_name} done!", 1)


def optimize_batches_in_todo(
    model_trained,
    engine_settings,
    rule_ids,
    paranoia_level,
    max_processes,
    data_path,
):
    """
    Reads batches from data_path/tmp/todo and optimizes them using the trained model and engine settings

    Parameters:
        model_trained (sklearn.base.BaseEstimator): Trained model
        engine_settings (dict): Settings for the evasion engine
        rule_ids (list): List of rule IDs
        paranoia_level (int): Paranoia level
        max_processes (int): Maximum number of processes to use
        data_path (str): Path to the data directory

    Returns:
        pd.DataFrame, pd.DataFrame: Train and test dataframes with adversarial payloads
    """

    # get all filenames in data_path/tmp/todo
    todo_files_all = os.listdir(f"{data_path}/tmp/todo")

    log(f"optimizing {len(todo_files_all)} batches...", 2)

    # optimize is prone to TimeoutError, so use multiprocessing
    with multiprocessing.Pool(processes=max_processes) as pool:
        pool.starmap(
            optimize,
            [
                (
                    data_path,
                    file,
                    model_trained,
                    engine_settings,
                    rule_ids,
                    paranoia_level,
                )
                for file in todo_files_all
            ],
        )

    # Read and concatenate optimized train
    optimized_files_train = [
        file for file in os.listdir(f"{data_path}/tmp/optimized") if "train" in file
    ]
    optimized_files_test = [
        file for file in os.listdir(f"{data_path}/tmp/optimized") if "test" in file
    ]

    log(
        f"concatenating {len(optimized_files_train)} optimized train baches and {len(optimized_files_test)} optimized test batches...",
        2,
    )

    train_adv = load_and_concat_csv(optimized_files_train, data_path)
    test_adv = load_and_concat_csv(optimized_files_test, data_path)

    # Add vector for payloads in train and test
    log("adding vectors...", 2)
    modsec = init_modsec()
    train_adv = add_vec(train_adv, rule_ids, modsec, paranoia_level)
    test_adv = add_vec(test_adv, rule_ids, modsec, paranoia_level)

    return train_adv, test_adv
