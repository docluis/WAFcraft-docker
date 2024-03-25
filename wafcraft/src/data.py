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
from src.utils import log


wafamole_log = "logs/wafamole_log.txt"


def add_vec(data, rule_ids, modsec, paranoia_level):
    """
    Returns a dataframe with vectorized payloads

    Parameters:
        data (pd.DataFrame): Dataframe containing payloads
        rule_ids (list): List of rule IDs
        modsec (modsecurity.ModSecurity): ModSecurity instance
        paranoia_level (int): Paranoia level

    Returns:
        pd.DataFrame: Dataframe with vectorized payloads
    """
    tqdm.pandas(desc="Processing payloads")
    data["vector"] = data["data"].progress_apply(
        lambda x: payload_to_vec(x, rule_ids, modsec, paranoia_level)
    )
    return data


def create_train_test_split(
    attack_file,
    sane_file,
    train_attacks_size,
    train_sanes_size,
    test_attacks_size,
    test_sanes_size,
    rule_ids,
    paranoia_level,
):
    """
    Returns train and test dataframes with vectorized payloads

    Parameters:
        attack_file (str): Path to the file containing attack payloads
        sane_file (str): Path to the file containing sane payloads
        train_attacks_size (float): Number of attack payloads to use for training
        train_sanes_size (float): Number of sane payloads to use for training
        test_attacks_size (float): Number of attack payloads to use for testing
        test_sanes_size (float): Number of sane payloads to use for testing
        rule_ids (list): List of rule IDs
        paranoia_level (int): Paranoia level
    """

    def read_and_parse(file_path):
        content = open(file_path, "r").read()
        statements = sqlparse.split(content)

        parsed_data = []
        for statement in statements:
            base64_statement = base64.b64encode(statement.encode("utf-8")).decode(
                "utf-8"
            )
            parsed_data.append({"data": base64_statement})

        return pd.DataFrame(parsed_data)

    log("Reading and parsing data...", 2)

    attacks = read_and_parse(attack_file)
    attacks["label"] = 1
    sanes = read_and_parse(sane_file)
    sanes["label"] = 0

    log("Splitting into train and test...", 2)
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

    # Concatenate and shuffle
    train = (
        pd.concat([train_attacks, train_sanes]).sample(frac=1).reset_index(drop=True)
    )
    test = pd.concat([test_attacks, test_sanes]).sample(frac=1).reset_index(drop=True)

    # Add vector for payloads in train and test
    log("Creating vectors...", 2)
    modsec = init_modsec()
    train = add_vec(train, rule_ids, modsec, paranoia_level)
    test = add_vec(test, rule_ids, modsec, paranoia_level)

    log("Done!")
    log(f"Train shape: {train.shape} | Test shape: {test.shape}", 2)
    return train, test


def prepare_batches_to_todo(
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


def optimize_data_in_todo(
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

    def load_and_concat_csv(files):
        return pd.concat(
            [
                pd.read_csv(
                    f"{data_path}/tmp/optimized/{file}",
                    names=["data", "label"],
                    header=None,
                )
                for file in files
            ]
        )

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

    train_adv = load_and_concat_csv(optimized_files_train)
    test_adv = load_and_concat_csv(optimized_files_test)

    # Add vector for payloads in train and test
    log("adding vectors...", 2)
    modsec = init_modsec()
    train_adv = add_vec(train_adv, rule_ids, modsec, paranoia_level)
    test_adv = add_vec(test_adv, rule_ids, modsec, paranoia_level)

    return train_adv, test_adv
