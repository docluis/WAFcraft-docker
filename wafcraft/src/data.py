import base64
import contextlib
import os
import multiprocessing

import numpy as np
import pandas as pd

from tqdm import tqdm
from sklearn.model_selection import train_test_split
from wafamole.evasion import EvasionEngine  # type: ignore
from src.modsec import init_modsec
from src.model import create_wafamole_model, payload_to_vec
from src.utils import load_data_label_vector, log, read_and_parse_sql, split_in_batches


wafamole_log = "logs/wafamole_log.txt"


def create_overlapping_dataset(A, B, C_size, data_overlap):
    shared_size = int(C_size * data_overlap)
    unique_size = C_size - shared_size
    log(f"shared_size: {shared_size}, unique_size: {unique_size}", 2)

    shared = B.sample(shared_size, replace=False)
    log(f"shared: {len(shared)}", 2)

    unique_candidates = A[~A["data"].isin(B["data"])]
    # remove the ones that are already in shared
    unique_candidates = unique_candidates[
        ~unique_candidates["data"].isin(shared["data"])
    ]
    if len(unique_candidates) < unique_size:
        # create an exception here
        raise Exception(f"unique_candidates too small!")
    else:
        unique = unique_candidates.sample(unique_size, replace=False)
        log(f"unique: {len(unique)}", 2)
        return pd.concat([shared, unique], ignore_index=True).sample(frac=1)


def prepare_batches_for_addvec(
    attack_file,
    sane_file,
    train_attacks_size,
    train_sanes_size,
    test_attacks_size,
    test_sanes_size,
    data_path,
    batch_size,
    data_overlap_path,
    data_overlap,
):
    log("reading and parsing raw sql files...", 2)
    attacks = read_and_parse_sql(attack_file)
    attacks["label"] = 1
    attacks = attacks.drop_duplicates(subset=["data"])

    sanes = read_and_parse_sql(sane_file)
    sanes["label"] = 0
    sanes = sanes.drop_duplicates(subset=["data"])

    log("splitting data into train and test...", 2)
    if data_overlap is not None:
        log(f"with overlap {data_overlap} from {data_overlap_path}", 2)
        # load the data from from the previous
        base_train = load_data_label_vector(f"{data_overlap_path}/train.csv")
        base_test = load_data_label_vector(f"{data_overlap_path}/test.csv")
        # drop vector column
        base_train = base_train.drop(columns=["vector"])
        base_test = base_test.drop(columns=["vector"])
        # now split into base_train_attacks, base_train_sanes, base_test_attacks, base_test_sanes
        base_train_attacks = base_train[base_train["label"] == 1]
        base_train_sanes = base_train[base_train["label"] == 0]
        base_test_attacks = base_test[base_test["label"] == 1]
        base_test_sanes = base_test[base_test["label"] == 0]

        train_attacks = create_overlapping_dataset(
            attacks, base_train_attacks, train_attacks_size, data_overlap
        )
        train_sanes = create_overlapping_dataset(
            sanes, base_train_sanes, train_sanes_size, data_overlap
        )
        test_attacks = create_overlapping_dataset(
            attacks, base_test_attacks, test_attacks_size, data_overlap
        )
        test_sanes = create_overlapping_dataset(
            sanes, base_test_sanes, test_sanes_size, data_overlap
        )
    else:
        log("without overlap...", 2)
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

    log("splitting data into batches...", 2)

    # split in batches on disk
    split_in_batches(train, batch_size, f"{data_path}/tmp_addvec", "train")
    split_in_batches(test, batch_size, f"{data_path}/tmp_addvec", "test")


def add_vec(data_path_tmp, file_name, rule_ids, paranoia_level):
    modsec = init_modsec()
    data_set = pd.read_csv(f"{data_path_tmp}/todo/{file_name}")

    tqdm.pandas(desc="Processing payloads")
    data_set["vector"] = data_set["data"].progress_apply(
        lambda x: payload_to_vec(x, rule_ids, modsec, paranoia_level)
    )

    data_set.to_csv(
        f"{data_path_tmp}/addedvec/{file_name}",
        mode="a",
        index=False,
        header=False,
    )


def addvec_batches_in_data_path_tmp(
    data_path_tmp, rule_ids, paranoia_level, max_processes
):
    def load_and_concat_addedvec_batches(files):
        data = pd.concat(
            [
                pd.read_csv(
                    f"{data_path_tmp}/addedvec/{file}",
                    names=["data", "label", "vector"],
                    header=None,
                )
                for file in files
            ]
        )
        data["vector"] = data["vector"].apply(lambda x: np.fromstring(x[1:-1], sep=" "))
        return data

    todo_files_all = os.listdir(f"{data_path_tmp}/todo")
    log(f"addvec to {len(todo_files_all)} batches...", 2)

    # addvec for all batches parallel
    with multiprocessing.Pool(processes=max_processes) as pool:
        pool.starmap(
            add_vec,
            [
                (
                    data_path_tmp,
                    file,
                    rule_ids,
                    paranoia_level,
                )
                for file in todo_files_all
            ],
        )

    # concat
    addvec_files_train = [
        file for file in os.listdir(f"{data_path_tmp}/addedvec") if "train" in file
    ]
    addvec_files_test = [
        file for file in os.listdir(f"{data_path_tmp}/addedvec") if "test" in file
    ]

    log(
        f"concatenating {len(addvec_files_train)} addedvec train baches and {len(addvec_files_test)} addedvec test batches...",
        2,
    )

    train = load_and_concat_addedvec_batches(addvec_files_train)
    test = load_and_concat_addedvec_batches(addvec_files_test)

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
    Prepares batches for optimization and saves them in data_path/tmp_optimize/todo

    Parameters:
        train (pd.DataFrame): Train dataframe
        test (pd.DataFrame): Test dataframe
        train_adv_size (int): Number of adversarial payloads to use for training
        test_adv_size (int): Number of adversarial payloads to use for testing
        batch_size (int): Number of payloads in each batch
        data_path (str): Path to the data directory

    """
    # Sample train and test dataframes, only use attack payloads
    train_adv = (
        train[train["label"] == 1].sample(n=train_adv_size).drop(columns=["vector"])
    )
    test_adv = test[test["label"] == 1].sample(n=test_adv_size).drop(columns=["vector"])

    split_in_batches(train_adv, batch_size, f"{data_path}/tmp_optimize", "train")
    split_in_batches(test_adv, batch_size, f"{data_path}/tmp_optimize", "test")


def optimize(
    data_path,
    file_name,
    model_trained,
    engine_settings,
    rule_ids,
    paranoia_level,
):
    """
    Optimizes payloads in data_path/tmp_optimize/todo/{label}_adv_{batch_number}.csv
    Data is saved to data_path/tmp_optimize/optimized/{label}_adv_{batch_number}.csv

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
    data_set = pd.read_csv(f"{data_path}/tmp_optimize/todo/{file_name}")
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
        f"{data_path}/tmp_optimize/optimized/{file_name}",
        mode="a",
        index=False,
        header=False,
    )
    os.remove(f"{data_path}/tmp_optimize/todo/{file_name}")
    log(f"{file_name} done!", 1)


def optimize_batches_in_todo(
    model_trained,
    engine_settings,
    rule_ids,
    paranoia_level,
    max_processes,
    data_path,
    batch_size,
):
    """
    Reads batches from data_path/tmp_optimize/todo and optimizes them using the trained model and engine settings

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

    def load_and_concat_optimized_batches(files):
        return pd.concat(
            [
                pd.read_csv(
                    f"{data_path}/tmp_optimize/optimized/{file}",
                    names=["data", "label"],
                    header=None,
                )
                for file in files
            ]
        )

    # get all filenames in data_path/tmp_optimize/todo
    todo_files_all = os.listdir(f"{data_path}/tmp_optimize/todo")

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
        file
        for file in os.listdir(f"{data_path}/tmp_optimize/optimized")
        if "train" in file
    ]
    optimized_files_test = [
        file
        for file in os.listdir(f"{data_path}/tmp_optimize/optimized")
        if "test" in file
    ]

    log(
        f"concatenating {len(optimized_files_train)} optimized train baches and {len(optimized_files_test)} optimized test batches...",
        2,
    )

    train_adv = load_and_concat_optimized_batches(optimized_files_train)
    test_adv = load_and_concat_optimized_batches(optimized_files_test)

    # Add vector for payloads in train and test
    log("adding vectors...", 2)
    split_in_batches(
        train_adv, batch_size, f"{data_path}/tmp_addvec_after_optimize", "train_adv"
    )
    split_in_batches(
        test_adv, batch_size, f"{data_path}/tmp_addvec_after_optimize", "test_adv"
    )

    train_adv, test_adv = addvec_batches_in_data_path_tmp(
        f"{data_path}/tmp_addvec_after_optimize",
        rule_ids,
        paranoia_level,
        max_processes,
    )

    return train_adv, test_adv
