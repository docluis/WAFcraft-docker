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


f = io.StringIO()


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


def optimize(
    data_path,
    batch_number,
    label,
    model_trained,
    engine_settings,
    rule_ids,
    paranoia_level,
    tmp_path,
):
    modsec = init_modsec()
    wafamole_model = create_wafamole_model(
        model_trained, modsec, rule_ids, paranoia_level
    )
    data_set = pd.read_csv(data_path)
    engine = EvasionEngine(wafamole_model)
    with open("/app/wafcraft/logs/wafamole_log.txt", "a") as f:
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
        f"{tmp_path}/optimized/{label}_adv_{batch_number}.csv",
        mode="a",
        index=False,
        header=False,
    )
    log(f"{label} batch {batch_number} done!")


def create_adv_train_test_split(
    train,
    test,
    train_adv_size,
    test_adv_size,
    model_trained,
    engine_settings,
    rule_ids,
    paranoia_level,
    batch_size,
    max_processes,
    tmp_path,
):
    """
    Returns train and test dataframes with adversarial payloads

    Parameters:
        train (pd.DataFrame): Train dataframe
        test (pd.DataFrame): Test dataframe
        train_adv_size (float): Number of adversarial payloads to generate for training
        test_adv_size (float): Number of adversarial payloads to generate for testing
        model_trained (sklearn.ensemble.RandomForestClassifier): Trained model
        engine_settings (dict): Settings for the model
        rule_ids (list): List of rule IDs
        paranoia_level (int): Paranoia level
        batch_size (int): Batch size for optimization

    Returns:
        pd.DataFrame, pd.DataFrame: Train and test dataframes with adversarial payloads
    """
    # create directories
    os.makedirs(f"{tmp_path}/todo", exist_ok=True)
    os.makedirs(f"{tmp_path}/optimized", exist_ok=True)

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

    # Save each batch to a csv
    for i, batch in enumerate(train_adv_batches):
        batch.to_csv(f"{tmp_path}/todo/train_adv_{i}.csv", index=False, header=True)
    for i, batch in enumerate(test_adv_batches):
        batch.to_csv(f"{tmp_path}/todo/test_adv_{i}.csv", index=False, header=True)

    # Optimize each batch with subproceesses

    # optimize is prone to TimeoutError, so use multiprocessing

    with multiprocessing.Pool(processes=max_processes) as pool:
        pool.starmap(
            optimize,
            [
                (
                    f"{tmp_path}/todo/train_adv_{i}.csv",
                    i,
                    "train",
                    model_trained,
                    engine_settings,
                    rule_ids,
                    paranoia_level,
                    tmp_path,
                )
                for i in range(len(train_adv_batches))
            ],
        )
        pool.starmap(
            optimize,
            [
                (
                    f"{tmp_path}/todo/test_adv_{i}.csv",
                    i,
                    "test",
                    model_trained,
                    engine_settings,
                    rule_ids,
                    paranoia_level,
                    tmp_path,
                )
                for i in range(len(test_adv_batches))
            ],
        )

    log("Done optimizing, concatenating...", 2)

    # Read and concatenate optimized batches (keep in mind that there are no names for the columns)
    # TODO: ? some files may not exist, so use try-except
    train_adv = pd.concat(
        [
            pd.read_csv(
                f"{tmp_path}/optimized/train_adv_{i}.csv",
                names=["data", "label"],
                header=None,
            )
            for i in range(len(train_adv_batches))
        ]
    )
    test_adv = pd.concat(
        [
            pd.read_csv(
                f"{tmp_path}/optimized/test_adv_{i}.csv",
                names=["data", "label"],
                header=None,
            )
            for i in range(len(test_adv_batches))
        ]
    )

    log(f"Train_adv shape: {train_adv.shape} | Test_adv shape: {test_adv.shape}", 2)

    # Add vector for payloads in train and test
    log("Creating vectors...", 2)
    modsec = init_modsec()
    train_adv = add_vec(train_adv, rule_ids, modsec, paranoia_level)
    test_adv = add_vec(test_adv, rule_ids, modsec, paranoia_level)

    return train_adv, test_adv
