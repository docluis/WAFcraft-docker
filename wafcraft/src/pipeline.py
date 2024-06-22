import os

import joblib
import pandas as pd

from src.model import train_model
from src.data import (
    addvec_batches_in_tmp_addvec_dir,
    addvec_to_optmized_batches,
    optimize_batches_in_todo,
    prepare_batches_for_addvec,
    prepare_batches_for_optimization,
)
from src.utils import (
    determine_progress,
    load_data_label_vector,
    log,
    read_and_parse_sql,
    remove_tmp_dirs,
    save_model,
)


def run_model_pipeline(Config, workspace):
    (
        has_model,
        has_tmp_optimize_todos,
        has_adv_model,
        has_tmp_samples_todos,
        has_sample_adv_csv,
    ) = determine_progress(workspace)

    log(f"has_model: {has_model}", 2)
    log(f"has_tmp_optimize_todos: {has_tmp_optimize_todos}", 2)
    log(f"has_adv_model: {has_adv_model}", 2)
    log(f"has_tmp_samples_todos: {has_tmp_samples_todos}", 2)
    log(f"has_sample_adv_csv: {has_sample_adv_csv}", 2)

    # ----------------- model -----------------
    if has_model:
        model_trained = joblib.load(f"{workspace}/model/model.joblib")
        threshold = float(open(f"{workspace}/model/threshold.txt", "r").read())
        train = load_data_label_vector(f"{workspace}/train.csv")
        test = load_data_label_vector(f"{workspace}/test.csv")
        log(f"[model] loaded model, train and test", 2)
    else:
        os.makedirs(f"{workspace}/model", exist_ok=True)
        os.makedirs(f"{workspace}/tmp_addvec", exist_ok=True)
        os.makedirs(f"{workspace}/tmp_addvec/todo", exist_ok=True)
        os.makedirs(f"{workspace}/tmp_addvec/addedvec", exist_ok=True)
        # make data
        prepare_batches_for_addvec(
            attack_file=Config.ATTACK_DATA_PATH,
            sane_file=Config.SANE_DATA_PATH,
            train_attacks_size=Config.TRAIN_ATTACKS_SIZE,
            train_sanes_size=Config.TRAIN_SANES_SIZE,
            test_attacks_size=Config.TEST_ATTACKS_SIZE,
            test_sanes_size=Config.TEST_SANES_SIZE,
            data_path=workspace,
            batch_size=Config.BATCH_SIZE,
            overlap_settings=Config.OVERLAP_SETTINGS,
        )
        train, test = addvec_batches_in_tmp_addvec_dir(
            tmp_addvec_dir=f"{workspace}/tmp_addvec",
            rule_ids=Config.RULE_IDS,
            paranoia_level=Config.PARANOIA_LEVEL,
            max_processes=Config.MAX_PROCESSES,
        )
        train.to_csv(f"{workspace}/train.csv", index=False)
        test.to_csv(f"{workspace}/test.csv", index=False)
        log(
            f"[model] prepared data, with train: {train.shape[0]} test: {test.shape[0]}",
            3,
        )

        # make model
        model_trained, threshold = train_model(
            train=train,
            test=test,
            model=Config.MODEL,
            desired_fpr=Config.DESIRED_FPR,
            image_path=f"{workspace}/model",
        )
        save_model(workspace, "model", model_trained, threshold, False)
        log(f"[model] inital model done", 2)

    # ----------------- adv_model -----------------
    if Config.ADVERSARIAL_TRAINING:
        if has_adv_model:
            model_adv_trained = joblib.load(f"{workspace}/model_adv/model_adv.joblib")
            threshold_adv = float(
                open(f"{workspace}/model_adv/threshold_adv.txt", "r").read()
            )
            train_adv = load_data_label_vector(f"{workspace}/train_adv.csv")
            test_adv = load_data_label_vector(f"{workspace}/test_adv.csv")
            log(f"[adv_model] loaded model_adv, train_adv and test_adv", 2)
        else:
            os.makedirs(f"{workspace}/model_adv", exist_ok=True)
            os.makedirs(f"{workspace}/tmp_optimize", exist_ok=True)
            os.makedirs(f"{workspace}/tmp_optimize/todo", exist_ok=True)
            os.makedirs(f"{workspace}/tmp_optimize/optimized", exist_ok=True)
            os.makedirs(f"{workspace}/tmp_addvec_after_optimize", exist_ok=True)
            os.makedirs(f"{workspace}/tmp_addvec_after_optimize/todo", exist_ok=True)
            os.makedirs(
                f"{workspace}/tmp_addvec_after_optimize/addedvec", exist_ok=True
            )
            # make adversarial data
            if not has_tmp_optimize_todos:
                prepare_batches_for_optimization(
                    data_set=train,
                    number=Config.TRAIN_ADV_SIZE,
                    batch_size=Config.BATCH_SIZE,
                    data_path=workspace,
                    tmp_dir="tmp_optimize",
                    label="train",
                )
                prepare_batches_for_optimization(
                    data_set=test,
                    number=Config.TEST_ADV_SIZE,
                    batch_size=Config.BATCH_SIZE,
                    data_path=workspace,
                    tmp_dir="tmp_optimize",
                    label="test",
                )

            log(f"[adv_model] start optimizing batches for adversarial training...", 2)
            train_adv, test_adv, _ = optimize_batches_in_todo(
                model_trained=model_trained,
                engine_settings=Config.ENGINE_SETTINGS,
                rule_ids=Config.RULE_IDS,
                paranoia_level=Config.PARANOIA_LEVEL,
                max_processes=Config.MAX_PROCESSES,
                data_path=workspace,
                tmp_dir=f"tmp_optimize",
            )
            log(f"[adv_model] optimized batches for adversarial training done", 2)

            train_adv, test_adv = addvec_to_optmized_batches(
                train_adv=train_adv,
                test_adv=test_adv,
                batch_size=Config.BATCH_SIZE,
                data_path=workspace,
                rule_ids=Config.RULE_IDS,
                paranoia_level=Config.PARANOIA_LEVEL,
                max_processes=Config.MAX_PROCESSES,
            )
            train_adv.to_csv(f"{workspace}/train_adv.csv", index=False)
            test_adv.to_csv(f"{workspace}/test_adv.csv", index=False)
            log(
                f"[adv_model] prepared adversarial data, with train_adv: {train_adv.shape[0]} test_adv: {test_adv.shape[0]}",
                3,
            )

            test_no_attacks = test[test["label"] == 0]
            log(f"Training data for adv: {test_no_attacks.shape[0]}, test_adv: {test_adv.shape[0]}", 2)
            log(f"Concat looks like this: {pd.concat([test_no_attacks, test_adv]).shape}", 2)

            # make adversarial model
            model_adv_trained, threshold_adv = train_model(
                train=pd.concat([train, train_adv])
                .sample(frac=1)
                .reset_index(drop=True),
                test=pd.concat([test_no_attacks, test_adv]),
                model=Config.MODEL_ADV,
                desired_fpr=Config.DESIRED_FPR,
                image_path=f"{workspace}/model_adv",
            )
            save_model(workspace, "model_adv", model_adv_trained, threshold_adv, True)
            log(f"[adv_model] adversarial model done", 2)
    else:
        model_adv_trained = None
        threshold_adv = None
    # ----------------- samples -----------------
    if Config.FIND_SAMPLES:
        if not has_tmp_samples_todos:
            os.makedirs(f"{workspace}/tmp_samples", exist_ok=True)
            os.makedirs(f"{workspace}/tmp_samples/todo", exist_ok=True)
            os.makedirs(f"{workspace}/tmp_samples/optimized", exist_ok=True)
            attacks = read_and_parse_sql(Config.ATTACK_DATA_PATH)
            attacks["label"] = 1
            prepare_batches_for_optimization(
                data_set=attacks,
                number=Config.SAMPLE_ATTEMPTS,
                batch_size=Config.BATCH_SIZE,
                data_path=workspace,
                tmp_dir="tmp_samples",
                label="sample",
            )

        log(f"[samples] start optimizing batches for sample creation...", 2)
        if not model_adv_trained and model_trained:
            log(f"[samples] using MODEL_TRAINED, not ADV_MODEL_TRAINED...", 3)
            model_adv_trained = model_trained  # simple solution in case this model has no adversarial training
            threshold_adv = threshold  # simple solution in case this model has no adversarial training
        _, _, samples = optimize_batches_in_todo(
            model_trained=model_adv_trained,
            engine_settings=Config.ENGINE_SETTINGS_SAMPLE_CREATION,
            rule_ids=Config.RULE_IDS,
            paranoia_level=Config.PARANOIA_LEVEL,
            max_processes=Config.MAX_PROCESSES,
            data_path=workspace,
            tmp_dir="tmp_samples",
        )
        log(f"[samples] optimized batches for sample creation", 2)
        # only keep samples with confidence below threshold_adv
        # print(samples["min_confidence"].dtype)
        samples = samples[samples["min_confidence"] < threshold_adv]

        if has_sample_adv_csv:
            samples.to_csv(
                f"{workspace}/sample_adv.csv", mode="a", header=False, index=False
            )
        else:
            samples.to_csv(f"{workspace}/sample_adv.csv", index=False)

        log(f"[samples] adversarial samples done, found {samples.shape[0]}", 3)

    remove_tmp_dirs(workspace)
