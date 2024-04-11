import os
import joblib
import pandas as pd
from tqdm import tqdm

from src.modsec import init_modsec
from src.model import payload_to_vec, predict_vec
from src.utils import log


def save_results(
    target_workspace,
    surrogate_workspace,
    target_threshold,
    surrogate_threshold,
    samples_tested,
):
    target_workspace_dir = target_workspace.split("/")[-1]
    surrogate_workspace_dir = surrogate_workspace.split("/")[-1]

    results_dir = f"/app/wafcraft/results/target_{target_workspace_dir}"
    os.makedirs(results_dir, exist_ok=True)
    results_file = f"{results_dir}/transferability.csv"

    samples_tested_dir = f"{results_dir}/samples_tested"
    os.makedirs(samples_tested_dir, exist_ok=True)
    samples_file = f"{samples_tested_dir}/surrogate_{surrogate_workspace_dir}.csv"

    total_samples_count = len(samples_tested)
    samples_evaded_count = samples_tested["evaded"].sum()
    samples_evaded_percentage = samples_evaded_count / total_samples_count

    target_confidence_mean = samples_tested["target_confidence"].mean()
    target_confidence_original_mean = samples_tested[
        "target_confidence_original"
    ].mean()

    target_confidence_reduction_mean = target_confidence_original_mean - target_confidence_mean

    surrogate_confidence_mean = samples_tested["surrogate_confidence"].mean()
    surrogate_confidence_original_mean = samples_tested[
        "surrogate_confidence_original"
    ].mean()

    surrogate_confidence_reduction_mean = surrogate_confidence_original_mean - surrogate_confidence_mean

    

    description = ""
    with open(f"{surrogate_workspace}/config.txt", "r") as file:
        for line in file:
            if line.strip().startswith("DESCRIPTION:"):
                description = line.strip().split("DESCRIPTION:", 1)[1].strip()

    results = pd.DataFrame(
        {
            "surrogate_workspace": surrogate_workspace_dir,
            "description": description,
            "target_threshold": target_threshold,
            "surrogate_threshold": surrogate_threshold,
            "total_samples_count": total_samples_count,
            "samples_evaded_count": samples_evaded_count,
            "samples_evaded_percentage": samples_evaded_percentage,
            "target_confidence_mean": target_confidence_mean,
            "target_confidence_original_mean": target_confidence_original_mean,
            "surrogate_confidence_mean": surrogate_confidence_mean,
            "surrogate_confidence_original_mean": surrogate_confidence_original_mean,
            "target_confidence_reduction_mean": target_confidence_reduction_mean,
            "surrogate_confidence_reduction_mean": surrogate_confidence_reduction_mean,
        },
        index=[0],
    )

    if os.path.exists(results_file):
        results = pd.concat([pd.read_csv(results_file), results], ignore_index=True)
    results.to_csv(results_file, index=False)
    samples_tested.to_csv(samples_file, index=False)


def load_model_threshold(workspace, use_adv):
    if use_adv:
        model = joblib.load(f"{workspace}/model_adv/model_adv.joblib")
        threshold = float(open(f"{workspace}/model_adv/threshold_adv.txt", "r").read())
        log(f"loaded adv model for {workspace}", 2)
    else:
        model = joblib.load(f"{workspace}/model/model.joblib")
        threshold = float(open(f"{workspace}/model/threshold.txt", "r").read())
        log(f"loaded non-adv model for {workspace}", 2)
    return model, threshold


def test_transferability(
    Target_Config,
    target_workspace,
    surrogate_workspace,
    target_use_adv=True,
    surrogate_use_adv=True,
):
    # load models (surrogate_model, not needed)
    target_model, target_threshold = load_model_threshold(
        target_workspace, target_use_adv
    )
    surrogate_model, surrogate_threshold = load_model_threshold(
        surrogate_workspace, surrogate_use_adv
    )

    # load adversarial samples from the surrogate model
    samples = pd.read_csv(f"{surrogate_workspace}/sample_adv.csv")

    # test the transferability
    modsec = init_modsec()
    rule_ids = Target_Config.RULE_IDS
    paranoia_level = Target_Config.PARANOIA_LEVEL

    samples_tested = pd.DataFrame(
        columns=[
            "data",
            "original",
            "target_confidence",
            "target_confidence_original",
            "surrogate_confidence",
            "surrogate_confidence_original",
            "evaded",
        ]
    )
    for i, row in tqdm(
        samples.iterrows(), total=samples.shape[0], desc="Testing samples"
    ):
        # calculate target confidence of sample (original)
        payload_base64_original = row["original"]
        vec_original = payload_to_vec(
            payload_base64_original, rule_ids, modsec, paranoia_level
        )
        confidence_original = predict_vec(vec_original, target_model)
        # calculate target confidence of sample (optimized)
        payload_base64 = row["data"]
        vec = payload_to_vec(payload_base64, rule_ids, modsec, paranoia_level)
        confidence = predict_vec(vec, target_model)

        # save result
        samples_tested.loc[i] = [
            payload_base64,
            payload_base64_original,
            confidence,
            confidence_original,
            row["min_confidence"],
            row["original_confidence"],
            confidence < target_threshold,
        ]

    save_results(
        target_workspace,
        surrogate_workspace,
        target_threshold,
        surrogate_threshold,
        samples_tested,
    )

    log(f"results saved to \"results/target_{target_workspace.split('/')[-1]}\"", 2)
