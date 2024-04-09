import os
import joblib
import pandas as pd

from src.modsec import init_modsec
from src.model import payload_to_vec, predict_vec
from src.utils import log


def format_transferability_results(
    target_workspace,
    surrogate_workspace,
    target_threshold,
    surrogate_threshold,
    samples_tested,
):
    total_samples_count = len(samples_tested)
    samples_evaded_count = samples_tested["evaded"].sum()
    samples_evaded_percentage = samples_evaded_count / total_samples_count

    target_confidence_mean = samples_tested["target_confidence"].mean()
    surrogate_confidence_mean = samples_tested["surrogate_confidence"].mean()

    results = (
        "Transferability results:\n"
        f"Target: {target_workspace}\n"
        f"Surrogate: {surrogate_workspace}\n"
        f"Target confidence mean: {target_confidence_mean:.5f} (threshold: {target_threshold})\n"
        f"Surrogate confidence mean: {surrogate_confidence_mean:.5f} (threshold: {surrogate_threshold})\n"
        f"Samples evaded: {samples_evaded_count} out of {total_samples_count} ({samples_evaded_percentage:.2%})"
    )
    return results


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
        columns=["data", "target_confidence", "surrogate_confidence", "evaded"]
    )
    for i, row in samples.iterrows():
        payload_base64 = row["data"]
        vec = payload_to_vec(payload_base64, rule_ids, modsec, paranoia_level)
        is_attack = predict_vec(vec, target_model)
        samples_tested.loc[i] = [
            payload_base64,
            is_attack,
            row["min_confidence"],
            is_attack < target_threshold,
        ]

    # write evaded samples to a file
    target_workspace_dir = target_workspace.split("/")[-1]
    transferability_dir = f"{surrogate_workspace}/transferability_{target_workspace_dir}"
    os.makedirs(transferability_dir, exist_ok=True)
    samples_tested.to_csv(f"{transferability_dir}/samples_tested.csv", index=False)

    results = format_transferability_results(
        target_workspace,
        surrogate_workspace,
        target_threshold,
        surrogate_threshold,
        samples_tested,
    )
    with open(f"{transferability_dir}/results.txt", "w") as f:
        f.write(results)
    log(results, 2)
