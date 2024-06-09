import pandas as pd
import numpy as np
import joblib
import base64
import os

from tqdm import tqdm
from sklearn.metrics import confusion_matrix

from src.model import create_wafamole_model
from src.modsec import init_modsec
from src.utils import load_data_label_vector, read_and_parse_sql

# This script rechecks the sample tansfer successrate

target = "2024-04-07_18-15-53_brown-lot"
# surrogates = [ # 0% overlap
#     "2024-04-18_14-12-51_lightblue-around",
#     "2024-05-10_15-03-09_darkred-number",
#     "2024-04-22_11-20-36_yellow-majority",
#     "2024-05-10_23-07-35_beige-western",
#     "2024-04-23_05-02-11_cadetblue-right",
#     "2024-04-08_21-57-36_greenyellow-fear",
#     "2024-05-11_07-05-20_honeydew-check",
#     "2024-04-23_02-58-14_blanchedalmond-table",
#     "2024-04-22_18-57-13_darkslateblue-air",
#     "2024-05-11_14-57-56_darkgray-general",
# ]
# surrogates = [  # 25% overlap
#     "2024-04-24_15-31-15_deeppink-special",
#     "2024-04-09_14-07-59_maroon-scene",
#     "2024-05-12_22-17-21_gainsboro-debate",
#     "2024-05-12_06-49-07_aliceblue-itself",
#     "2024-05-11_22-37-23_darkkhaki-those",
#     "2024-05-12_14-29-35_blue-tell",
#     "2024-04-23_14-54-40_darkkhaki-analysis",
#     "2024-04-23_12-52-13_blue-cover",
#     "2024-04-24_07-26-56_sandybrown-rich",
#     "2024-04-23_23-09-12_mediumslateblue-imagine",
# ]
surrogates = [  # 100% overlap
    "2024-04-28_07-18-23_lightcoral-pretty",
    "2024-04-28_22-59-40_blueviolet-physical",
    "2024-05-15_22-43-02_lightseagreen-first",
    "2024-05-15_06-47-54_mediumpurple-nothing",
    "2024-05-14_22-20-02_lightpink-medical",
    "2024-05-15_14-55-23_red-resource",
    "2024-04-10_13-36-12_darkcyan-why",
    "2024-04-29_15-29-39_darkkhaki-quickly",
    "2024-04-29_07-05-06_darkslategray-approach",
    "2024-04-28_15-07-53_aliceblue-suddenly",
]
use_adv_model = True  # Whether to use the adversarial model
test_description = "recheck 0% data overlap transfer"

rule_ids = [
    "942011",
    "942012",
    "942013",
    "942014",
    "942015",
    "942016",
    "942017",
    "942018",
    "942100",
    "942101",
    "942110",
    "942120",
    "942130",
    "942131",
    "942140",
    "942150",
    "942151",
    "942152",
    "942160",
    "942170",
    "942180",
    "942190",
    "942200",
    "942210",
    "942220",
    "942230",
    "942240",
    "942250",
    "942251",
    "942260",
    "942270",
    "942280",
    "942290",
    "942300",
    "942310",
    "942320",
    "942321",
    "942330",
    "942340",
    "942350",
    "942360",
    "942361",
    "942362",
    "942370",
    "942380",
    "942390",
    "942400",
    "942410",
    "942420",
    "942421",
    "942430",
    "942431",
    "942432",
    "942440",
    "942450",
    "942460",
    "942470",
    "942480",
    "942490",
    "942500",
    "942510",
    "942511",
    "942520",
    "942521",
    "942522",
    "942530",
    "942540",
    "942550",
    "942560",
]


def load_model(use_adv_model):
    model_type = "model_adv" if use_adv_model else "model"
    threshold_name = "threshold_adv" if use_adv_model else "threshold"

    workspace = "/app/wafcraft/data/prepared/2024-04-07_18-15-53_brown-lot"
    threshold = float(
        open(f"{workspace}/{model_type}/{threshold_name}.txt", "r").read()
    )

    modsec = init_modsec()
    model = joblib.load(f"{workspace}/{model_type}/{model_type}.joblib")
    wafamole_model = create_wafamole_model(model, modsec, rule_ids, 4)
    return wafamole_model, threshold


def evaluate_samples(wafamole_model, threshold, samples):
    attacks_identified = 0
    for i, row in tqdm(samples.iterrows(), total=len(samples)):
        payload_b64 = row["data"]
        payload = base64.b64decode(payload_b64)
        confidence_is_attack = wafamole_model.classify(payload.decode("utf-8"))
        if confidence_is_attack > threshold:
            attacks_identified += 1
    return attacks_identified


def main():
    for surrogate in surrogates:
        samples = pd.read_csv(f"data/prepared/{surrogate}/sample_adv.csv")
        print(f"Evaluating {surrogate} -> {target}")
        wafamole_model, threshold = load_model(use_adv_model)
        attacks_identified = evaluate_samples(wafamole_model, threshold, samples)

        if not os.path.exists("results/sanity_check_transferability.csv"):
            with open("results/sanity_check_transferability.csv", "w") as f:
                f.write(
                    "surrogate,total_samples,attacks_identified,transfer_success_rate\n"
                )

        with open("results/sanity_check_transferability.csv", "a") as f:
            f.write(
                f"{surrogate},{len(samples)},{attacks_identified},{1-(attacks_identified/len(samples))}\n"
            )


if __name__ == "__main__":
    main()
