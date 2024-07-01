import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_curve

from src.utils import load_data_label_vector


workspaces = {
    "Surrogate_Data_V5": [
        "2024-04-28_07-18-23_lightcoral-pretty",
        "2024-04-29_15-29-39_darkkhaki-quickly",
        "2024-04-28_22-59-40_blueviolet-physical",
        "2024-04-28_15-07-53_aliceblue-suddenly",
        "2024-05-14_22-20-02_lightpink-medical",
        "2024-04-10_13-36-12_darkcyan-why",
    ],
    "Surrogate_GBoost_V1": [
        "2024-05-23_06-06-40_darkslateblue-they",
        "2024-05-22_13-56-59_darkblue-until",
        "2024-05-23_21-59-14_palegreen-local",
        "2024-05-22_22-04-52_papayawhip-heavy",
        "2024-05-23_14-03-55_peru-civil",
        "2024-05-22_05-37-59_darkviolet-but",
    ],
    "Surrogate_KNN_V1": [
        "2024-05-27_15-25-51_peachpuff-from",
        "2024-05-27_12-31-57_darkorchid-involve",
        "2024-05-27_09-13-03_seashell-believe",
        "2024-05-27_18-35-07_beige-indicate",
        "2024-05-27_21-38-23_papayawhip-sound",
        "2024-05-28_00-47-49_mediumblue-phone",
    ],
    "Surrogate_LogReg_V1": [
        "2024-05-25_07-38-08_thistle-station",
        "2024-05-25_16-01-13_honeydew-million",
        "2024-05-27_00-49-34_aqua-gas",
        "2024-05-26_08-50-46_darkmagenta-price",
        "2024-05-26_00-32-57_mediumvioletred-open",
        "2024-05-26_16-57-36_mediumslateblue-agency",
    ],
    "Surrogate_NaiveBayes_V1": [
        "2024-05-24_21-29-40_chocolate-yourself",
        "2024-05-24_08-27-38_lightskyblue-might",
        "2024-05-24_10-34-12_darkmagenta-data",
        "2024-05-24_06-02-52_cadetblue-lay",
        "2024-05-25_05-33-38_darkslateblue-even",
        "2024-05-24_19-15-34_olive-simple",
    ],
    "Surrogate_SVM_V1": [
        "2024-05-21_21-37-36_seagreen-together",
        "2024-05-20_21-11-36_linen-civil",
        "2024-05-20_12-27-17_plum-television",
        "2024-05-21_04-59-27_peachpuff-perform",
        "2024-05-21_13-37-58_mediumvioletred-worker",
        "2024-04-08_10-49-51_paleturquoise-nor",
    ],
}


for config in workspaces:
    print(f"Config: {config}")
    tprs = []
    fprs = []
    for ws in workspaces[config]:
        model = joblib.load(
            f"/app/wafcraft/data/prepared/{ws}/model_adv/model_adv.joblib"
        )

        test_main = load_data_label_vector(f"/app/wafcraft/data/prepared/{ws}/test.csv")
        test_main_no_attacks = test_main[test_main["label"] == 0]
        test_adv = load_data_label_vector(
            f"/app/wafcraft/data/prepared/{ws}/test_adv.csv"
        )

        test = pd.concat([test_main_no_attacks, test_adv])

        # predict vector
        probs = model.predict_proba(test["vector"].tolist())

        # calculate tpr at fpr 0.01
        desired_fpr = 0.01
        fpr, tpr, thresholds = roc_curve(test["label"], probs[:, 1])
        closest_idx = np.argmin(np.abs(fpr - desired_fpr))
        threshold = thresholds[closest_idx]

        predictions = (probs[:, 1] >= threshold).astype(int)

        tn, fp, fn, tp = confusion_matrix(test["label"], predictions).ravel()
        tpr_actual = tp / (tp + fn)
        fpr_actual = fp / (fp + tn)

        tprs.append(tpr_actual)
        fprs.append(fpr_actual)

        print(f"  Workspace: {ws} TPR: {tpr_actual}, FPR: {fpr_actual}")

    
    print(f"Mean FPR: {np.mean(fprs)*100}, Mean TPR: {np.mean(tprs)*100}")
    print()
