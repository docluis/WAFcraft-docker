import base64

import numpy as np

from wafamole.models import Model  # type: ignore
from sklearn.metrics import (
    classification_report,
    roc_curve,
    confusion_matrix,
)

from wafamole.evasion import EvasionEngine  # type: ignore
from src.modsec import get_activated_rules
from src.utils import (
    log,
    plot_cm,
    plot_roc,
    plot_precision_recall_curve,
)


def payload_to_vec(payload_base64, rule_ids, modsec, paranoia_level):
    """
    Returns a vectorized representation of a payload based on the activated rules

    Parameters:
        payload_base64 (str): Base64-encoded payload
        rule_ids (list): List of rule IDs
        modsec (modsecurity.ModSecurity): ModSecurity instance
        paranoia_level (int): Paranoia level

    Returns:
        numpy.ndarray: Vectorized payload
    """

    matches = get_activated_rules(
        payloads_base64=[payload_base64], modsec=modsec, paranoia_level=paranoia_level
    )
    # rule_array as numpy array of 0s and 1s
    rule_array = [1 if int(rule_id) in set(matches) else 0 for rule_id in rule_ids]
    return np.array(rule_array)


def train_model(train, test, model, desired_fpr):
    """
    Returns a trained model and the threshold for the desired FPR

    Parameters:
        train (pd.DataFrame): Train dataframe
        test (pd.DataFrame): Test dataframe
        model: Model to train
        desired_fpr (float): Desired false positive rate

    Returns:
        model: Trained Model, float: Trained model and threshold
    """

    threshold = 0.5
    # Extract features and labels
    X_train, y_train = list(train["vector"]), train["label"]
    X_test, y_test = list(test["vector"]), test["label"]

    # Train the model
    model.fit(X_train, y_train)
    log("Model trained successfully!")

    # Evaluate the model
    log("Evaluating model...")
    log(f"Default threshold: {threshold}")
    # calculate FPR = FP / (FP + TN)
    current_fpr = confusion_matrix(y_test, model.predict(X_test))[0, 1] / (
        confusion_matrix(y_test, model.predict(X_test))[0, 1]
        + confusion_matrix(y_test, model.predict(X_test))[1, 1]
    )

    log(f"FRP is currently at {round(current_fpr, 4)}")
    predictions = model.predict(X_test)
    log(classification_report(y_test, predictions))

    cm = confusion_matrix(y_test, predictions)
    plot_cm(cm)

    if desired_fpr is not None:
        log(f"Adjusting threshold to match desired FPR of {desired_fpr}")
        # 'attack' is considered the positive class (1) and 'sane' is the negative class (0)
        probabilities = model.predict_proba(X_test)[:, 1]

        fpr, tpr, thresholds = roc_curve(y_test, probabilities)  # plot ROC curve
        closest_idx = np.argmin(np.abs(fpr - desired_fpr))  # threshold closest to FPR
        threshold = thresholds[closest_idx]
        adjusted_predictions = (probabilities >= threshold).astype(int)  #  new preds

        plot_roc(fpr, tpr, closest_idx, desired_fpr)
        plot_precision_recall_curve(y_test, probabilities)

        log(
            f"Adjusted threshold: {round(threshold, 4)} with FPR of {round(fpr[closest_idx], 4)} (closest to desired FPR {desired_fpr})"
        )
        log(classification_report(y_test, adjusted_predictions))

        cm = confusion_matrix(y_test, adjusted_predictions)
        plot_cm(cm)

    return model, threshold


def create_wafamole_model(
    model,
    modsec,
    rule_ids,
    paranoia_level,
):
    """
    Returns a WAFamole model

    Parameters:
        model (sklearn.ensemble.RandomForestClassifier): Trained model
        modsec (modsecurity.ModSecurity): ModSecurity instance
        rule_ids (list): List of rule IDs
        paranoia_level (int): Paranoia level

    Returns:
        wafamole.models.Model: WAFamole model
    """

    def predict_vec(vec, model):
        probs = model.predict_proba([vec])[0]
        attack_index = list(model.classes_).index(1)
        confidence = probs[attack_index]
        return confidence

    class WAFamoleModel(Model):
        def extract_features(self, value: str):
            payload_base64 = base64.b64encode(value.encode("utf-8")).decode("utf-8")
            return payload_to_vec(
                payload_base64=payload_base64,
                rule_ids=rule_ids,
                modsec=modsec,
                paranoia_level=paranoia_level,
            )

        def classify(self, value: str):
            vec = self.extract_features(value)
            return predict_vec(
                vec=vec,
                model=model,
            )

    wafamole_model = WAFamoleModel()
    return wafamole_model


def test_evasion(
    payload,
    threshold,
    engine_eval_settings,
    model,
    rule_ids,
    modsec,
    paranoia_level,
):
    wafamole_model = create_wafamole_model(model, modsec, rule_ids, paranoia_level)
    engine = EvasionEngine(wafamole_model)
    payload_base64 = base64.b64encode(payload.encode("utf-8")).decode("utf-8")
    vec = payload_to_vec(payload_base64, rule_ids, modsec, paranoia_level)
    is_attack = wafamole_model.classify(payload)
    log(f"Payload: {payload}")
    log(f"Vec: {vec}")
    log(f"Confidence: {round(is_attack, 5)}")

    min_confidence, min_payload = engine.evaluate(
        payload=payload,
        **engine_eval_settings,
    )
    log(f"Min payload: {min_payload.encode('utf-8')}")
    log(f"Min confidence: {round(min_confidence, 5)}")
    log(
        f"Reduced confidence from {round(is_attack, 5)} to {round(min_confidence, 5)} (reduction of {round(is_attack - min_confidence, 5)})"
    )

    log("\nEvasion successful" if min_confidence < threshold else "Evasion failed")
