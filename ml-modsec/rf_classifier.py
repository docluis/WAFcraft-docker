from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from ast import literal_eval


def generate_rf_model(train_data, test_data):
    # convert the 'vector' column, only when it is a string
    # train_data['vector'] = train_data['vector'].apply(literal_eval)
    # test_data['vector'] = test_data['vector'].apply(literal_eval)

    # splitting the datasets into features (X) and target label (y)
    X_train = list(train_data["vector"])
    y_train = train_data["label"]
    X_test = list(test_data["vector"])
    y_test = test_data["label"]

    # create and train the Random Forest model
    # number of trees is set to 160 for PLs other than PL1 as per the paper
    rf_model = RandomForestClassifier(n_estimators=160, random_state=42)
    rf_model.fit(X_train, y_train)

    print("Model trained successfully!")

    # evaluating model performance on test data
    print(classification_report(y_test, rf_model.predict(X_test)))
    return rf_model
