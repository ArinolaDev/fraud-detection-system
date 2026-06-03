import joblib

model = joblib.load("models/fraud_random_forest.pkl")


def predict_fraud(transaction):

    prediction = model.predict(transaction)[0]

    probability = model.predict_proba(transaction)[0][1]

    return prediction, probability