import joblib

model = joblib.load("models/fraud_random_forest.pkl")

print(model)