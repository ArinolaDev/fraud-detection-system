import joblib
import pandas as pd

model = joblib.load("models/fraud_random_forest.pkl")

df = pd.read_csv("data/raw/creditcard.csv")

print(df.head())

transaction = df[df["Class"] == 1].drop("Class", axis=1).iloc[[0]]

print(transaction)

prediction = model.predict(transaction)

print(prediction)

probability = model.predict_proba(transaction)

print(probability)

fraud_probability = probability[0][1]

print(f"Fraud Risk Score: {fraud_probability:.2%}")