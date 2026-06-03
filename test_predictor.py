import pandas as pd

from src.utils.predictor import predict_fraud

df = pd.read_csv("data/raw/creditcard.csv")

transaction = df.drop("Class", axis=1).iloc[[0]]

prediction, probability = predict_fraud(transaction)

print(prediction)
print(probability)