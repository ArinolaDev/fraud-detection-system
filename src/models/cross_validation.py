import pandas as pd

from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv("data/raw/creditcard.csv")

X = df.drop("Class", axis=1)
y = df["Class"]

model = RandomForestClassifier(
    random_state=42,
    class_weight="balanced"
)

scores = cross_val_score(
    model,
    X,
    y,
    cv=5,
    scoring="f1"
)

print(scores)
print(scores.mean())
print("Standard Deviation:", scores.std())