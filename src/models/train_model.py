import warnings
warnings.filterwarnings("ignore")

from sklearn.metrics import confusion_matrix

import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import classification_report

df = pd.read_csv("data/raw/creditcard.csv")

X = df.drop("Class", axis=1)

y = df["Class"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = LogisticRegression(class_weight="balanced")

model.fit(X_train, y_train)
predictions = model.predict(X_test)

probabilities = model.predict_proba(X_test)
print(probabilities[:10])

fraud_probabilities = probabilities[:, 1]
custom_predictions = (fraud_probabilities > 0.7).astype(int)

print(classification_report(y_test, custom_predictions))
custom_cm = confusion_matrix(y_test, custom_predictions)

print(custom_cm)

print(classification_report(y_test, predictions))

cm = confusion_matrix(y_test, predictions)

print(cm)