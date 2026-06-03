import joblib
import warnings
warnings.filterwarnings("ignore")

import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

df = pd.read_csv("data/raw/creditcard.csv")

X = df.drop("Class", axis=1)

y = df["Class"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight="balanced"
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

print(classification_report(y_test, predictions))

cm = confusion_matrix(y_test, predictions)

print(cm)

feature_importances = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
})

feature_importances = feature_importances.sort_values(
    by="Importance",
    ascending=False
)

print(feature_importances.head(10))
joblib.dump(model, "models/fraud_random_forest.pkl")