import joblib
import pandas as pd
import shap
import matplotlib.pyplot as plt

model = joblib.load("models/fraud_random_forest.pkl")

df = pd.read_csv("data/raw/creditcard.csv")

X = df.drop("Class", axis=1)

fraud_transaction = df[df["Class"] == 1].drop("Class", axis=1).iloc[[0]]

print(fraud_transaction)

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(fraud_transaction)

print(type(shap_values))
print(shap_values.shape)
print(shap_values[0])

fraud_shap = shap_values[0, :, 1]

shap_df = pd.DataFrame({
    "Feature": X.columns,
    "SHAP_Value": fraud_shap
})

print(
    shap_df.sort_values(
        by="SHAP_Value",
        ascending=False
    ).head(10)
)

sample_X = X.sample(1000, random_state=42)
sample_shap_values = explainer.shap_values(sample_X)

shap.summary_plot(
    sample_shap_values[:, :, 1],
    sample_X,
    show=False
)

plt.savefig(
    "images/shap_summary.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()