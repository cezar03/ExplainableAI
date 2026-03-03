import openml
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

DATASET_ID = 31 

dataset = openml.datasets.get_dataset(DATASET_ID)

print("Dataset name:", dataset.name)
print("Dataset ID:", dataset.dataset_id)
print("Dataset version:", dataset.version)
print("Default target:", dataset.default_target_attribute)
print("-" * 50)

X, y, categorical_indicator, feature_names = dataset.get_data(
    dataset_format="dataframe",
    target=dataset.default_target_attribute
)

num_X = X.select_dtypes(include="number")

print("Numerical features:", list(num_X.columns))
print("Shape (numerical only):", num_X.shape)


num_X = num_X.dropna()

corr = num_X.corr()

print("\nTop absolute correlations (excluding self-correlation):")
corr_pairs = (
    corr.abs()
    .unstack()
    .sort_values(ascending=False)
)

corr_pairs = corr_pairs[corr_pairs < 1.0]
print(corr_pairs.head(15))

plt.figure(figsize=(10, 8))
sns.heatmap(corr, cmap="coolwarm", center=0, annot=False)
plt.title("Correlation Heatmap (Numerical Features)")
plt.tight_layout()
plt.show()
from statsmodels.stats.outliers_influence import variance_inflation_factor

constant_cols = [c for c in num_X.columns if num_X[c].nunique() <= 1]
if constant_cols:
    print("\nRemoving constant columns for VIF:", constant_cols)
    num_X_vif = num_X.drop(columns=constant_cols)
else:
    num_X_vif = num_X

vif_df = pd.DataFrame()
vif_df["feature"] = num_X_vif.columns
vif_df["VIF"] = [
    variance_inflation_factor(num_X_vif.values, i)
    for i in range(num_X_vif.shape[1])
]

vif_df = vif_df.sort_values("VIF", ascending=False)

print("\nVIF results (higher = more multicollinearity):")
print(vif_df)

print("\nFeatures with VIF > 5 (problematic):")
print(vif_df[vif_df["VIF"] > 5])

print("\nFeatures with VIF > 10 (strong multicollinearity):")
print(vif_df[vif_df["VIF"] > 10])