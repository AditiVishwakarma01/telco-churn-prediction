import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


# 1. Load data

data = pd.read_csv("telco_churn.csv")
print("Data shape:", data.shape)
print(data.head())


# 2. Basic cleaning
# Drop customerID (identifier, not a feature)
if "customerID" in data.columns:
    data = data.drop("customerID", axis=1)

# TotalCharges has spaces -> convert to numeric
data["TotalCharges"] = pd.to_numeric(
    data["TotalCharges"].replace(" ", np.nan), errors="coerce"
)

# Handle missing values in TotalCharges
data["TotalCharges"] = data["TotalCharges"].fillna(
    data["TotalCharges"].median())

# Convert target 'Churn' to 0/1
data["Churn"] = data["Churn"].map({"No": 0, "Yes": 1})

print("\nChurn value counts:\n", data["Churn"].value_counts())


# 3. Feature / target split

X = data.drop("Churn", axis=1)
y = data["Churn"]

# Separate numeric and categorical columns
numeric_features = X.select_dtypes(
    include=["int64", "float64"]).columns.tolist()
categorical_features = X.select_dtypes(
    include=["object", "bool"]).columns.tolist()

print("\nNumeric features:", numeric_features)
print("Categorical features:", categorical_features)


# 4. Preprocessing pipeline

numeric_transformer = Pipeline(
    steps=[("scaler", StandardScaler())]
)

categorical_transformer = Pipeline(
    steps=[("onehot", OneHotEncoder(handle_unknown="ignore"))]
)

preprocess = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)


# 5. Train-test split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

print("\nTrain size:", X_train.shape, "Test size:", X_test.shape)


# 6. Baseline model: Logistic Regression

log_reg_pipeline = Pipeline(
    steps=[
        ("preprocess", preprocess),
        ("model", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ]
)

log_reg_pipeline.fit(X_train, y_train)
y_pred_lr = log_reg_pipeline.predict(X_test)
y_proba_lr = log_reg_pipeline.predict_proba(X_test)[:, 1]

print("\n=== Logistic Regression Performance ===")
print("Accuracy:", accuracy_score(y_test, y_pred_lr))
print("Precision:", precision_score(y_test, y_pred_lr))
print("Recall:", recall_score(y_test, y_pred_lr))
print("F1-score:", f1_score(y_test, y_pred_lr))
print("ROC-AUC:", roc_auc_score(y_test, y_proba_lr))


# 7. Advanced model: Random Forest + GridSearchCV

rf_pipeline = Pipeline(
    steps=[
        ("preprocess", preprocess),
        ("model", RandomForestClassifier(
            random_state=42,
            class_weight="balanced",
            n_jobs=-1
        )),
    ]
)

param_grid = {
    "model__n_estimators": [100, 200],
    "model__max_depth": [None, 10, 20],
    "model__min_samples_split": [2, 5],
}

grid_search = GridSearchCV(
    rf_pipeline,
    param_grid,
    cv=3,
    scoring="roc_auc",
    n_jobs=-1,
    verbose=1,
)

grid_search.fit(X_train, y_train)

print("\nBest parameters from GridSearchCV:")
print(grid_search.best_params_)

best_rf_model = grid_search.best_estimator_

y_pred_rf = best_rf_model.predict(X_test)
y_proba_rf = best_rf_model.predict_proba(X_test)[:, 1]

print("\n=== Random Forest (Tuned) Performance ===")
print("Accuracy:", accuracy_score(y_test, y_pred_rf))
print("Precision:", precision_score(y_test, y_pred_rf))
print("Recall:", recall_score(y_test, y_pred_rf))
print("F1-score:", f1_score(y_test, y_pred_rf))
print("ROC-AUC:", roc_auc_score(y_test, y_proba_rf))

print("\nClassification Report (Random Forest):\n")
print(classification_report(y_test, y_pred_rf))


# 8. Confusion matrix heatmap (Random Forest)

cm = confusion_matrix(y_test, y_pred_rf)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix - Random Forest")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()


# 9. Feature importance (top 15 features)
# Extract feature names after one-hot encoding
preprocess_step = best_rf_model.named_steps["preprocess"]
ohe = preprocess_step.named_transformers_["cat"]["onehot"]
cat_feature_names = ohe.get_feature_names_out(categorical_features)

all_feature_names = list(numeric_features) + list(cat_feature_names)

rf = best_rf_model.named_steps["model"]
importances = rf.feature_importances_

feat_imp = pd.Series(
    importances, index=all_feature_names).sort_values(ascending=False)

top_n = 15
plt.figure(figsize=(8, 6))
sns.barplot(x=feat_imp.head(top_n), y=feat_imp.head(top_n).index)
plt.title(f"Top {top_n} Feature Importances (Random Forest)")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()
