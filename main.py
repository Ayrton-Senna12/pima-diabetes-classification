import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, RocCurveDisplay
)

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, StackingClassifier

# ===============================
# 1. LOAD DATASET
# ===============================
df = pd.read_csv("diabetes.csv")

print(df.head())
print(df.info())

# ===============================
# 2. BASIC EDA
# ===============================
sns.countplot(x="Outcome", data=df)
plt.title("Class Distribution")
plt.show()

df.hist(figsize=(12, 8))
plt.tight_layout()
plt.show()

# ===============================
# 3. PREPROCESSING
# ===============================
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 80 / 10 / 10 split
X_train, X_temp, y_train, y_temp = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

# ===============================
# 4. METRIC FUNCTION
# ===============================
def evaluate_model(model, X, y, name):
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]

    print(f"\n{name} Results")
    print("Accuracy :", accuracy_score(y, y_pred))
    print("Precision:", precision_score(y, y_pred))
    print("Recall   :", recall_score(y, y_pred))
    print("F1-Score :", f1_score(y, y_pred))
    print("ROC-AUC  :", roc_auc_score(y, y_prob))

    cm = confusion_matrix(y, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"{name} Confusion Matrix")
    plt.show()

    RocCurveDisplay.from_predictions(y, y_prob)
    plt.title(f"{name} ROC Curve")
    plt.show()

# ===============================
# 5. INDIVIDUAL MODELS
# ===============================

# 1. Logistic Regression
lr_default = LogisticRegression(max_iter=1000)
lr_default.fit(X_train, y_train)
evaluate_model(lr_default, X_test, y_test, "Logistic Regression (Default)")

lr_tuned = LogisticRegression(C=0.1, solver="liblinear", max_iter=1000)
lr_tuned.fit(X_train, y_train)
evaluate_model(lr_tuned, X_test, y_test, "Logistic Regression (Tuned)")

# 2. KNN
knn_default = KNeighborsClassifier()
knn_default.fit(X_train, y_train)
evaluate_model(knn_default, X_test, y_test, "KNN (Default)")

knn_tuned = KNeighborsClassifier(n_neighbors=7, weights="distance")
knn_tuned.fit(X_train, y_train)
evaluate_model(knn_tuned, X_test, y_test, "KNN (Tuned)")

# 3. SVM
svm_default = SVC(probability=True)
svm_default.fit(X_train, y_train)
evaluate_model(svm_default, X_test, y_test, "SVM (Default)")

svm_tuned = SVC(C=1.0, kernel="rbf", gamma="scale", probability=True)
svm_tuned.fit(X_train, y_train)
evaluate_model(svm_tuned, X_test, y_test, "SVM (Tuned)")

# 4. Decision Tree
dt_default = DecisionTreeClassifier(random_state=42)
dt_default.fit(X_train, y_train)
evaluate_model(dt_default, X_test, y_test, "Decision Tree (Default)")

dt_tuned = DecisionTreeClassifier(max_depth=4, min_samples_split=10, random_state=42)
dt_tuned.fit(X_train, y_train)
evaluate_model(dt_tuned, X_test, y_test, "Decision Tree (Tuned)")

# 5. Random Forest
rf_default = RandomForestClassifier(random_state=42)
rf_default.fit(X_train, y_train)
evaluate_model(rf_default, X_test, y_test, "Random Forest (Default)")

rf_tuned = RandomForestClassifier(
    n_estimators=200, max_depth=6, random_state=42
)
rf_tuned.fit(X_train, y_train)
evaluate_model(rf_tuned, X_test, y_test, "Random Forest (Tuned)")

# ===============================
# 6. STACKING ENSEMBLE
# ===============================
base_models = [
    ("rf", rf_tuned),
    ("svm", svm_tuned)
]

meta_model = LogisticRegression(max_iter=1000)

stacking_model = StackingClassifier(
    estimators=base_models,
    final_estimator=meta_model,
    passthrough=False
)

stacking_model.fit(X_train, y_train)
evaluate_model(stacking_model, X_test, y_test, "Stacking Ensemble")