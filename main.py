import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load dataset
df = pd.read_csv("breast-cancer.csv")

# Features and target
X = df.drop("target", axis=1)   # assumes dataset has "target" column
y = df["target"]

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# --------------------------
# Linear Kernel SVM
# --------------------------
svm_linear = SVC(kernel="linear", C=1, random_state=42)
svm_linear.fit(X_train, y_train)
y_pred_linear = svm_linear.predict(X_test)

print("Linear SVM Accuracy:", accuracy_score(y_test, y_pred_linear))
print("\nClassification Report (Linear SVM):\n", classification_report(y_test, y_pred_linear))

# Confusion Matrix - Linear
cm_linear = confusion_matrix(y_test, y_pred_linear)
sns.heatmap(cm_linear, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix - Linear SVM")
plt.savefig("confusion_matrix_linear.png")
plt.close()

# --------------------------
# RBF Kernel SVM
# --------------------------
svm_rbf = SVC(kernel="rbf", C=1, gamma="scale", random_state=42)
svm_rbf.fit(X_train, y_train)
y_pred_rbf = svm_rbf.predict(X_test)

print("RBF SVM Accuracy:", accuracy_score(y_test, y_pred_rbf))
print("\nClassification Report (RBF SVM):\n", classification_report(y_test, y_pred_rbf))

# Confusion Matrix - RBF
cm_rbf = confusion_matrix(y_test, y_pred_rbf)
sns.heatmap(cm_rbf, annot=True, fmt="d", cmap="Oranges")
plt.title("Confusion Matrix - RBF SVM")
plt.savefig("confusion_matrix_rbf.png")
plt.close()

# --------------------------
# Hyperparameter Tuning (RBF)
# --------------------------
param_grid = {"C": [0.1, 1, 10], "gamma": [0.01, 0.1, 1]}
grid = GridSearchCV(SVC(kernel="rbf"), param_grid, cv=5, scoring="accuracy")
grid.fit(X_scaled, y)
print("Best Parameters (RBF):", grid.best_params_)
print("Best CV Score (RBF):", grid.best_score_)

# --------------------------
# Cross-validation
# --------------------------
cv_linear = cross_val_score(svm_linear, X_scaled, y, cv=5).mean()
cv_rbf = cross_val_score(svm_rbf, X_scaled, y, cv=5).mean()
print(f"Linear SVM CV Accuracy: {cv_linear:.4f}")
print(f"RBF SVM CV Accuracy: {cv_rbf:.4f}")

# --------------------------
# Decision Boundary (2D Visualization using first 2 features)
# --------------------------
X_vis = X_scaled[:, :2]
y_vis = y

X_train_vis, X_test_vis, y_train_vis, y_test_vis = train_test_split(X_vis, y_vis, test_size=0.2, random_state=42, stratify=y_vis)

svm_vis = SVC(kernel="rbf", C=1, gamma="scale")
svm_vis.fit(X_train_vis, y_train_vis)

x_min, x_max = X_vis[:, 0].min() - 1, X_vis[:, 0].max() + 1
y_min, y_max = X_vis[:, 1].min() - 1, X_vis[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                     np.arange(y_min, y_max, 0.02))

Z = svm_vis.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.figure(figsize=(8,6))
plt.contourf(xx, yy, Z, alpha=0.3, cmap="coolwarm")
sns.scatterplot(x=X_vis[:,0], y=X_vis[:,1], hue=y, palette="deep", edgecolor="k")
plt.title("SVM Decision Boundary (RBF, first 2 features)")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.savefig("svm_decision_boundary.png")
plt.close()
