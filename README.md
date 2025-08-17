# Task 7: Support Vector Machines (SVM) (Breast Cancer Dataset)

## Objective
Use SVM for binary classification with linear and RBF kernels.

## Steps Done
1. Loaded and normalized dataset
2. Trained **Linear SVM**
3. Trained **RBF SVM**
4. Visualized **Confusion Matrices**
5. Tuned hyperparameters (`C`, `gamma`) using `GridSearchCV`
6. Evaluated with **Cross-validation**
7. Visualized **Decision Boundaries** using first 2 features

## Results
- Linear and RBF kernels compared
- RBF often performs better due to non-linear decision boundaries
- Hyperparameter tuning improves model performance
- Visualization shows clear separation with RBF kernel

## Files
- `main.py` → Main code
- `breast-cancer.csv` → Dataset
- `confusion_matrix_linear.png` → Confusion matrix (Linear SVM)
- `confusion_matrix_rbf.png` → Confusion matrix (RBF SVM)
- `svm_decision_boundary.png` → Decision boundary plot
- `requirements.txt` → Dependencies
