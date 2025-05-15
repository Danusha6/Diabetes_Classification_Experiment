# Diabetes & Heart Disease Classification
This project applies machine learning models to two medical classification problems: predicting diabetes and heart disease. The workflow includes data cleaning, standardization, PCA, SMOTE for class balancing, model evaluation, and visualization.

## Datasets
- `diabetes.csv`: From Pima Indian Diabetes Dataset
- `heart.csv`: Synthetic dataset based on UCI Heart Disease data

## Models Used
- K-Nearest Neighbors (KNN)
- Naive Bayes
- Logistic Regression
- Decision Tree
- Support Vector Machine (SVM)

## ML Pipeline
1. Load and explore data
2. Handle missing/invalid values
3. Encode categorical features (for heart dataset)
4. Standardize numerical values
5. Apply PCA (reduce to 5 components)
6. Train-test split (80/20)
7. Use SMOTE to balance training data
8. Train 5 classification models
9. Evaluate using CV F1 scores, confusion matrix, classification report
10. Visualize correlations, distributions, and confusion matrices