# Codsoft-customer-churn-prediction
# Bank Customer Churn Prediction

An end-to-end Machine Learning pipeline developed in Python to predict whether a banking customer is at risk of churning (leaving the bank). This project evaluates three core classification algorithms to identify the optimal model for proactive business retention strategies.

## 📊 Dataset
The dataset utilized is the classic **Kaggle Bank Customer Churn dataset** (10,000 records). It contains demographic and financial attributes such as Credit Score, Age, Tenure, Balance, Number of Products, and Estimated Salary.

## ⚙️ Workflow & Architecture
1. **Data Preprocessing:** Automated handling of missing numbers (Median Imputation) and categorical variables (One-Hot Encoding).
2. **Feature Scaling:** Standardized numeric parameters using `StandardScaler` to ensure optimal gradient conversion.
3. **Model Evaluation:** Built pipelines comparing **Logistic Regression**, **Random Forest**, and **Gradient Boosting**.

## 📈 Results & Performance
The models achieved the following performance metrics on a 20% test slice (2,000 customers):

* **Overall Accuracy:** 87%
* **ROC-AUC Score:** ~0.8708
* **Precision (Class 1):** 0.79 (High reliability, minimizing false positives)

Ensemble tree-based methods significantly outperformed basic linear baselines by capturing complex customer behavioral thresholds.

## 🛠️ How to Run
1. Clone this repository.
2. Install dependencies: `pip install pandas numpy scikit-learn matplotlib`
3. Run the pipeline execution: `python CHURN.PY`
