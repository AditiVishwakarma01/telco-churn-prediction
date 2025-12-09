# Customer Churn Prediction – Machine Learning Project

This project predicts whether a customer is likely to churn based on their service usage and demographic data.  
It uses **real-world Telco customer data** and multiple machine learning models to identify people most at risk of leaving the service.

---

## Features

✔ Data Cleaning & Preprocessing  
✔ Categorical Encoding & Feature Scaling  
✔ Machine Learning Pipeline  
✔ Hyperparameter Tuning using GridSearchCV  
✔ Model Evaluation using ROC-AUC, Precision, Recall & Confusion Matrix  
✔ Feature Importance Analysis (Top Predictors of Churn)

---

##  Dataset

- **Dataset used:** Telco Customer Churn
- **Target Variable:** `Churn` (Yes/No)
- Includes demographics, billing, tenure and service subscription details.

---

##  Machine Learning Models Used

| Model | Purpose | Status |
|-------|---------|-------|
| Logistic Regression | Baseline model | ✓ |
| Random Forest Classifier | Tuned for best performance | ✓ |

**Best Model:** Tuned Random Forest  
✔ High ROC-AUC improvement  
✔ Better detection of actual churn customers

---

##  Results

| Metric | Logistic Regression | Random Forest (Tuned) |
|--------|-------------------|----------------------|
| Accuracy | ✓ | Higher |
| Recall | Good | **Better at catching churn** |
| ROC-AUC | Good | **Best** |

📌 Visual outputs included:
- Confusion Matrix
- Feature Importance Plot

---

##  Key Insights

- Tenure, Online Security, Contract Type & MonthlyCharges are the **strongest predictors**
- Customers with **month-to-month contracts** and **high monthly charges** churn most often
- Longer subscribed customers are **less likely to leave**

---

##  Tech Stack

- Python
- Scikit-learn
- Pandas
- NumPy
- Seaborn / Matplotlib





