# Telecom Customer Churn Prediction

## Overview
This project focuses on predicting customer churn in the telecom industry using supervised machine learning models.  
The goal is to identify customers likely to discontinue service and understand the key factors influencing churn.

## Dataset
- Public telecom customer churn dataset
- Includes demographic details, service usage, contract type, billing information, and churn labels

## Methodology
1. Data cleaning and preprocessing
2. Encoding categorical variables
3. Feature scaling
4. Train-test split
5. Model training and evaluation

## Models Used
- Logistic Regression
- Random Forest Classifier

## Evaluation Metrics
- Accuracy
- Precision, Recall, F1-score
- Confusion Matrix

## Results
- Random Forest outperformed Logistic Regression
- Clear separation between churn and non-churn customers
- Feature importance highlights contract type, tenure, and monthly charges as major churn drivers

## Visualizations
- Confusion matrix
- Feature importance plots (Top 15 features)

## How to Run
```bash
pip install -r requirements_telco.txt
python telco_churn.py
