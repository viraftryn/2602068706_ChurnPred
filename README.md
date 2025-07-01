# Customer Churn Prediction

## 📋 Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Features](#features)
- [Model Performance](#model-performance)
- [Results](#results)
- [Technologies Used](#technologies-used)

## 🎯 Overview

This project was developed as part of my academic work to explore predictive modeling techniques in data science. This project focuses on predicting customer churn using machine learning techniques. Customer churn prediction is crucial for businesses to identify customers who are likely to stop using their services, enabling proactive retention strategies.

### Objectives
- Analyze customer behavior patterns to identify churn indicators
- Build and compare multiple machine learning models for churn prediction
- Provide actionable insights for customer retention strategies
- Achieve high accuracy in predicting customer churn

## 📊 Dataset

### Data Source
- **Dataset:** the dataset is given by lecturer (data_D.csv)
- **Size:** 41.259 records, 14 features
- **Target Variable:** Churn (Binary: 1 = Churned, 0 = Retained)

### Key Features
- **Demographics:** Age, Gender, Location
- **Account Information:** Tenure, Contract type, Payment method
- **Service Usage:** Monthly charges, Total charges, Service subscriptions
- **Customer Service:** Number of complaints, Support tickets

## ✨ Features

### Data Preprocessing
- ✅ Data splitting into train and test data
- ✅ Converting categorical data into numerical data
- ✅ Feature encoding (One-hot, Label encoding)
- ✅ Handling missing values
- ✅ Drop unused features
- ✅ Feature scaling and normalization

### Exploratory Data Analysis
- ✅ Statistical summary and distributions
- ✅ Correlation analysis
- ✅ Churn rate analysis by different segments
- ✅ Visualization of key patterns and trends

### Machine Learning Models
- ✅ Random Forest
- ✅ Gradient Boosting (XGBoost)

### Tunning Hyperparameter Method
- ✅ GridSearchCV

### Model Evaluation
- ✅ Classification report


## 📈 Model Performance

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Random Forest | 0.84 | 0.65 | 0.55 | 0.59 |
| XGBoost | 0.85 | 0.66 | 0.59 | 0.63 |

**Best Performing Model:** XGBoost with 85% accuracy score

## 🔍 Results

### Key Findings
1. **Top Churn Indicators:**
   - Contract type (Month-to-month contracts have higher churn)
   - Tenure (Customers with less than 6 months tenure)
   - Monthly charges (Higher charges correlate with increased churn)
   - Customer service calls (Frequent complaints indicate dissatisfaction)

2. **Business Insights:**
   - Customers with electronic check payments are more likely to churn
   - Senior citizens show higher churn rates
   - Fiber optic internet users have elevated churn probability

3. **Actionable Recommendations:**
   - Implement retention campaigns for high-risk customers
   - Improve customer service quality
   - Offer incentives for annual contracts
   - Develop targeted pricing strategies

### Visualizations
- Churn distribution analysis
- Feature importance plots
- ROC curves comparison
- Customer segmentation analysis

## 🛠 Technologies Used

- **Programming Language:** Python 3.8+
- **Data Manipulation:** Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn
- **Machine Learning:** Scikit-learn, XGBoost
- **Development Environment:** Jupyter Notebook
