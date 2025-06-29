# Customer Churn Prediction

## 📋 Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Model Performance](#model-performance)
- [Results](#results)
- [Technologies Used](#technologies-used)
- [Project Structure](#project-structure)
- [License](#license)

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

## 🚀 Installation

### Prerequisites
- Python 3.7+
- pip package manager

### Required Libraries
```bash
pip install pandas numpy matplotlib seaborn
pip install scikit-learn xgboost 
pip install jupyter notebook
```

### Clone Repository
```bash
git clone https://github.com/viraftryn/2602068706_ChurnPred.git
cd 2602068706_ChurnPred
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

## 💻 Usage

### 1. Data Preparation
```python
# Load and preprocess the data
python data_preprocessing.py
```

### 2. Exploratory Data Analysis
```python
# Run EDA notebook
jupyter notebook EDA.ipynb
```

### 3. Model Training
```python
# Train multiple models
python train_models.py

# Or train specific model
python train_models.py --model random_forest
```

### 4. Model Evaluation
```python
# Evaluate all models
python evaluate_models.py

# Generate predictions
python predict.py --input data/test_data.csv --output predictions.csv
```

### 5. Run Complete Pipeline
```python
# Execute the entire pipeline
python main.py
```

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
- **Version Control:** Git

## 📁 Project Structure

```
2602068706_ChurnPred/
├── data/
│   ├── raw/                    # Original dataset
│   ├── processed/              # Cleaned and preprocessed data
│   └── predictions/            # Model predictions
├── notebooks/
│   ├── EDA.ipynb              # Exploratory Data Analysis
│   ├── preprocessing.ipynb     # Data preprocessing
│   ├── modeling.ipynb         # Model development
│   └── evaluation.ipynb       # Model evaluation
├── src/
│   ├── data_preprocessing.py   # Data cleaning functions
│   ├── feature_engineering.py # Feature creation functions
│   ├── models.py              # ML model implementations
│   ├── evaluation.py          # Model evaluation metrics
│   └── utils.py               # Utility functions
├── models/
│   ├── trained_models/        # Serialized trained models
│   └── model_configs/         # Model configuration files
├── results/
│   ├── figures/               # Generated plots and charts
│   ├── reports/               # Analysis reports
│   └── metrics/               # Model performance metrics
├── requirements.txt           # Python dependencies
├── README.md                  # Project documentation
├── main.py                    # Main execution script
└── config.py                  # Configuration settings
```
