# Credit Card Client Risk Prediction

## Objective
This project implements a robust predictive risk modeling pipeline to identify potential credit card default and delinquency. By accurately predicting high-risk clients, financial institutions can proactively mitigate potential losses and promote responsible credit allocation.

## Dataset
The dataset utilized is the standard "Default of Credit Card Clients" dataset, containing 30,000 instances with 23 features representing payment history, demographic factors, credit limits, and historical bill amounts.

## Pipeline Architecture

### 1. Data Acquisition
The dataset is programmatically fetched from the OpenML repository (Dataset ID: 42477) and stored locally for reproducibility. The generic feature names are mapped to their standard domain-specific counterparts (e.g., `limit_bal`, `age`, `pay_0`).

### 2. Exploratory Data Analysis (EDA)
An initial analysis highlights the severe class imbalance in the target variable, where defaults comprise only ~22% of the total dataset. EDA also visualizes correlation amongst the numeric features to better understand multicollinearity.

### 3. Data Preprocessing
The preprocessing routine is encapsulated in a scikit-learn `ColumnTransformer` to prevent data leakage and streamline transformations:
- **Feature Scaling**: Applied `StandardScaler` to numerical columns for improved stability in modeling.
- **Categorical Encoding**: Applied `OneHotEncoder` to categorical and nominal features.
- **Class Imbalance Mitigation**: Utilized **SMOTE** (Synthetic Minority Over-sampling Technique) strictly on the training set to synthetically balance the representation of the default class.

### 4. Modeling & Benchmarking
The project trains and benchmarks two tree-based ensemble models known for their robustness on tabular data:
- **Random Forest Classifier**
- **XGBoost Classifier**

Both models are trained on the SMOTE-resampled training data to maximize recall for the minority class without drastically compromising precision.

## Results
The models were evaluated on a hold-out test set using F1-score and ROC-AUC as primary metrics.
- **Random Forest**: Achieved an F1-Score of **0.53** and ROC-AUC of **0.77**.
- **XGBoost**: Achieved an F1-Score of **0.51** and ROC-AUC of **0.76**.

These metrics reflect a highly competent diagnostic capability on a realistically noisy and imbalanced financial dataset.

## How to Run
1. Create and activate a Python virtual environment.
2. Install dependencies: `pip install pandas numpy scikit-learn xgboost imbalanced-learn matplotlib seaborn fpdf2`
3. Execute the pipeline scripts in order:
   - `python fetch_data.py`
   - `python eda.py`
   - `python preprocess.py`
   - `python train.py`
