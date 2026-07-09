import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

def main():
    print("Starting data preprocessing...")
    
    # Load data
    data_path = os.path.join('data', 'raw_credit_card_data.csv')
    df = pd.read_csv(data_path)
    
    # Standardize column names to lowercase for consistency
    df.columns = [c.lower() for c in df.columns]
    
    # Identify target
    target_candidates = [c for c in df.columns if 'default' in c or 'class' in c]
    if not target_candidates:
        raise ValueError("Could not find target column. Please check the column names.")
    target_col = target_candidates[-1]
    print(f"Using target column: {target_col}")
    
    # Convert target to 0/1 integers
    df[target_col] = df[target_col].astype(str)
    # The default class in this dataset is typically '1' or 'yes'
    # For safety, let's look at unique values
    unique_vals = df[target_col].unique()
    # Usually, 1 = default, 0 = not default
    if '1' in unique_vals and '0' in unique_vals:
        df[target_col] = df[target_col].astype(int)
    else:
        # Just map whatever is the minority class to 1
        val_counts = df[target_col].value_counts()
        minority_val = val_counts.idxmin()
        df[target_col] = (df[target_col] == minority_val).astype(int)
        
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Split the data BEFORE preprocessing to avoid data leakage
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Define features based on the typical dataset structure
    # Nominal categorical: sex, marriage
    # Ordinal categorical: education, pay_0 ... pay_6
    # Numerical: limit_bal, age, bill_amt*, pay_amt*
    
    # Automatically infer column types based on number of unique values if names are generic
    categorical_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
    numeric_cols = X_train.select_dtypes(include=['int64', 'float64', 'int32']).columns.tolist()
    
    # Wait, in the credit card dataset, 'sex', 'education', 'marriage', 'pay_*' are often numeric types but represent categories.
    # Let's enforce standard logic for this specific dataset if the columns exist.
    known_cat_cols = ['sex', 'education', 'marriage', 'pay_0', 'pay_2', 'pay_3', 'pay_4', 'pay_5', 'pay_6']
    actual_cat_cols = [c for c in known_cat_cols if c in X_train.columns]
    
    # For columns not in known_cat_cols, treat as numeric
    actual_num_cols = [c for c in X_train.columns if c not in actual_cat_cols]
    
    # We will use OneHotEncoding for all categories for simplicity and robustness, though education could be ordinal.
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), actual_num_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore', drop='first', sparse_output=False), actual_cat_cols)
        ]
    )
    
    print("Fitting preprocessor...")
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    # Get feature names after transformation
    cat_feature_names = preprocessor.named_transformers_['cat'].get_feature_names_out(actual_cat_cols)
    all_feature_names = actual_num_cols + list(cat_feature_names)
    
    print(f"Original shape: {X_train.shape}, Processed shape: {X_train_processed.shape}")
    
    print("Applying SMOTE to training data...")
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_processed, y_train)
    
    print(f"Resampled training shape: {X_train_resampled.shape}")
    print(f"Original class distribution: {np.bincount(y_train)}")
    print(f"Resampled class distribution: {np.bincount(y_train_resampled)}")
    
    # Save the processed data
    # Create pandas dataframes for easier handling later
    train_df = pd.DataFrame(X_train_resampled, columns=all_feature_names)
    train_df['target'] = y_train_resampled
    
    test_df = pd.DataFrame(X_test_processed, columns=all_feature_names)
    # y_test is a series, index must match
    test_df['target'] = y_test.values
    
    train_path = os.path.join('data', 'train_processed.csv')
    test_path = os.path.join('data', 'test_processed.csv')
    
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    # Save the preprocessor pipeline
    os.makedirs('models', exist_ok=True)
    joblib.dump(preprocessor, os.path.join('models', 'preprocessor.pkl'))
    
    print(f"Processed data saved to {train_path} and {test_path}")

if __name__ == "__main__":
    main()
