import os
import pandas as pd
from sklearn.datasets import fetch_openml

def main():
    print("Fetching dataset 'default of credit card clients' from OpenML...")
    # OpenML ID 42477 is the default of credit card clients dataset
    data = fetch_openml(data_id=42477, as_frame=True)
    df = data.frame
    
    # Rename columns to actual names if they are generic
    if len(df.columns) == 24:
        df.columns = [
            'limit_bal', 'sex', 'education', 'marriage', 'age',
            'pay_0', 'pay_2', 'pay_3', 'pay_4', 'pay_5', 'pay_6',
            'bill_amt1', 'bill_amt2', 'bill_amt3', 'bill_amt4', 'bill_amt5', 'bill_amt6',
            'pay_amt1', 'pay_amt2', 'pay_amt3', 'pay_amt4', 'pay_amt5', 'pay_amt6',
            'default'
        ]
    
    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    
    # Save the raw data
    output_path = os.path.join('data', 'raw_credit_card_data.csv')
    df.to_csv(output_path, index=False)
    print(f"Dataset successfully saved to {output_path}")
    print(f"Dataset shape: {df.shape}")

if __name__ == "__main__":
    main()
