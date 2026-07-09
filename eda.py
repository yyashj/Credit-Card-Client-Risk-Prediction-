import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def main():
    print("Loading data for EDA...")
    data_path = os.path.join('data', 'raw_credit_card_data.csv')
    if not os.path.exists(data_path):
        print("Data not found. Please run fetch_data.py first.")
        return
        
    df = pd.read_csv(data_path)
    
    print("\n--- Basic Info ---")
    print(df.info())
    
    print("\n--- Target Variable Distribution ---")
    # In OpenML 42477, target is usually 'class' or 'default.payment.next.month'
    target_col = [col for col in df.columns if 'default' in col.lower() or 'class' in col.lower()][-1]
    print(f"Target column identified as: {target_col}")
    print(df[target_col].value_counts(normalize=True))
    
    # Create EDA visualizations directory
    os.makedirs('eda_plots', exist_ok=True)
    
    # 1. Target Distribution Plot
    plt.figure(figsize=(6,4))
    sns.countplot(data=df, x=target_col)
    plt.title('Distribution of Target Variable')
    plt.savefig('eda_plots/target_distribution.png')
    plt.close()
    
    # 2. Correlation Heatmap
    # Convert categorical to numeric if necessary just for the heatmap
    numeric_df = df.select_dtypes(include=['float64', 'int64', 'int32'])
    if len(numeric_df.columns) > 1:
        plt.figure(figsize=(12,10))
        corr = numeric_df.corr()
        sns.heatmap(corr, annot=False, cmap='coolwarm')
        plt.title('Correlation Heatmap of Numeric Features')
        plt.savefig('eda_plots/correlation_heatmap.png')
        plt.close()
        
    print("EDA plots saved in 'eda_plots' directory.")

if __name__ == "__main__":
    main()
