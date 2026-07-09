import pandas as pd
import numpy as np
import os
import joblib
import json
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, f1_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay, roc_curve

def main():
    print("Loading processed data for training...")
    train_path = os.path.join('data', 'train_processed.csv')
    test_path = os.path.join('data', 'test_processed.csv')
    
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        raise FileNotFoundError("Processed data not found. Please run preprocess.py first.")
        
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    X_train = train_df.drop(columns=['target'])
    y_train = train_df['target']
    
    X_test = test_df.drop(columns=['target'])
    y_test = test_df['target']
    
    print("Training models...")
    
    # Initialize models
    # We use class_weight='balanced' or rely on SMOTE. Since we used SMOTE, we don't strictly need balanced weights here, but we can tune.
    rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    
    # XGBoost configuration
    # For newer XGBoost, use_label_encoder=False is recommended (or default)
    xgb = XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42, n_jobs=-1)
    
    models = {
        'Random Forest': rf,
        'XGBoost': xgb
    }
    
    results = {}
    os.makedirs('models', exist_ok=True)
    os.makedirs('evaluation', exist_ok=True)
    
    plt.figure(figsize=(8,6))
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        
        # Metrics
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_prob)
        
        print(f"{name} Results:")
        print(f"F1-Score: {f1:.4f}")
        print(f"ROC-AUC: {roc_auc:.4f}")
        print(classification_report(y_test, y_pred))
        
        # Save model
        joblib.dump(model, os.path.join('models', f'{name.lower().replace(" ", "_")}.pkl'))
        
        # Save results
        results[name] = {
            'f1_score': f1,
            'roc_auc': roc_auc
        }
        
        # Confusion Matrix Plot
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Not Default', 'Default'])
        disp.plot(cmap='Blues', values_format='d')
        plt.title(f'{name} Confusion Matrix')
        plt.savefig(os.path.join('evaluation', f'{name.lower().replace(" ", "_")}_cm.png'))
        plt.close() # Close the current figure
        
        # Add to ROC Curve plot
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        # Assuming we plotted on a fresh figure previously, but let's re-open the big one
        
    # Plot combined ROC curves
    plt.figure(figsize=(8,6))
    for name, model in models.items():
        y_prob = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc = roc_auc_score(y_test, y_prob)
        plt.plot(fpr, tpr, label=f"{name} (AUC = {auc:.3f})")
    
    plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves')
    plt.legend(loc='lower right')
    plt.savefig(os.path.join('evaluation', 'combined_roc_curve.png'))
    plt.close()
    
    # Save metrics to JSON
    with open(os.path.join('evaluation', 'metrics.json'), 'w') as f:
        json.dump(results, f, indent=4)
        
    print("\nTraining and evaluation complete. Results saved in 'evaluation' directory.")

if __name__ == "__main__":
    main()
