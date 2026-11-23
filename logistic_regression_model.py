"""
Predictive Modeling of Quick-Commerce Delivery Delays Using Logistic Regression
Course: Fundamentals of Data Science (CSDS2002P) — UPES Dehradun
Author: Anukalp Verma (SAP ID: 590028327, Batch B2 Data Science)
"""
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)

DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "blinkit_orders_cleaned.csv")

def run_pipeline():
    print("=" * 80)
    print("PREDICTIVE MODELING OF QUICK-COMMERCE DELIVERY DELAYS (LOGISTIC REGRESSION)")
    print("=" * 80)
    
    # 1. Ingestion
    df = pd.read_csv(DATA_PATH)
    print(f"Dataset Loaded: {len(df)} transactions.")
    
    # 2. Feature Engineering
    df["order_datetime"] = pd.to_datetime(df["order_date"])
    df["order_hour"] = df["order_datetime"].dt.hour
    df["day_of_week"] = df["order_datetime"].dt.day_of_week
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    df["is_peak_hours"] = df["order_hour"].isin([18, 19, 20, 21, 22]).astype(int)
    df["is_delayed"] = (df["delivery_status"] != "On Time").astype(int)
    
    print(f"Class Balance: On Time = {(df['is_delayed'] == 0).sum()} ({(df['is_delayed'] == 0).mean()*100:.1f}%), "
          f"Delayed = {(df['is_delayed'] == 1).sum()} ({(df['is_delayed'] == 1).mean()*100:.1f}%)")
          
    feature_cols = ["order_total", "order_hour", "is_weekend", "is_peak_hours"]
    X = pd.get_dummies(df[feature_cols + ["payment_method"]], columns=["payment_method"], drop_first=True)
    y = df["is_delayed"]
    
    # 3. Stratified Split & Feature Scaling
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 4. Logistic Regression Modeling
    clf = LogisticRegression(class_weight="balanced", random_state=42, max_iter=1000)
    clf.fit(X_train_scaled, y_train)
    
    y_pred = clf.predict(X_test_scaled)
    y_prob = clf.predict_proba(X_test_scaled)[:, 1]
    
    # 5. Metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_prob)
    
    print("\n--- Model Evaluation (20% Holdout Test Partition) ---")
    print(f"Accuracy:  {acc*100:.2f}%")
    print(f"Precision: {prec*100:.2f}%")
    print(f"Recall:    {rec*100:.2f}%")
    print(f"F1-Score:  {f1:.4f}")
    print(f"ROC-AUC:   {auc:.4f}")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    odds_ratios = pd.DataFrame({
        "Feature": X.columns,
        "Coefficient (Beta)": clf.coef_[0].round(4),
        "Odds Ratio exp(Beta)": np.exp(clf.coef_[0]).round(4)
    }).sort_values("Odds Ratio exp(Beta)", ascending=False).reset_index(drop=True)
    print("\nFeature Coefficients and Odds Ratios:")
    print(odds_ratios.to_string(index=False))
    
    # 6. Sample Live Inference
    print("\n--- Live Prediction Example ---")
    sample_order = pd.DataFrame([{col: 0 for col in X.columns}])
    sample_order["order_total"] = 2800.0
    sample_order["order_hour"] = 20
    sample_order["is_weekend"] = 0
    sample_order["is_peak_hours"] = 1
    sample_order["payment_method_Cash"] = 1
    
    prob_delay = clf.predict_proba(scaler.transform(sample_order))[0, 1]
    print(f"Simulated Order: Rs. 2,800 cart, 8:00 PM (Dinner Peak Rush), Cash on Delivery")
    print(f"Predicted Delay Probability: {prob_delay*100:.2f}% -> Outcome: {'Delayed (SLA Breach)' if prob_delay >= 0.5 else 'On Time'}")

if __name__ == "__main__":
    run_pipeline()
