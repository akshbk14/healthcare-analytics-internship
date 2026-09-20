"""
Week 4: Predictive Modeling & Comparative Machine Learning Strategy
Comparative Assessment: Logistic Regression vs Random Forest vs XGBoost
Target: 30-Day Hospital Readmission Risk (Cohort N = 10,000)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, average_precision_score, classification_report

def create_modeling_cohort(n_samples=10000, random_seed=42):
    """Generates an empirical clinical dataset with non-linear interaction terms."""
    np.random.seed(random_seed)
    age = np.clip(np.random.normal(65.4, 12.8, n_samples), 18, 95)
    los = np.clip(np.random.exponential(4.2, n_samples) + 1, 1, 18)
    num_meds = np.clip(np.random.poisson(14 + (age / 10), n_samples), 1, 45)
    glucose = np.clip(np.random.lognormal(4.9, 0.3, n_samples), 60, 450)
    prior_stays = np.random.poisson(0.85, n_samples)
    
    # Interaction: Polypharmacy + Prior stays exponentially increases risk
    risk_score = (
        -3.8 
        + 0.02 * age 
        + 0.12 * los 
        + 0.05 * num_meds 
        + 0.004 * glucose 
        + 0.45 * prior_stays 
        + 0.08 * (los * prior_stays)
    )
    risk_probs = 1 / (1 + np.exp(-risk_score))
    readmitted = (np.random.rand(n_samples) < risk_probs).astype(int)
    
    return pd.DataFrame({
        'age': age,
        'length_of_stay': los,
        'num_medications': num_meds,
        'glucose_level': glucose,
        'prior_inpatient_stays': prior_stays,
        'readmitted_30d': readmitted
    })

def evaluate_models():
    df = create_modeling_cohort()
    X = df.drop('readmitted_30d', axis=1)
    y = df['readmitted_30d']
    
    # 80/20 Stratified Partitioning
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Define models
    imbalance_weight = (len(y_train) - sum(y_train)) / sum(y_train)
    models = {
        "Logistic Regression (Baseline)": LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=150, max_depth=6, class_weight='balanced', random_state=42),
        "XGBoost Classifier": XGBClassifier(n_estimators=150, max_depth=4, learning_rate=0.05, scale_pos_weight=imbalance_weight, random_state=42, eval_metric='logloss')
    }
    
    results = {}
    
    print("=" * 80)
    print(f"{'Model Architecture':<32} | {'ROC-AUC':<9} | {'PR-AUC':<9} | {'Recall':<8} | {'F1-Score':<8}")
    print("-" * 80)
    
    plt.figure(figsize=(14, 6), dpi=300)
    
    # Subplot 1: ROC Curves
    plt.subplot(1, 2, 1)
    for name, model in models.items():
        if "Logistic" in name:
            model.fit(X_train_scaled, y_train)
            probs = model.predict_proba(X_test_scaled)[:, 1]
            preds = model.predict(X_test_scaled)
        else:
            model.fit(X_train, y_train)
            probs = model.predict_proba(X_test)[:, 1]
            preds = model.predict(X_test)
            
        roc_auc = roc_auc_score(y_test, probs)
        pr_auc = average_precision_score(y_test, probs)
        fpr, tpr, _ = roc_curve(y_test, probs)
        
        report = classification_report(y_test, preds, output_dict=True)
        recall = report['1']['recall']
        f1 = report['1']['f1-score']
        
        results[name] = {"roc_auc": roc_auc, "pr_auc": pr_auc, "recall": recall, "f1": f1}
        print(f"{name:<32} | {roc_auc:.4f}    | {pr_auc:.4f}    | {recall:.4f}   | {f1:.4f}")
        
        plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.3f})", lw=2)
        
    plt.plot([0, 1], [0, 1], 'k--', lw=1.5, label='Chance Baseline (AUC = 0.500)')
    plt.title('Receiver Operating Characteristic (ROC) Curves', fontsize=12, fontweight='bold')
    plt.xlabel('False Positive Rate (1 - Specificity)')
    plt.ylabel('True Positive Rate (Sensitivity / Recall)')
    plt.legend(loc='lower right', frameon=True)
    plt.grid(True, alpha=0.3)
    
    # Subplot 2: Precision-Recall Curves
    plt.subplot(1, 2, 2)
    for name, model in models.items():
        if "Logistic" in name:
            probs = model.predict_proba(X_test_scaled)[:, 1]
        else:
            probs = model.predict_proba(X_test)[:, 1]
            
        prec, rec, _ = precision_recall_curve(y_test, probs)
        pr_auc = average_precision_score(y_test, probs)
        plt.plot(rec, prec, label=f"{name} (PR-AUC = {pr_auc:.3f})", lw=2)
        
    baseline_pr = sum(y_test) / len(y_test)
    plt.axhline(y=baseline_pr, color='k', linestyle='--', label=f'Class Prevalence ({baseline_pr:.3f})')
    plt.title('Precision-Recall (PR) Curves (Class Imbalance Metric)', fontsize=12, fontweight='bold')
    plt.xlabel('Recall (Sensitivity)')
    plt.ylabel('Precision (Positive Predictive Value)')
    plt.legend(loc='upper right', frameon=True)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('model_performance_comparison.png')
    print("=" * 80)
    print("\n[+] Visual deliverable saved as 'model_performance_comparison.png'")

if __name__ == "__main__":
    evaluate_models()
