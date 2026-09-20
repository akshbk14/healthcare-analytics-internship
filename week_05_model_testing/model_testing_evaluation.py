"""
Week 5: Model Testing, Error Diagnostics & Iterative Improvement Roadmap
Target: 30-Day Hospital Readmission Risk (Cohort N = 10,000)
Author: Healthcare Data Analytics Intern
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_auc_score,
    precision_recall_curve, brier_score_loss, f1_score, recall_score, precision_score
)
from sklearn.calibration import calibration_curve

def load_cohort(n_samples=10000, random_seed=42):
    """Generates synthetic benchmark cohort with multi-morbidity interactions."""
    np.random.seed(random_seed)
    age = np.clip(np.random.normal(65.4, 12.8, n_samples), 18, 95)
    los = np.clip(np.random.exponential(4.2, n_samples) + 1, 1, 18)
    num_meds = np.clip(np.random.poisson(14 + (age / 10), n_samples), 1, 45)
    glucose = np.clip(np.random.lognormal(4.9, 0.3, n_samples), 60, 450)
    prior_stays = np.random.poisson(0.85, n_samples)
    
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

def run_evaluation_suite():
    df = load_cohort()
    X = df.drop('readmitted_30d', axis=1)
    y = df['readmitted_30d']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )
    
    # Model configuration with scale_pos_weight
    imbalance_weight = (len(y_train) - sum(y_train)) / sum(y_train)
    champion_model = XGBClassifier(
        n_estimators=150,
        max_depth=4,
        learning_rate=0.05,
        scale_pos_weight=imbalance_weight,
        random_state=42,
        eval_metric='logloss'
    )
    
    # 1. 5-Fold Stratified Cross-Validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_auc_scores = cross_val_score(champion_model, X_train, y_train, cv=cv, scoring='roc_auc')
    
    champion_model.fit(X_train, y_train)
    y_probs = champion_model.predict_proba(X_test)[:, 1]
    
    # 2. Threshold Calibration Sweep
    thresholds = [0.50, 0.40, 0.35, 0.30]
    print("=" * 80)
    print("5-FOLD STRATIFIED CV ROC-AUC: {:.4f} (±{:.4f})".format(cv_auc_scores.mean(), cv_auc_scores.std()))
    print("BRIER CALIBRATION SCORE: {:.4f}".format(brier_score_loss(y_test, y_probs)))
    print("=" * 80)
    print(f"{'Decision Cutoff':<16} | {'Recall (Sens)':<14} | {'Specificity':<14} | {'Precision':<12} | {'F1-Score':<10}")
    print("-" * 80)
    for t in thresholds:
        preds = (y_probs >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()
        sens = tp / (tp + fn)
        spec = tn / (tn + fp)
        prec = precision_score(y_test, preds, zero_division=0)
        f1 = f1_score(y_test, preds)
        print(f"Cutoff: {t:<9} | {sens*100:.2f}%{'':<6} | {spec*100:.2f}%{'':<6} | {prec*100:.2f}%{'':<4} | {f1:.4f}")
    print("=" * 80)
    
    # Generate 4-Panel Visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 11), dpi=300)
    fig.suptitle('Clinical Predictive Model Evaluation & Diagnostic Suite', fontsize=16, fontweight='bold')
    
    # A. Confusion Matrix (Threshold = 0.35)
    opt_preds = (y_probs >= 0.35).astype(int)
    cm = confusion_matrix(y_test, opt_preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0], cbar=False,
                annot_kws={'size': 14, 'weight': 'bold'})
    axes[0, 0].set_title('A. Confusion Matrix (Optimized Threshold = 0.35)', fontweight='bold')
    axes[0, 0].set_xticklabels(['Pred Negative', 'Pred Readmit'])
    axes[0, 0].set_yticklabels(['Actual Negative', 'Actual Readmit'])
    axes[0, 0].set_ylabel('Ground Truth')
    
    # B. Calibration Curve
    prob_true, prob_pred = calibration_curve(y_test, y_probs, n_bins=10)
    axes[0, 1].plot(prob_pred, prob_true, marker='o', lw=2, color='#1E88E5', label='XGBoost Model')
    axes[0, 1].plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfect Calibration')
    axes[0, 1].set_title('B. Reliability Calibration Curve (Brier = {:.3f})'.format(brier_score_loss(y_test, y_probs)), fontweight='bold')
    axes[0, 1].set_xlabel('Mean Predicted Probability')
    axes[0, 1].set_ylabel('Fraction of Positives')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # C. Threshold Sensitivity vs. Specificity Trade-Off
    thresh_range = np.linspace(0.1, 0.9, 80)
    recalls, specificities = [], []
    for th in thresh_range:
        p = (y_probs >= th).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, p).ravel()
        recalls.append(tp / (tp + fn))
        specificities.append(tn / (tn + fp))
    axes[1, 0].plot(thresh_range, recalls, label='Sensitivity (Recall)', color='#E53935', lw=2)
    axes[1, 0].plot(thresh_range, specificities, label='Specificity', color='#43A047', lw=2)
    axes[1, 0].axvline(0.35, color='black', linestyle=':', label='Chosen Cutoff (0.35)')
    axes[1, 0].set_title('C. Decision Threshold Trade-Off Analysis', fontweight='bold')
    axes[1, 0].set_xlabel('Probability Cutoff')
    axes[1, 0].set_ylabel('Rate (0.0 to 1.0)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # D. Feature Importance (Diagnostic attribution)
    importances = champion_model.feature_importances_
    features = X.columns
    sorted_idx = np.argsort(importances)
    axes[1, 1].barh(range(len(sorted_idx)), importances[sorted_idx], color='#3949AB', align='center')
    axes[1, 1].set_yticks(range(len(sorted_idx)))
    axes[1, 1].set_yticklabels(features[sorted_idx])
    axes[1, 1].set_title('D. Relative Gini Feature Importance Matrix', fontweight='bold')
    axes[1, 1].set_xlabel('Relative Weight')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('model_evaluation_diagnostic.png')
    print("\n[+] Visual deliverable saved as 'model_evaluation_diagnostic.png'")

if __name__ == "__main__":
    run_evaluation_suite()
