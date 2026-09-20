"""
Week 6: Capstone Executive Summary & Final Presentation Dashboard
Synthesizing Weeks 1-5 into an Executive Presentation Artifact
Project: 30-Day Hospital Readmission Risk Prediction Pipeline
Author: Healthcare Data Analytics Intern
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def generate_executive_dashboard():
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(15, 11), dpi=300)
    fig.suptitle('Executive Capstone: 30-Day Hospital Readmission Analytics Framework', fontsize=16, fontweight='bold')

    # 1. Pipeline Stages Performance Progression
    stages = ['Baseline\nLogistic Reg', 'Random Forest\nEnsemble', 'XGBoost\nDefault (0.50)', 'XGBoost\nCalibrated (0.35)']
    recalls = [68.31, 74.42, 81.10, 89.24]
    bars = axes[0, 0].bar(stages, recalls, color=['#90CAF9', '#42A5F5', '#1E88E5', '#0D47A1'], width=0.55)
    axes[0, 0].set_title('A. Clinical Sensitivity Progression (Recall %)', fontweight='bold')
    axes[0, 0].set_ylabel('High-Risk Detection Rate (%)')
    axes[0, 0].set_ylim(50, 100)
    for bar in bars:
        yval = bar.get_height()
        axes[0, 0].text(bar.get_x() + bar.get_width()/2.0, yval + 1.2, f"{yval:.1f}%", ha='center', fontweight='bold')

    # 2. Financial & Clinical Impact (Cost Savings Projection)
    categories = ['Unmitigated Readmissions\n(Status Quo)', 'Standard Discharge\n(0.50 Threshold)', 'Optimized Intervention\n(0.35 Calibrated)']
    penalties_in_k = [520, 280, 110] # Financial penalty estimates in $ thousands
    bars2 = axes[0, 1].bar(categories, penalties_in_k, color=['#E53935', '#FB8C00', '#43A047'], width=0.5)
    axes[0, 1].set_title('B. Projected CMS Penalty Exposure ($k)', fontweight='bold')
    axes[0, 1].set_ylabel('Estimated Hospital Penalty ($k USD)')
    for bar in bars2:
        yval = bar.get_height()
        axes[0, 1].text(bar.get_x() + bar.get_width()/2.0, yval + 10, f"${yval}k", ha='center', fontweight='bold')

    # 3. Top Risk Drivers (Clinical Explainability)
    drivers = ['Length of Stay (>6 Days)', 'Prior Stays (>=2 Visits)', 'Polypharmacy (>22 Meds)', 'Cardiovascular Diagnosis', 'Glucose Level (>180 mg/dL)']
    weights = [0.34, 0.28, 0.18, 0.12, 0.08]
    axes[1, 0].barh(drivers[::-1], weights[::-1], color='#5C6BC0')
    axes[1, 0].set_title('C. Top Clinical Determinants of Readmission', fontweight='bold')
    axes[1, 0].set_xlabel('Relative Risk Impact Factor')

    # 4. Multi-Tier Patient Triage Distribution
    labels = ['Low Risk\n(Routine Discharge)', 'Medium Risk\n(Tele-Health Call)', 'High Risk\n(Transitional Care Plan)']
    sizes = [69.9, 19.3, 10.8]
    colors = ['#81C784', '#FFD54F', '#E57373']
    axes[1, 1].pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140, colors=colors,
                   textprops={'fontsize': 10, 'weight': 'bold'}, explode=(0, 0.05, 0.1))
    axes[1, 1].set_title('D. Operational Patient Risk Triage Distribution', fontweight='bold')

    plt.tight_layout()
    plt.savefig('final_executive_presentation.png')
    print("[+] Final presentation dashboard saved successfully as 'final_executive_presentation.png'")

if __name__ == "__main__":
    generate_executive_dashboard()
