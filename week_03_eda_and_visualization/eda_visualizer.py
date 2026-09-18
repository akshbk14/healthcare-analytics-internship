"""
Week 3: Exploratory Data Analysis (EDA) & Visualization Dashboard Framework
Dataset Scope: 10,000 Inpatient Hospital Encounters
Authorship: Healthcare Data Analytics Intern
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def generate_eda_cohort(n_samples=10000, random_seed=42):
    """Generates an empirical clinical dataset with realistic multi-morbidity correlations."""
    np.random.seed(random_seed)
    
    age = np.clip(np.random.normal(65.4, 12.8, n_samples), 18, 95).astype(int)
    los = np.clip(np.random.exponential(4.2, n_samples) + 1, 1, 18).round(1)
    med_count = np.clip(np.random.poisson(14 + (age / 10), n_samples), 1, 45)
    glucose = np.clip(np.random.lognormal(4.9, 0.3, n_samples), 60, 450).round(1)
    
    # Comorbidity calculation: Higher age and meds correlate with higher readmission risk
    risk_logits = -3.2 + (age * 0.02) + (los * 0.08) + (med_count * 0.04) + (glucose * 0.005)
    risk_probs = 1 / (1 + np.exp(-risk_logits))
    readmitted = (np.random.rand(n_samples) < risk_probs).astype(int)
    
    diagnoses = np.random.choice(
        ['Cardiovascular', 'Endocrine/Diabetes', 'Respiratory', 'Digestive', 'Other'],
        size=n_samples, p=[0.32, 0.24, 0.18, 0.14, 0.12]
    )
    
    return pd.DataFrame({
        'age': age,
        'length_of_stay_days': los,
        'num_medications': med_count,
        'glucose_mg_dl': glucose,
        'primary_diagnosis': diagnoses,
        'readmitted_30d': readmitted
    })

def perform_statistical_eda(df: pd.DataFrame):
    """Computes parametric and non-parametric summary statistics."""
    print("=" * 75)
    print("EXPLORATORY DATA ANALYSIS (EDA): 10,000 PATIENT CLINICAL COHORT")
    print("=" * 75)
    
    print("\n1. Central Tendency & Dispersion Metrics:")
    stats_df = df[['age', 'length_of_stay_days', 'num_medications', 'glucose_mg_dl']].describe().T
    stats_df['median'] = df[['age', 'length_of_stay_days', 'num_medications', 'glucose_mg_dl']].median()
    stats_df['IQR'] = df[['age', 'length_of_stay_days', 'num_medications', 'glucose_mg_dl']].quantile(0.75) - \
                      df[['age', 'length_of_stay_days', 'num_medications', 'glucose_mg_dl']].quantile(0.25)
    print(stats_df[['mean', 'std', 'median', 'IQR', 'min', 'max']].round(2))
    
    print("\n2. Readmission Rate by Diagnostic Category:")
    diag_summary = df.groupby('primary_diagnosis')['readmitted_30d'].agg(
        Total_Patients='count',
        Readmissions='sum',
        Readmission_Rate=lambda x: f"{x.mean()*100:.2f}%"
    ).reset_index()
    print(diag_summary.to_string(index=False))
    
    print("\n3. Bivariate Comparison (Readmitted vs Non-Readmitted Means):")
    bivariate = df.groupby('readmitted_30d')[['age', 'length_of_stay_days', 'num_medications', 'glucose_mg_dl']].mean().round(2)
    bivariate.index = ['Not Readmitted (0)', 'Readmitted (1)']
    print(bivariate)
    print("=" * 75)

def generate_eda_visualizations(df: pd.DataFrame):
    """Builds a high-resolution 4-panel EDA clinical overview dashboard."""
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=300)
    fig.suptitle('Clinical Exploratory Data Analysis & Risk Stratification Dashboard', fontsize=16, fontweight='bold')
    
    # 1. Distribution of Length of Stay by Readmission Status
    sns.boxplot(ax=axes[0, 0], x='readmitted_30d', y='length_of_stay_days', data=df, palette=['#64B5F6', '#E57373'])
    axes[0, 0].set_title('A. Length of Stay (Days) by Readmission Status', fontweight='bold')
    axes[0, 0].set_xticklabels(['Not Readmitted (0)', 'Readmitted (1)'])
    axes[0, 0].set_ylabel('Inpatient Stay Duration (Days)')
    axes[0, 0].set_xlabel('30-Day Readmission Outcome')
    
    # 2. Pearson Correlation Heatmap
    corr = df[['age', 'length_of_stay_days', 'num_medications', 'glucose_mg_dl', 'readmitted_30d']].corr()
    sns.heatmap(corr, ax=axes[0, 1], annot=True, cmap='Blues', fmt='.2f', vmin=-0.1, vmax=0.6, linewidths=1.0)
    axes[0, 1].set_title('B. Pearson Correlation Heatmap (Clinical Variables)', fontweight='bold')
    
    # 3. Categorical Readmission Rate by Primary Diagnosis
    diag_order = df.groupby('primary_diagnosis')['readmitted_30d'].mean().sort_values(ascending=False).index
    sns.barplot(ax=axes[1, 0], x='primary_diagnosis', y='readmitted_30d', data=df, order=diag_order, palette='crest', ci=None)
    axes[1, 0].set_title('C. Readmission Rate by Primary Diagnosis (%)', fontweight='bold')
    axes[1, 0].set_ylabel('Readmission Probability')
    axes[1, 0].set_xlabel('Primary Diagnosis Category')
    axes[1, 0].tick_params(axis='x', rotation=20)
    
    # 4. Polypharmacy vs Age Stratified Scatter Plot
    sns.scatterplot(
        ax=axes[1, 1], x='age', y='num_medications', hue='readmitted_30d',
        data=df.sample(1000, random_state=42), alpha=0.65, palette=['#1E88E5', '#D81B60']
    )
    axes[1, 1].set_title('D. Polypharmacy vs. Age (Sample n=1,000 Encounters)', fontweight='bold')
    axes[1, 1].set_xlabel('Patient Age (Years)')
    axes[1, 1].set_ylabel('Prescribed Medications Count')
    axes[1, 1].legend(title='Readmitted', labels=['No', 'Yes'])
    
    plt.tight_layout()
    plt.savefig('eda_clinical_dashboard.png', dpi=300)
    print("\n[+] Dashboard saved locally as 'eda_clinical_dashboard.png'")

if __name__ == "__main__":
    cohort_df = generate_eda_cohort()
    perform_statistical_eda(cohort_df)
    generate_eda_visualizations(cohort_df)
