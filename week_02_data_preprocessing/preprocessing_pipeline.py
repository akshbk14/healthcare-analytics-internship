"""
Week 2: Healthcare Data Collection, Cleaning & Preprocessing Workflow
Automated End-to-End Pipeline using Pandas, NumPy, and Scikit-Learn
"""

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, RobustScaler

def create_raw_healthcare_sample(n=300):
    """Simulates realistic messy raw EHR records with missingness and anomalies."""
    np.random.seed(42)
    data = {
        'patient_id': [f"PT-{i:04d}" for i in range(1, n + 1)],
        'age': np.random.choice([25, 45, 65, 80, np.nan, 140], size=n, p=[0.25, 0.35, 0.25, 0.1, 0.03, 0.02]),
        'systolic_bp': np.random.choice([120, 135, 160, 290, np.nan], size=n, p=[0.4, 0.3, 0.2, 0.05, 0.05]),
        'glucose_mg_dl': np.random.choice([95, 140, 220, 450, np.nan], size=n, p=[0.35, 0.35, 0.2, 0.05, 0.05]),
        'admission_type': np.random.choice(['Emergency', 'Elective', 'Urgent', None], size=n, p=[0.5, 0.3, 0.15, 0.05]),
        'primary_diagnosis': np.random.choice(['Circulatory', 'Respiratory', 'Diabetes', 'Digestive', 'Other'], size=n),
        'prior_inpatient_stays': np.random.randint(0, 10, size=n),
        'readmitted_30d': np.random.choice([0, 1], size=n, p=[0.82, 0.18])
    }
    return pd.DataFrame(data)

def clean_clinical_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    """Handles domain-specific outliers, unphysical physiological bounds, and duplicates."""
    df_clean = df.drop_duplicates(subset=['patient_id']).copy()
    
    # Clip/replace unphysical human vital parameters with NaN for statistical imputation
    df_clean.loc[(df_clean['age'] < 0) | (df_clean['age'] > 115), 'age'] = np.nan
    df_clean.loc[(df_clean['systolic_bp'] < 50) | (df_clean['systolic_bp'] > 260), 'systolic_bp'] = np.nan
    df_clean.loc[(df_clean['glucose_mg_dl'] < 30) | (df_clean['glucose_mg_dl'] > 600), 'glucose_mg_dl'] = np.nan
    
    return df_clean

def build_preprocessing_pipeline(num_cols, cat_cols):
    """Constructs modular Scikit-Learn transformers to prevent data leakage."""
    num_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', RobustScaler())  # Robust to skewed healthcare lab distributions
    ])
    
    cat_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer(transformers=[
        ('num', num_pipeline, num_cols),
        ('cat', cat_pipeline, cat_cols)
    ])
    return preprocessor

def execute_preprocessing_run():
    print("=" * 65)
    print("STEP 1: Ingesting Raw Healthcare Records...")
    raw_df = create_raw_healthcare_sample()
    print(f"Ingested {len(raw_df)} records. Missing values per column:\n{raw_df.isnull().sum()}")
    
    print("\nSTEP 2: Cleaning Domain Anomalies & Deduplication...")
    cleaned_df = clean_clinical_anomalies(raw_df)
    
    num_features = ['age', 'systolic_bp', 'glucose_mg_dl', 'prior_inpatient_stays']
    cat_features = ['admission_type', 'primary_diagnosis']
    
    print("\nSTEP 3: Executing Sklearn ColumnTransformer Pipeline...")
    pipeline = build_preprocessing_pipeline(num_features, cat_features)
    
    X = cleaned_df[num_features + cat_features]
    X_processed = pipeline.fit_transform(X)
    
    feature_names = (
        num_features + 
        pipeline.named_transformers_['cat'].named_steps['encoder'].get_feature_names_out(cat_features).tolist()
    )
    
    processed_df = pd.DataFrame(X_processed, columns=feature_names)
    print(f"Preprocessing Complete. Processed Matrix Shape: {processed_df.shape}")
    print("\nSample Preprocessed Features (First 3 Rows):")
    print(processed_df.head(3))
    print("=" * 65)

if __name__ == "__main__":
    execute_preprocessing_run()
