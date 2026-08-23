"""
Week 1: Strategic Planning & Architecture Baseline
Healthcare Data Analytics - 30-Day Hospital Readmission Risk Framework
"""

import pandas as pd
import numpy as np

def init_project_scope():
    """Generates the foundational scope and baseline schema for the project."""
    project_metadata = {
        "Project Title": "30-Day Hospital Readmission Risk Prediction & Stratification",
        "Domain": "Healthcare Operations & Clinical Decision Support",
        "Target Variable": "readmitted_30d (Binary: 0=No, 1=Yes)",
        "Core Tech Stack": ["Python", "Pandas", "Scikit-Learn", "Matplotlib", "Seaborn"],
        "Target Milestones": 6
    }
    
    print("=" * 60)
    print("HEALTHCARE ANALYTICS PROJECT SCOPE INITIALIZED")
    print("=" * 60)
    for key, value in project_metadata.items():
        print(f"{key:<20}: {value}")
    print("=" * 60)

def simulate_raw_clinical_data(n_records=500):
    """Simulates raw clinical data structure for scoping validation."""
    np.random.seed(42)
    data = {
        'patient_id': range(1001, 1001 + n_records),
        'age': np.random.randint(18, 92, size=n_records),
        'time_in_hospital_days': np.random.randint(1, 15, size=n_records),
        'num_lab_procedures': np.random.randint(5, 120, size=n_records),
        'num_medications': np.random.randint(1, 40, size=n_records),
        'num_diagnoses': np.random.randint(1, 16, size=n_records),
        'had_emergency_visit_past_year': np.random.choice([0, 1], size=n_records, p=[0.7, 0.3]),
        'readmitted_30d': np.random.choice([0, 1], size=n_records, p=[0.82, 0.18])
    }
    df = pd.DataFrame(data)
    print(f"\nSynthetic Clinical Dataset Scoped Successfully ({n_records} rows).")
    print(df.head())
    print("\nTarget Class Distribution:")
    print(df['readmitted_30d'].value_counts(normalize=True))
    return df

if __name__ == "__main__":
    init_project_scope()
    simulate_raw_clinical_data()
