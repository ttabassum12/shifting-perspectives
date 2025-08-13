#!/usr/bin/env python3

import pandas as pd
import numpy as np
import os

def calculate_performance_changes():
    """Calculate average BBQ increase and MMLU decrease at coefficient 1.6 vs baseline (0.0)"""
    
    # Define bias axes
    axes = ['age', 'appearance', 'disability', 'gender', 
            'nationality', 'race', 'religion', 'socioeconomic']
    
    # Load all data
    all_bbq_data = []
    all_mmlu_data = []
    coeffs = None
    
    for axis in axes:
        file_path = f"../data/coeff_scores/mistral/top_train+prompt/{axis}_train+prompt.csv"
        
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            if coeffs is None:
                coeffs = df['coeff'].values
            all_bbq_data.append(df['bbq_accuracy'].values)
            all_mmlu_data.append(df['mmlu_accuracy'].values)
    
    if not all_bbq_data:
        print("No data available")
        return
    
    # Calculate averages across all axes
    avg_bbq = np.mean(all_bbq_data, axis=0) * 100  # Convert to percentage
    avg_mmlu = np.mean(all_mmlu_data, axis=0) * 100  # Convert to percentage
    
    # Find indices for coefficient 0.0 and 1.6
    coeff_0_idx = np.argmin(np.abs(coeffs - 0.0))
    coeff_1_6_idx = np.argmin(np.abs(coeffs - 1.6))
    
    # Get baseline and target values
    bbq_baseline = avg_bbq[coeff_0_idx]
    bbq_at_1_6 = avg_bbq[coeff_1_6_idx]
    mmlu_baseline = avg_mmlu[coeff_0_idx]
    mmlu_at_1_6 = avg_mmlu[coeff_1_6_idx]
    
    # Calculate changes
    bbq_increase = bbq_at_1_6 - bbq_baseline
    mmlu_decrease = mmlu_baseline - mmlu_at_1_6  # Positive value for decrease
    
    print(f"Performance changes at coefficient 1.6 (vs baseline at 0.0):")
    print(f"")
    print(f"BBQ accuracy:")
    print(f"  Baseline (coeff 0.0): {bbq_baseline:.1f}%")
    print(f"  At coefficient 1.6:   {bbq_at_1_6:.1f}%")
    print(f"  Average increase:     +{bbq_increase:.1f} percentage points")
    print(f"")
    print(f"MMLU accuracy:")
    print(f"  Baseline (coeff 0.0): {mmlu_baseline:.1f}%")
    print(f"  At coefficient 1.6:   {mmlu_at_1_6:.1f}%")
    print(f"  Average decrease:     -{mmlu_decrease:.1f} percentage points")
    print(f"")
    print(f"Summary:")
    print(f"  BBQ improvement: +{bbq_increase:.1f}pp")
    print(f"  MMLU cost:       -{mmlu_decrease:.1f}pp")

if __name__ == "__main__":
    calculate_performance_changes()