import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import chi2_contingency

print("="*80)
print("RANDOMIZATION BALANCE CHECKS")
print("="*80)

# Load data
randomization_df = pd.read_csv('../data/randomization-check.csv')   

# Get one row per participant
participant_df = randomization_df.drop_duplicates(subset='Participant ID').copy()

print(f"\nTotal participants: {len(participant_df)}")
print(f"Participants per condition:")
print(participant_df['Condition'].value_counts().sort_index())

# ============================================================================
# 1. AGE (CONTINUOUS)
# ============================================================================
print("\n" + "="*80)
print("CONTINUOUS VARIABLES: AGE")
print("="*80)

# Descriptive statistics by condition
print("\nAge by Condition:")
age_summary = participant_df.groupby('Condition')['Age'].agg(['mean', 'std', 'count', 'min', 'max'])
print(age_summary.round(2))

# One-way ANOVA
age_groups = [participant_df[participant_df['Condition'] == c]['Age'].dropna().values 
              for c in participant_df['Condition'].unique()]
f_stat, p_val = stats.f_oneway(*age_groups)

print(f"\nOne-way ANOVA: F({len(age_groups)-1}, {len(participant_df)-len(age_groups)}) = {f_stat:.3f}, p = {p_val:.4f}")

if p_val < 0.05:
    print("⚠ WARNING: Significant difference in age across conditions")
else:
    print("✓ No significant difference in age across conditions (balanced)")

# ============================================================================
# 2. CATEGORICAL VARIABLES
# ============================================================================

categorical_vars = {
    'Gender': 'Gender',
    'Education': 'Education level',
    'Marital_Status': 'Marital status',
    'Race': 'Race/Ethnicity',
    'Political_Affiliation': 'Political affiliation'
}

print("\n" + "="*80)
print("CATEGORICAL VARIABLES")
print("="*80)

balance_results = []

# Add Age to results first
balance_results.append({
    'Variable': 'Age (continuous)',
    'Test': f'F = {f_stat:.3f}',
    'df': f'{len(age_groups)-1}, {len(participant_df)-len(age_groups)}',
    'p-value': f'{p_val:.4f}',
    'Balanced': 'Yes' if p_val >= 0.05 else 'No'
})

for var_name, var_label in categorical_vars.items():
    # Handle column name variations
    col_name = var_name.replace('_', ' ')  # Try with space
    if col_name not in participant_df.columns:
        col_name = var_name  # Try original
    if col_name not in participant_df.columns:
        print(f"\nSkipping {var_label} (column not found)")
        continue
        
    print(f"\n{'-'*80}")
    print(f"{var_label.upper()}")
    print(f"{'-'*80}")
    
    # Create contingency table
    contingency = pd.crosstab(
        participant_df['Condition'], 
        participant_df[col_name], 
        margins=False
    )
    
    print("\nCross-tabulation (counts):")
    print(contingency)
    
    # Proportions by condition
    print("\nProportions within each condition:")
    proportions = pd.crosstab(
        participant_df['Condition'], 
        participant_df[col_name], 
        normalize='index'
    ) * 100  # Convert to percentages
    print(proportions.round(1))
    
    # Chi-square test
    chi2, p_val, dof, expected = chi2_contingency(contingency)
    
    print(f"\nChi-square test: χ²({dof}) = {chi2:.3f}, p = {p_val:.4f}")
    
    # Calculate Cramer's V
    n = contingency.sum().sum()
    rows = len(contingency)
    cols = len(contingency.columns)
    cramers = np.sqrt(chi2 / (n * min(rows - 1, cols - 1)))
    
    print(f"Cramer's V = {cramers:.3f}", end="")
    if cramers < 0.1:
        print(" (negligible)")
    elif cramers < 0.3:
        print(" (small)")
    elif cramers < 0.5:
        print(" (medium)")
    else:
        print(" (large)")
    
    # Store results
    balance_results.append({
        'Variable': var_label,
        'Test': f'χ² = {chi2:.3f}',
        'df': str(dof),
        'p-value': f'{p_val:.4f}',
        'Balanced': 'Yes' if p_val >= 0.05 else 'No'
    })
    
    if p_val < 0.05:
        print(f"WARNING: Significant difference in {var_label} across conditions")
    else:
        print(f"No significant difference in {var_label} across conditions (balanced)")

# ============================================================================
# 3. SUMMARY TABLE
# ============================================================================
print("\n" + "="*80)
print("BALANCE CHECK SUMMARY")
print("="*80)

balance_summary = pd.DataFrame(balance_results)
print("\n", balance_summary.to_string(index=False))

# Check if all balanced
all_balanced = balance_summary['Balanced'].eq('Yes').all()
if all_balanced:
    print("\n" + "="*80)
    print("ALL DEMOGRAPHIC VARIABLES ARE BALANCED ACROSS CONDITIONS")
    print("="*80)
else:
    print("\n" + "="*80)
    print("WARNING: SOME DEMOGRAPHIC VARIABLES SHOW IMBALANCE")
    print("="*80)
    imbalanced = balance_summary[balance_summary['Balanced'] == 'No']['Variable'].tolist()
    print("Imbalanced variables:", imbalanced)

# ============================================================================
# 4. DETAILED BREAKDOWN BY CONDITION
# ============================================================================
print("\n" + "="*80)
print("DETAILED BREAKDOWN BY CONDITION")
print("="*80)

for var_name, var_label in categorical_vars.items():
    col_name = var_name.replace('_', ' ')
    if col_name not in participant_df.columns:
        col_name = var_name
    if col_name not in participant_df.columns:
        continue
        
    print(f"\n{var_label}:")
    breakdown = pd.crosstab(
        participant_df['Condition'], 
        participant_df[col_name], 
        margins=True
    )
    print(breakdown)

print("\n" + "="*80)
print("BALANCE CHECKS COMPLETE")
print("="*80)

# ============================================================================
# 5. CREATE LATEX TABLE DATA
# ============================================================================
print("\n" + "="*80)
print("LATEX TABLE DATA")
print("="*80)

print("\nFor simple summary table:")
for idx, row in balance_summary.iterrows():
    if idx == 0:  # Age row
        print(f"{row['Variable']} & {row['Test']} & {row['df']} & {row['p-value']} & {row['Balanced']} \\\\")
    else:
        print(f"{row['Variable']} & {row['Test']} & {row['df']} & {row['p-value']} & {row['Balanced']} \\\\")

# ============================================================================
# 6. EXPORT TO CSV FOR EASY COPYING
# ============================================================================
balance_summary.to_csv('../output/balance_check_results.csv', index=False)
print("\nResults exported to '../output/balance_check_results.csv'")