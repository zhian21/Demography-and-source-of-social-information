import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.formula.api import ols
import statsmodels.api as sm

robustness_df = pd.read_csv('../data/robustness-check.csv')

# Rename columns to remove spaces (easier for formulas)
robustness_df.columns = robustness_df.columns.str.replace(' ', '_')

# ============================================================================
# ROBUSTNESS CHECK 1: GENDER HETEROGENEITY
# ============================================================================
print("\n" + "="*80)
print("ROBUSTNESS CHECK 1: GENDER HETEROGENEITY")
print("="*80)

# Aggregate to participant level
participant_robust = robustness_df.groupby(['Participant_ID', 'Condition', 'Gender']).agg({
    'Initial_Error': 'mean',
    'Final_Error': 'mean',
    'Error_Reduction': 'mean',
    'Influence_Score': 'mean'
}).reset_index()

# Focus on Male/Female only (exclude small categories)
gender_data = participant_robust[participant_robust['Gender'].isin(['Male', 'Female'])]

# Test interaction: Condition x Gender
print("\nTesting Condition x Gender Interaction:")
model = ols('Error_Reduction ~ C(Condition) * C(Gender)', data=gender_data).fit()
anova_table = sm.stats.anova_lm(model, typ=2)
print(anova_table)

# Separate analyses by gender
print("\n" + "-"*80)
print("EFFECTS BY GENDER")
print("-"*80)

for gender in ['Male', 'Female']:
    gender_subset = gender_data[gender_data['Gender'] == gender]
    
    print(f"\n{gender} Participants (N = {len(gender_subset)}):")
    print("\nMean Error Reduction by Condition:")
    print(gender_subset.groupby('Condition')['Error_Reduction'].agg(['mean', 'std', 'count']))
    
    # ANOVA within gender
    conditions = [gender_subset[gender_subset['Condition'] == c]['Error_Reduction'].values 
                  for c in gender_subset['Condition'].unique()]
    f_stat, p_val = stats.f_oneway(*conditions)
    print(f"\nOne-way ANOVA: F = {f_stat:.2f}, p = {p_val:.4f}")
    
    # Effect sizes vs control
    control_data = gender_subset[gender_subset['Condition'] == 'Individual']['Error_Reduction']
    
    for treatment in ['Age', 'Conformity', 'Prestige', 'Season']:
        treat_data = gender_subset[gender_subset['Condition'] == treatment]['Error_Reduction']
        
        if len(treat_data) > 0 and len(control_data) > 0:
            # Cohen's d
            pooled_std = np.sqrt((treat_data.std()**2 + control_data.std()**2) / 2)
            cohens_d = (treat_data.mean() - control_data.mean()) / pooled_std
            
            # t-test
            t_stat, p_val = stats.ttest_ind(treat_data, control_data)
            
            print(f"  {treatment} vs Control: M_diff = {treat_data.mean() - control_data.mean():.2f}, "
                  f"d = {cohens_d:.2f}, p = {p_val:.4f}")

# ============================================================================
# ROBUSTNESS CHECK 2: INITIAL BELIEF STRENGTH
# ============================================================================
print("\n" + "="*80)
print("ROBUSTNESS CHECK 2: INITIAL BELIEF STRENGTH")
print("="*80)

# Create quartiles based on initial misperception
participant_robust['Initial_Quartile'] = pd.qcut(
    participant_robust['Initial_Error'], 
    q=4, 
    labels=['Low', 'Medium-Low', 'Medium-High', 'High']
)

print("\nInitial Misperception Quartile Distribution:")
print(participant_robust['Initial_Quartile'].value_counts().sort_index())

# Test interaction: Condition × Initial Quartile
model2 = ols('Error_Reduction ~ C(Condition) * C(Initial_Quartile)', 
             data=participant_robust).fit()
anova_table2 = sm.stats.anova_lm(model2, typ=2)
print("\nCondition × Initial Quartile Interaction:")
print(anova_table2)

# Show means by condition and quartile
print("\nMean Error Reduction by Condition and Initial Quartile:")
pivot = participant_robust.pivot_table(
    values='Error_Reduction',
    index='Condition',
    columns='Initial_Quartile',
    aggfunc='mean'
)
print(pivot.round(2))

# ============================================================================
# ROBUSTNESS CHECK 3: STATEMENT-LEVEL HETEROGENEITY
# ============================================================================
print("\n" + "="*80)
print("ROBUSTNESS CHECK 3: STATEMENT-LEVEL HETEROGENEITY")
print("="*80)

# Define statement types
equitable_statements = [1, 2, 3, 4, 5, 6]
inequitable_statements = [7, 8, 9, 10, 11, 12]

for stmt_type, stmt_ids in [('Equitable', equitable_statements), 
                             ('Inequitable', inequitable_statements)]:
    
    print(f"\n{'-'*80}")
    print(f"{stmt_type.upper()} STATEMENTS (IDs: {stmt_ids})")
    print(f"{'-'*80}")
    
    # Filter data
    stmt_data = robustness_df[robustness_df['Statement_ID'].isin(stmt_ids)]
    
    # Aggregate to participant level
    participant_stmt = stmt_data.groupby(['Participant_ID', 'Condition']).agg({
        'Error_Reduction': 'mean',
        'Initial_Error': 'mean',
        'Final_Error': 'mean'
    }).reset_index()
    
    print(f"\nDescriptive Statistics by Condition:")
    print(participant_stmt.groupby('Condition')['Error_Reduction'].describe().round(2))
    
    # ANOVA
    conditions = [participant_stmt[participant_stmt['Condition'] == c]['Error_Reduction'].values 
                  for c in participant_stmt['Condition'].unique()]
    f_stat, p_val = stats.f_oneway(*conditions)
    print(f"\nOne-way ANOVA: F = {f_stat:.2f}, p = {p_val:.4f}")
    
    # Effect sizes vs control
    control = participant_stmt[participant_stmt['Condition'] == 'Individual']['Error_Reduction']
    
    for treatment in ['Age', 'Conformity', 'Prestige', 'Season']:
        treat = participant_stmt[participant_stmt['Condition'] == treatment]['Error_Reduction']
        
        if len(treat) > 0 and len(control) > 0:
            pooled_std = np.sqrt((treat.std()**2 + control.std()**2) / 2)
            cohens_d = (treat.mean() - control.mean()) / pooled_std
            print(f"  {treatment} vs Control: d = {cohens_d:.2f}")


import pandas as pd
import numpy as np

print("="*80)
print("STATEMENT-LEVEL ERROR REDUCTION TABLE")
print("="*80)

# Calculate mean error reduction by statement and condition
statement_level = robustness_df.groupby(['Statement_ID', 'Condition'])['Error_Reduction'].mean().reset_index()

# Pivot to wide format
statement_pivot = statement_level.pivot(index='Statement_ID', columns='Condition', values='Error_Reduction')

# Reorder columns
column_order = ['Individual', 'Age', 'Conformity', 'Prestige', 'Season']
statement_pivot = statement_pivot[column_order]

print("\nMean Error Reduction by Statement and Condition:")
print(statement_pivot.round(2))

# Statement labels (shortened for table)
statement_labels = {
    1: 'Sexual autonomy',
    2: "Women's education",
    3: 'Divorce freedom',
    4: 'Professional equality',
    5: 'Political opportunities',
    6: 'Life without marriage',
    7: 'Exaggerate violence',
    8: "Boys' education priority",
    9: 'Stay for children',
    10: 'Claim discrimination',
    11: 'Men better leaders',
    12: 'Women need children'
}

# Generate LaTeX output
print("\n" + "="*80)
print("LATEX TABLE OUTPUT")
print("="*80)
print()

print("\\begin{table}[h]")
print("\\centering")
print("\\small")
print("\\begin{tabular}{clrrrrr}")
print("\\toprule")
print("\\textbf{ID} & \\textbf{Statement} & \\textbf{Individual} & \\textbf{Age} & \\textbf{Conformity} & \\textbf{Prestige} & \\textbf{Season} \\\\")
print("\\midrule")
print("\\multicolumn{7}{l}{\\textit{Equitable Norms (Underestimation)}} \\\\")
print("\\midrule")

for stmt_id in range(1, 7):
    row = statement_pivot.loc[stmt_id]
    label = statement_labels[stmt_id]
    print(f"{stmt_id} & {label} & {row['Individual']:.1f} & {row['Age']:.1f} & {row['Conformity']:.1f} & {row['Prestige']:.1f} & {row['Season']:.1f} \\\\")

print("\\midrule")
print("\\multicolumn{7}{l}{\\textit{Inequitable Norms (Overestimation)}} \\\\")
print("\\midrule")

for stmt_id in range(7, 13):
    row = statement_pivot.loc[stmt_id]
    label = statement_labels[stmt_id]
    print(f"{stmt_id} & {label} & {row['Individual']:.1f} & {row['Age']:.1f} & {row['Conformity']:.1f} & {row['Prestige']:.1f} & {row['Season']:.1f} \\\\")

print("\\bottomrule")
print("\\end{tabular}")
print("\\caption{\\textbf{Statement-Level Error Reduction by Condition.} Mean error reduction (percentage points) for each gender norm statement across experimental conditions. Statements 1-6: equitable norms (underestimation of peer support); Statements 7-12: inequitable norms (overestimation). Complete statement text in Table S1.}")
print("\\label{tab:statement-level}")
print("\\end{table}")

# Export to CSV
statement_pivot.to_csv('../output/statement_level_results.csv')
print("\nExported to 'statement_level_results.csv'")

# print("\n" + "="*80)
# print("COMPLETE")
# print("="*80)


# ============================================================================
# ROBUSTNESS CHECK 4: OUTLIER SENSITIVITY
# ============================================================================
print("\n" + "="*80)
print("ROBUSTNESS CHECK 4: OUTLIER SENSITIVITY")
print("="*80)

from scipy.stats.mstats import winsorize

# Winsorize a copy at 1st and 99th percentiles; preserve the original outcome
original_error_reduction = participant_robust['Error_Reduction'].copy(deep=True)
participant_robust['Error_Reduction_Winsorized'] = winsorize(
    participant_robust['Error_Reduction'].to_numpy(dtype=float, copy=True),
    limits=[0.01, 0.01]
)
pd.testing.assert_series_equal(
    participant_robust['Error_Reduction'], original_error_reduction
)

# Original analysis
print("\nORIGINAL ANALYSIS:")
conditions_orig = [participant_robust[participant_robust['Condition'] == c]['Error_Reduction'].values 
                   for c in participant_robust['Condition'].unique()]
f_orig, p_orig = stats.f_oneway(*conditions_orig)
print(f"One-way ANOVA: F = {f_orig:.2f}, p = {p_orig:.4f}")

# Winsorized analysis
print("\nWINSORIZED ANALYSIS (1st-99th percentiles):")
conditions_wins = [participant_robust[participant_robust['Condition'] == c]['Error_Reduction_Winsorized'].values 
                   for c in participant_robust['Condition'].unique()]
f_wins, p_wins = stats.f_oneway(*conditions_wins)
print(f"One-way ANOVA: F = {f_wins:.2f}, p = {p_wins:.4f}")

# Trimmed analysis (exclude top/bottom 5%)
lower_bound = participant_robust['Error_Reduction'].quantile(0.05)
upper_bound = participant_robust['Error_Reduction'].quantile(0.95)
trimmed_data = participant_robust[
    (participant_robust['Error_Reduction'] >= lower_bound) & 
    (participant_robust['Error_Reduction'] <= upper_bound)
]

print(f"\nTRIMMED ANALYSIS (5th-95th percentiles, N = {len(trimmed_data)}):")
conditions_trim = [trimmed_data[trimmed_data['Condition'] == c]['Error_Reduction'].values 
                   for c in trimmed_data['Condition'].unique()]
f_trim, p_trim = stats.f_oneway(*conditions_trim)
print(f"One-way ANOVA: F = {f_trim:.2f}, p = {p_trim:.4f}")

print("\nComparison:")
print(f"Original:    F = {f_orig:.2f}, p = {p_orig:.4f}")
print(f"Winsorized:  F = {f_wins:.2f}, p = {p_wins:.4f}")
print(f"Trimmed 5%:  F = {f_trim:.2f}, p = {p_trim:.4f}")

# ============================================================================
# ROBUSTNESS CHECK 5: ALTERNATIVE SPECIFICATIONS
# ============================================================================
print("\n" + "="*80)
print("ROBUSTNESS CHECK 5: ALTERNATIVE DEPENDENT VARIABLE SPECIFICATIONS")
print("="*80)

# Create alternative DVs
participant_robust['Proportional_Reduction'] = (
    participant_robust['Error_Reduction'] / participant_robust['Initial_Error']
)

# Handle infinite values in proportional reduction
participant_robust['Proportional_Reduction'] = participant_robust['Proportional_Reduction'].replace(
    [np.inf, -np.inf], np.nan
)

dvs = {
    'Absolute Reduction': 'Error_Reduction',
    'Proportional Reduction': 'Proportional_Reduction',
    'Final Accuracy': 'Final_Error',
}

for dv_name, dv_col in dvs.items():
    print(f"\n{'-'*80}")
    print(f"DV: {dv_name}")
    print(f"{'-'*80}")
    
    # Remove NaN values
    clean_data = participant_robust[participant_robust[dv_col].notna()]
    
    conditions_dv = [clean_data[clean_data['Condition'] == c][dv_col].values 
                     for c in clean_data['Condition'].unique()]
    
    # ANOVA
    f_stat, p_val = stats.f_oneway(*conditions_dv)
    print(f"One-way ANOVA: F = {f_stat:.2f}, p = {p_val:.4f}")
    
    # Kruskal-Wallis (non-parametric)
    h_stat, h_pval = stats.kruskal(*conditions_dv)
    print(f"Kruskal-Wallis: H = {h_stat:.2f}, p = {h_pval:.4f}")
    
    # Means by condition
    print(f"\nMeans by Condition:")
    print(clean_data.groupby('Condition')[dv_col].mean().round(2))

# ============================================================================
# ROBUSTNESS CHECK 6: POLITICAL ORIENTATION
# ============================================================================
print("\n" + "="*80)
print("ROBUSTNESS CHECK 6: POLITICAL ORIENTATION")
print("="*80)

# Get political affiliation from original data (one per participant)
political_info = robustness_df.groupby('Participant_ID')['Political_Affiliation'].first().reset_index()

# Merge with participant_robust
participant_robust_pol = participant_robust.merge(political_info, on='Participant_ID', how='left')

# Simplify political categories
def simplify_politics(affiliation):
    if pd.isna(affiliation):
        return 'Other'
    elif 'Democratic' in affiliation:
        return 'Democrat'
    elif 'Republican' in affiliation:
        return 'Republican'
    elif affiliation == 'Independent':
        return 'Independent'
    else:
        return 'Other'

participant_robust_pol['Political_Simple'] = participant_robust_pol['Political_Affiliation'].apply(simplify_politics)

print("\nPolitical Orientation Distribution:")
print(participant_robust_pol['Political_Simple'].value_counts())

# Focus on main three groups
pol_data = participant_robust_pol[participant_robust_pol['Political_Simple'].isin(['Democrat', 'Republican', 'Independent'])]

# Test interaction
model3 = ols('Error_Reduction ~ C(Condition) * C(Political_Simple)', data=pol_data).fit()
anova_table3 = sm.stats.anova_lm(model3, typ=2)
print("\nCondition × Political Orientation Interaction:")
print(anova_table3)

# Separate analyses by political orientation
print("\n" + "-"*80)
print("EFFECTS BY POLITICAL ORIENTATION")
print("-"*80)

for political_group in ['Democrat', 'Republican', 'Independent']:
    pol_subset = pol_data[pol_data['Political_Simple'] == political_group]
    
    print(f"\n{political_group.upper()} (N = {len(pol_subset)}):")
    print("\nMean Error Reduction by Condition:")
    summary = pol_subset.groupby('Condition')['Error_Reduction'].agg(['mean', 'std', 'count'])
    print(summary.round(2))
    
    # ANOVA within political group
    conditions = [pol_subset[pol_subset['Condition'] == c]['Error_Reduction'].values 
                  for c in pol_subset['Condition'].unique()]
    
    # Check if all conditions have data
    if all(len(c) > 1 for c in conditions):
        f_stat, p_val = stats.f_oneway(*conditions)
        print(f"\nOne-way ANOVA: F = {f_stat:.2f}, p = {p_val:.4f}")

print("\n" + "="*80)
print("ROBUSTNESS CHECKS COMPLETE")
print("="*80)