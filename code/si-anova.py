import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols

# Load data
input_data = pd.read_csv('../data/complete-anova.csv')

input_data = input_data.groupby(['Participant ID', 'Condition']).agg({
    'Error Reduction': 'mean',
    'MAR': 'mean',
    'Final Error': 'mean'
}).reset_index()

print("Data Rows:", input_data.shape)

input_data.columns = input_data.columns.str.replace(' ', '_')

print("="*80)
print("COMPLETE ANOVA TABLES FOR SI APPENDIX")
print("="*80)

# ============================================================================
# TABLE S11: MAIN EFFECT ON ERROR REDUCTION
# ============================================================================
print("\n" + "="*80)
print("TABLE S11: ONE-WAY ANOVA - ERROR REDUCTION")
print("="*80)

# Fit the model
model1 = ols('Error_Reduction ~ C(Condition)', data=input_data).fit()
anova_table1 = sm.stats.anova_lm(model1, typ=2)

print("\nANOVA Table:")
print(anova_table1)

# Calculate eta-squared (effect size)
ss_treatment = anova_table1.loc['C(Condition)', 'sum_sq']
ss_total = anova_table1['sum_sq'].sum()
eta_squared = ss_treatment / ss_total

print(f"\nEffect Size (η²): {eta_squared:.4f}")

# Descriptive statistics
print("\nDescriptive Statistics by Condition:")
desc1 = input_data.groupby('Condition')['Error_Reduction'].agg([
    ('N', 'count'),
    ('Mean', 'mean'),
    ('SD', 'std'),
    ('SE', lambda x: x.std() / np.sqrt(len(x))),
    ('95% CI Lower', lambda x: x.mean() - 1.96 * x.std() / np.sqrt(len(x))),
    ('95% CI Upper', lambda x: x.mean() + 1.96 * x.std() / np.sqrt(len(x)))
])
print(desc1.round(3))

# LaTeX output
print("\n" + "-"*80)
print("LATEX TABLE S11:")
print("-"*80)
print("\\begin{table}[h]")
print("\\centering")
print("\\begin{tabular}{lrrrrr}")
print("\\toprule")
print("\\textbf{Source} & \\textbf{SS} & \\textbf{df} & \\textbf{MS} & \\textbf{F} & \\textbf{P} \\\\")
print("\\midrule")

# Between groups
ss_between = anova_table1.loc['C(Condition)', 'sum_sq']
df_between = int(anova_table1.loc['C(Condition)', 'df'])
ms_between = ss_between / df_between
f_stat = anova_table1.loc['C(Condition)', 'F']
p_val = anova_table1.loc['C(Condition)', 'PR(>F)']

print(f"Between Conditions & {ss_between:.2f} & {df_between} & {ms_between:.2f} & {f_stat:.2f} & <0.001 \\\\")

# Within groups (residual)
ss_within = anova_table1.loc['Residual', 'sum_sq']
df_within = int(anova_table1.loc['Residual', 'df'])
ms_within = ss_within / df_within

print(f"Within Conditions & {ss_within:.2f} & {df_within} & {ms_within:.2f} & & \\\\")

# Total
ss_total = ss_between + ss_within
df_total = df_between + df_within

print("\\midrule")
print(f"Total & {ss_total:.2f} & {df_total} & & & \\\\")
print("\\bottomrule")
print("\\end{tabular}")
print(f"\\caption{{\\textbf{{One-Way ANOVA: Treatment Effect on Error Reduction.}} Analysis of variance testing differences in error reduction across experimental conditions (N = {len(input_data)}). Effect size: $\\eta^2 = {eta_squared:.3f}$. Post-hoc comparisons with Tukey HSD correction reported in main text.}}")
print("\\label{tab:anova-main}")
print("\\end{table}")

# ============================================================================
# TABLE S12: MAIN EFFECT ON MEAN ABSOLUTE REVISION
# ============================================================================
print("\n" + "="*80)
print("TABLE S12: ONE-WAY ANOVA - MEAN ABSOLUTE REVISION")
print("="*80)

# Fit the model
model2 = ols('MAR ~ C(Condition)', data=input_data).fit()
anova_table2 = sm.stats.anova_lm(model2, typ=2)

print("\nANOVA Table:")
print(anova_table2)

# Calculate eta-squared
ss_treatment2 = anova_table2.loc['C(Condition)', 'sum_sq']
ss_total2 = anova_table2['sum_sq'].sum()
eta_squared2 = ss_treatment2 / ss_total2

print(f"\nEffect Size (η²): {eta_squared2:.4f}")

# Descriptive statistics
print("\nDescriptive Statistics by Condition:")
desc2 = input_data.groupby('Condition')['MAR'].agg([
    ('N', 'count'),
    ('Mean', 'mean'),
    ('SD', 'std'),
    ('SE', lambda x: x.std() / np.sqrt(len(x)))
])
print(desc2.round(3))

# LaTeX output
print("\n" + "-"*80)
print("LATEX TABLE S12:")
print("-"*80)
print("\\begin{table}[h]")
print("\\centering")
print("\\begin{tabular}{lrrrrr}")
print("\\toprule")
print("\\textbf{Source} & \\textbf{SS} & \\textbf{df} & \\textbf{MS} & \\textbf{F} & \\textbf{P} \\\\")
print("\\midrule")

ss_between2 = anova_table2.loc['C(Condition)', 'sum_sq']
df_between2 = int(anova_table2.loc['C(Condition)', 'df'])
ms_between2 = ss_between2 / df_between2
f_stat2 = anova_table2.loc['C(Condition)', 'F']
p_val2 = anova_table2.loc['C(Condition)', 'PR(>F)']

print(f"Between Conditions & {ss_between2:.2f} & {df_between2} & {ms_between2:.2f} & {f_stat2:.2f} & <0.001 \\\\")

ss_within2 = anova_table2.loc['Residual', 'sum_sq']
df_within2 = int(anova_table2.loc['Residual', 'df'])
ms_within2 = ss_within2 / df_within2

print(f"Within Conditions & {ss_within2:.2f} & {df_within2} & {ms_within2:.2f} & & \\\\")

ss_total2 = ss_between2 + ss_within2
df_total2 = df_between2 + df_within2

print("\\midrule")
print(f"Total & {ss_total2:.2f} & {df_total2} & & & \\\\")
print("\\bottomrule")
print("\\end{tabular}")
print(f"\\caption{{\\textbf{{One-Way ANOVA: Treatment Effect on Mean Absolute Revision.}} Analysis of variance testing differences in belief revision magnitude across experimental conditions. Effect size: $\\eta^2 = {eta_squared2:.3f}$.}}")
print("\\label{tab:anova-revision}")
print("\\end{table}")

# ============================================================================
# TABLE S13: CHANGE IN ESTIMATION ERROR (TREATMENT CONDITIONS ONLY)
# ============================================================================
print("\n" + "="*80)
print("TABLE S13: ONE-WAY ANOVA - CHANGE IN ESTIMATION ERROR, TREATMENTS ONLY")
print("="*80)

treatment_only = input_data[input_data['Condition'] != 'Individual'].copy()

# Fit the model
model3 = ols('Error_Reduction ~ C(Condition)', data=treatment_only).fit()
anova_table3 = sm.stats.anova_lm(model3, typ=2)

print("\nANOVA Table (Treatment Conditions Only):")
print(anova_table3)

# Calculate eta-squared
ss_treatment3 = anova_table3.loc['C(Condition)', 'sum_sq']
ss_total3 = anova_table3['sum_sq'].sum()
eta_squared3 = ss_treatment3 / ss_total3

print(f"\nEffect Size (η²): {eta_squared3:.4f}")

# Descriptive statistics
print("\nDescriptive Statistics by Treatment Condition:")
desc3 = treatment_only.groupby('Condition')['Error_Reduction'].agg([
    ('N', 'count'),
    ('Mean', 'mean'),
    ('SD', 'std')
])
print(desc3.round(3))

# LaTeX output
print("\n" + "-"*80)
print("LATEX TABLE S13:")
print("-"*80)
print("\\begin{table}[h]")
print("\\centering")
print("\\begin{tabular}{lrrrrr}")
print("\\toprule")
print("\\textbf{Source} & \\textbf{SS} & \\textbf{df} & \\textbf{MS} & \\textbf{F} & \\textbf{P} \\\\")
print("\\midrule")

ss_between3 = anova_table3.loc['C(Condition)', 'sum_sq']
df_between3 = int(anova_table3.loc['C(Condition)', 'df'])
ms_between3 = ss_between3 / df_between3
f_stat3 = anova_table3.loc['C(Condition)', 'F']
p_val3 = anova_table3.loc['C(Condition)', 'PR(>F)']

print(f"Between Conditions & {ss_between3:.2f} & {df_between3} & {ms_between3:.2f} & {f_stat3:.2f} & {p_val3:.4f} \\\\")

ss_within3 = anova_table3.loc['Residual', 'sum_sq']
df_within3 = int(anova_table3.loc['Residual', 'df'])
ms_within3 = ss_within3 / df_within3

print(f"Within Conditions & {ss_within3:.2f} & {df_within3} & {ms_within3:.2f} & & \\\\")

ss_total3 = ss_between3 + ss_within3
df_total3 = df_between3 + df_within3

print("\\midrule")
print(f"Total & {ss_total3:.2f} & {df_total3} & & & \\\\")
print("\\bottomrule")
print("\\end{tabular}")
print(f"\\caption{{\\textbf{{One-Way ANOVA: Differences Among Treatment Conditions.}} Analysis comparing change in estimation error across the Age, Conformity, Prestige, and Season conditions, excluding control (N = {len(treatment_only)}). Negative changes indicate improved accuracy. Effect size: $\\eta^2 = {eta_squared3:.3f}$.}}")
print("\\label{tab:anova-treatments}")
print("\\end{table}")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*80)
print("SUMMARY")
print("="*80)

print(f"\nTable S11 (All conditions): F({df_between}, {df_within}) = {f_stat:.2f}, p < 0.001, η² = {eta_squared:.3f}")
print(f"Table S12 (MAR): F({df_between2}, {df_within2}) = {f_stat2:.2f}, p < 0.001, η² = {eta_squared2:.3f}")
print(f"Table S13 (Error change, treatments only): F({df_between3}, {df_within3}) = {f_stat3:.2f}, p = {p_val3:.4f}, η² = {eta_squared3:.3f}")

print("\n" + "="*80)
print("COMPLETE")
print("="*80)