import numpy as np
import pandas as pd
from scipy import stats

# load estimates.csv into pandas dataframe
estimates_df = pd.read_csv('../data/estimates.csv')
estimates_df.head()

####################################################
################# Self Assessment ##################
####################################################
# =============================================================================
# purpose: summarize participants' self-reported agreement with each statement
#          and prepare the Figure 1A-B input table
# output:
#   - assessment_df with one row per statement and percentage distributions
#     across Agreement-Text response categories
# note:
#   - percentages are computed within each statement
#   - output is saved to ../output/figure1a-b.csv
# =============================================================================
assessment_df = pd.DataFrame()
for statement_id in estimates_df['Statement ID'].unique():
    var_statement = estimates_df[estimates_df['Statement ID']==statement_id]['Agreement-Text'].value_counts(normalize=True) * 100
    var_statement_df = var_statement.reset_index().T
    var_statement_df.columns = var_statement_df.iloc[0]
    var_statement_df = var_statement_df.drop(var_statement_df.index[0])
    var_statement_df.insert(0, 'Statement ID', statement_id)
    assessment_df = pd.concat([assessment_df, var_statement_df], ignore_index=True)
assessment_df.index.name = None
assessment_df.to_csv('../output/figure1a-b.csv', index=False)

print("\n" + "="*80)
print("Figure 1A-B: Self-Assessment of Agreement with Each Statement")
print("="*80)
print (assessment_df.to_string(index=False))

####################################################
################# Misperceptions ###################
####################################################
# =============================================================================
# purpose: recode all 12 statements to a common gender-inequity scale and
#          compute signed misperception for each participant-statement response
# measure:
#   - Misperception % = perceived support for gender inequity minus actual
#     support for gender inequity, in percentage points
# interpretation:
#   - positive values indicate overestimation of support for gender inequity
#   - negative values indicate underestimation
# coding:
#   - statements 1-6 are equitable as worded and must be reversed
#   - statements 7-12 are inequitable as worded and are kept as they are
# note:
#   - after recoding, higher values always indicate greater support for gender
#     inequity across all 12 statements
# =============================================================================
# statements 1-6 are equitable as worded and need to be reversed
equitable_ids = [1, 2, 3, 4, 5, 6]

# observed agreement with each statement as worded
agreement_df = assessment_df[['Statement ID']].copy()
agreement_df['Agreement'] = (
    pd.to_numeric(assessment_df['Strongly agree'], errors='raise') +
    pd.to_numeric(assessment_df['Agree'], errors='raise')
)

# create misperception dataframe
misperception_df = estimates_df[
    ['Participant ID', 'Condition', 'Statement ID', 'Initial Estimate', 'Final Estimate']
].copy()

misperception_df = misperception_df.merge(
    agreement_df[['Statement ID', 'Agreement']],
    on='Statement ID',
    how='left'
)

# convert estimates and observed agreement to numeric values
for col in ['Initial Estimate', 'Final Estimate', 'Agreement']:
    misperception_df[col] = pd.to_numeric(misperception_df[col], errors='raise')

# reverse equitable statements so higher values always indicate greater support
# for gender inequity
mask_equitable = misperception_df['Statement ID'].isin(equitable_ids)

misperception_df.loc[mask_equitable, 'Initial Estimate'] = (
    100 - misperception_df.loc[mask_equitable, 'Initial Estimate']
)
misperception_df.loc[mask_equitable, 'Final Estimate'] = (
    100 - misperception_df.loc[mask_equitable, 'Final Estimate']
)
misperception_df.loc[mask_equitable, 'Agreement'] = (
    100 - misperception_df.loc[mask_equitable, 'Agreement']
)

# signed misperception in gender-inequity space
misperception_df['Misperception %'] = (
    misperception_df['Initial Estimate'] - misperception_df['Agreement']
)

# =============================================================================
# purpose: summarize average participant-level misperception across all
#          12 statements
# measure:
#   - participant-level average misperception across the 12 statements
# output:
#   - printed mean, SD, 95% CI, and N for the participant-level average
# =============================================================================
def mean_std_ci(data):
    mean = data.mean()
    std = data.std()
    n = len(data)
    se = std / (n ** 0.5)
    ci_lower = mean - 1.96 * se
    ci_upper = mean + 1.96 * se
    return mean, std, (ci_lower, ci_upper)

participant_misperception_df = misperception_df.groupby(
    'Participant ID',
    as_index=False
).agg({
    'Misperception %': 'mean'
})

participant_misperception_stats = mean_std_ci(
    participant_misperception_df['Misperception %']
)

print("\n" + "="*80)
print("Overall Participant-Level Baseline Misperception")
print("="*80)
print(
    'Mean Misperception: {:.2f}, SD={:.2f}, 95% CI=({:.2f}, {:.2f}), N={}'.format(
        participant_misperception_stats[0],
        participant_misperception_stats[1],
        participant_misperception_stats[2][0],
        participant_misperception_stats[2][1],
        participant_misperception_df['Participant ID'].nunique()
    )
)

# =============================================================================
# purpose: summarize statement-level misperception and test whether the mean
#          misperception for each statement differs from zero
# test:
#   - one-sample t-test for each statement against 0
# hypotheses:
#   - H0: mean misperception = 0 (accurate perception)
#   - H1: mean misperception ≠ 0 (systematic bias)
# output:
#   - t_test_df with statement-level descriptive and inferential results
#   - LaTeX table saved to ../output/t_test_results.tex
# =============================================================================
t_test_results = []

for statement_id in sorted(misperception_df['Statement ID'].unique()):
    subset = misperception_df[misperception_df['Statement ID'] == statement_id].copy()

    misperception_values = subset['Misperception %']

    t_stat, p_value = stats.ttest_1samp(misperception_values, 0)

    # descriptive statistics for reporting
    mean_misperception, std_misperception, ci_misperception = mean_std_ci(misperception_values)

    # one-sample Cohen's d
    cohen = mean_misperception / std_misperception if std_misperception != 0 else float('inf')

    t_test_results.append({
        'Statement ID': statement_id,
        'Mean Misperception': mean_misperception,
        'SD Misperception': std_misperception,
        'CI Lower': ci_misperception[0],
        'CI Upper': ci_misperception[1],
        'T-Statistic': t_stat,
        'P-Value': p_value,
        "Cohen's d": cohen
    })

t_test_df = pd.DataFrame(t_test_results)

statement_mean_stats = mean_std_ci(t_test_df['Mean Misperception'])

print("\n" + "="*80)
print("Overall Statement-Level Mean Misperception")
print("="*80)
print(
    'Mean of 12 statement means: {:.2f}, SD across statements={:.2f}, 95% CI=({:.2f}, {:.2f}), N={}'.format(
        statement_mean_stats[0],
        statement_mean_stats[1],
        statement_mean_stats[2][0],
        statement_mean_stats[2][1],
        len(t_test_df)
    )
)

print("\n" + "="*80)
print("Statement-Level Misperception Tests")
print("="*80)
print(t_test_df.to_string(index=False))

# save statement-level test results as a LaTeX table
latex_table = t_test_df.to_latex(index=False, float_format="%.3f")
with open('../output/t_test_results.tex', 'w') as f:
    f.write(latex_table)
# =============================================================================
# purpose: save Figure 1C data as statement-level means by condition
# =============================================================================

agreement_cols = ["Statement ID", "Condition", "Agreement", "Initial Estimate"]

misperceptions_out = (
    misperception_df[agreement_cols]
    .groupby(["Statement ID", "Condition"], as_index=False)
    .mean()
    .sort_values(["Statement ID", "Condition"])
    .reset_index(drop=True)
)

misperceptions_out.to_csv('../output/figure1c-d.csv', index=False)

print("\n" + "="*80)
print("Figure 1C Output: Actual and Perceived Support for Gender Inequity")
print("="*80)
print(misperceptions_out.to_string(index=False))

###################################################################################
################ Belief Revision (Control vs. Pooled Treatment) ###################
###################################################################################
# =============================================================================
# purpose: compute MAR and revision rate, then aggregate them to the 
# participant level for Figure 3
# measures:
#   - MAR = absolute change between final and initial estimates
#   - Revised? = 1 if the estimate changed, 0 otherwise
# grouping:
#   - Group = control for Individual and treatment for all social-information
#     conditions combined
# =============================================================================
revision_df = misperception_df[
    ['Participant ID', 'Condition', 'Statement ID', 'Initial Estimate', 'Final Estimate', 'Agreement']
].copy()

# pooled control versus treatment grouping
revision_df['Group'] = revision_df['Condition'].apply(
    lambda x: 'control' if x == 'Individual' else 'treatment'
)

# statement-level revision measures
revision_df['MAR'] = abs(revision_df['Final Estimate'] - revision_df['Initial Estimate'])
revision_df['Revised?'] = (revision_df['MAR'] > 0).astype(int)

# participant-level summary for Figure 3 statistics
participant_revision_df = revision_df.groupby(
    ['Participant ID', 'Condition', 'Group'],
    as_index=False
).agg({
    'MAR': 'mean',
    'Revised?': 'mean'
})

# express participant-level revision rate as a percentage
participant_revision_df['Revision Rate'] = participant_revision_df['Revised?'] * 100

print("\n" + "="*80)
print("Figure 3: Revision Data Prepared")
print("="*80)
print("Statement-level revision_df")
print(revision_df.head().to_string(index=False))

print("\nParticipant-level participant_revision_df")
print(participant_revision_df.head().to_string(index=False))

# =============================================================================
# purpose: compare MAR between control and treatment for Figure 3 Panel A
# measure:
#   - MAR = mean absolute revision across the 12 statements for each participant
# tests:
#   - Welch t-test for the mean difference between groups
#   - Kruskal-Wallis test as a non-parametric robustness check
# effect sizes:
#   - Cohen's d
#   - epsilon-squared for Kruskal-Wallis
# output:
#   - figure3a.csv with participant-level MAR by group
# =============================================================================

# save participant-level MAR for the figure
participant_revision_df[["Group", "MAR"]].to_csv('../output/figure3a.csv', index=False)

print("\n" + "="*80)
print("Figure 3 Panel A: Control vs. Treatment on MAR")
print("="*80)

control_mar = participant_revision_df[participant_revision_df['Group'] == 'control']['MAR']
treatment_mar = participant_revision_df[participant_revision_df['Group'] == 'treatment']['MAR']

control_mar_stats = mean_std_ci(control_mar)
treatment_mar_stats = mean_std_ci(treatment_mar)

print("Mean Absolute Revision (participant-level):")
print(
    'Control MAR: {:.2f}, SD={:.2f}, 95% CI=({:.2f}, {:.2f})'.format(
        control_mar_stats[0], control_mar_stats[1],
        control_mar_stats[2][0], control_mar_stats[2][1]
    )
)
print(
    'Treatment MAR: {:.2f}, SD={:.2f}, 95% CI=({:.2f}, {:.2f})'.format(
        treatment_mar_stats[0], treatment_mar_stats[1],
        treatment_mar_stats[2][0], treatment_mar_stats[2][1]
    )
)

# Welch t-test for participant-level MAR
t_stat, p_value = stats.ttest_ind(control_mar, treatment_mar, equal_var=False)
print(f"Welch t-statistic: {t_stat:.2f}, P-value: {p_value:.4f}")

# Cohen's d
def cohen_d(x, y):
    nx = len(x)
    ny = len(y)
    dof = nx + ny - 2
    pooled_std = (((nx - 1) * x.std() ** 2 + (ny - 1) * y.std() ** 2) / dof) ** 0.5
    d = (x.mean() - y.mean()) / pooled_std
    return d

d = cohen_d(control_mar, treatment_mar)
print(f"Cohen's d: {d:.2f}")

# Kruskal-Wallis test for participant-level MAR
h_stat, p_value_kw = stats.kruskal(control_mar, treatment_mar)
print(f"Kruskal-Wallis H-statistic: {h_stat:.2f}, P-value: {p_value_kw:.4f}")

# epsilon-squared effect size for Kruskal-Wallis
n = len(participant_revision_df)
k = 2
epsilon_squared = (h_stat - k + 1) / (n - k)
print(f"Epsilon-squared (effect size): {epsilon_squared:.4f}")

# =================================================================================
# purpose: compare revision rate between control and treatment for Figure 3 Panel B
# measure:
#   - Revision Rate = percentage of the 12 statements revised by each participant
# tests:
#   - Welch t-test for the mean difference between groups
#   - Mann-Whitney U test as a non-parametric comparison
#   - Kruskal-Wallis test as a non-parametric robustness check
# effect size:
#   - Cohen's d
# output:
#   - figure3b.csv with participant-level revision rate by group
# =================================================================================
# save participant-level revision rate for the figure
participant_revision_df[["Group", "Revision Rate"]].to_csv('../output/figure3b.csv', index=False)

print("\n" + "="*80)
print("Figure 3 Panel B: Control vs. Treatment on Revision Rate")
print("="*80)

control_revision_rate = participant_revision_df[
    participant_revision_df['Group'] == 'control'
]['Revision Rate']

treatment_revision_rate = participant_revision_df[
    participant_revision_df['Group'] == 'treatment'
]['Revision Rate']

control_revision_rate_stats = mean_std_ci(control_revision_rate)
treatment_revision_rate_stats = mean_std_ci(treatment_revision_rate)

print("Revision Rate (participant-level):")
print(
    'Control Revision Rate: {:.2f}, SD={:.2f}, 95% CI=({:.2f}, {:.2f})'.format(
        control_revision_rate_stats[0], control_revision_rate_stats[1],
        control_revision_rate_stats[2][0], control_revision_rate_stats[2][1]
    )
)
print(
    'Treatment Revision Rate: {:.2f}, SD={:.2f}, 95% CI=({:.2f}, {:.2f})'.format(
        treatment_revision_rate_stats[0], treatment_revision_rate_stats[1],
        treatment_revision_rate_stats[2][0], treatment_revision_rate_stats[2][1]
    )
)

# Welch t-test for participant-level revision rate
t_stat, p_value = stats.ttest_ind(
    control_revision_rate,
    treatment_revision_rate,
    equal_var=False
)
print(f"Welch t-statistic: {t_stat:.2f}, P-value: {p_value:.4f}")

# Mann-Whitney U test for participant-level revision rate
u_stat, p_value_mw = stats.mannwhitneyu(
    control_revision_rate,
    treatment_revision_rate,
    alternative='two-sided'
)
print(f"Mann-Whitney U-statistic: {u_stat:.2f}, P-value: {p_value_mw:.4f}")

# Cohen's d
d = cohen_d(control_revision_rate, treatment_revision_rate)
print(f"Cohen's d: {d:.2f}")

# Kruskal-Wallis test for participant-level revision rate
h_stat, p_value_kw = stats.kruskal(
    control_revision_rate,
    treatment_revision_rate
)
print(f"Kruskal-Wallis H-statistic: {h_stat:.2f}, P-value: {p_value_kw:.4f}")

# epsilon-squared effect size for Kruskal-Wallis
n = len(participant_revision_df)
k = 2
epsilon_squared = (h_stat - k + 1) / (n - k)
print(f"Epsilon-squared (effect size): {epsilon_squared:.4f}")

# =============================================================================
# purpose: run supplementary item-level revision-rate tests for Figure 3
# note:
#   - this block is a robustness check only
#   - the main Figure 3 analysis is based on participant-level revision rates
# measure:
#   - Revised? = 1 if a statement-level estimate changed, 0 otherwise
# tests:
#   - two-proportion z-test
#   - chi-square test
# output:
#   - printed item-level revision counts, rates, and test results
# =============================================================================
print("\n" + "="*80)
print("Figure 3 Panel B: Supplementary Item-Level Revision-Rate Tests")
print("="*80)

control_items = revision_df[revision_df['Group'] == 'control']['Revised?']
treatment_items = revision_df[revision_df['Group'] == 'treatment']['Revised?']

control_revised = int(control_items.sum())
treatment_revised = int(treatment_items.sum())

control_total = int(len(control_items))
treatment_total = int(len(treatment_items))

control_rate = control_revised / control_total * 100
treatment_rate = treatment_revised / treatment_total * 100

print("Revision Rate (item-level):")
print(f"Control revision rate: {control_rate:.2f}%")
print(f"Treatment revision rate: {treatment_rate:.2f}%")

# two-proportion z-test
from statsmodels.stats.proportion import proportions_ztest

count = [control_revised, treatment_revised]
nobs = [control_total, treatment_total]
z_stat, p_value_z = proportions_ztest(count, nobs)

print(f"Two-proportion z-statistic: {z_stat:.2f}, P-value: {p_value_z:.4f}")

# chi-square test
from scipy.stats import chi2_contingency

contingency_table = pd.DataFrame({
    'Revised': [control_revised, treatment_revised],
    'Not Revised': [control_total - control_revised, treatment_total - treatment_revised]
}, index=['Control', 'Treatment'])

chi2, p_value_chi, dof, expected = chi2_contingency(contingency_table)

print(f"Chi-square statistic: {chi2:.2f}, P-value: {p_value_chi:.4f}")

############################################################################
################ Belief Revision (Control vs. Treatment) ###################
############################################################################

# =============================================================================
# purpose: prepare statement-level and participant-level data for Figure 4
# measures:
#   - MAR = absolute change between final and initial estimates
#   - Revision Rate = percentage of the 12 statements revised by each participant
#   - Influence Score = proportion of the initial gap to the social reference
#     that is closed after revision
# note:
#   - Influence Score is defined only for the four treatment conditions
#   - zero-gap items are treated as undefined and excluded from the participant
#     mean influence score
#   - Influence Score is computed in raw agreement space, using raw estimates
#     from estimates_df and displayed reference values from statements.csv
# =============================================================================
# start from revision_df so MAR and Revised? remain unchanged
correction_df = revision_df[
    ['Participant ID', 'Condition', 'Group', 'Statement ID',
     'Initial Estimate', 'Final Estimate', 'MAR', 'Revised?']
].copy()

# bring in raw agreement-space estimates for influence-score calculation
raw_estimates_df = estimates_df[
    ['Participant ID', 'Condition', 'Statement ID', 'Initial Estimate', 'Final Estimate']
].copy().rename(columns={
    'Initial Estimate': 'Raw Initial Estimate',
    'Final Estimate': 'Raw Final Estimate'
})

raw_estimates_df['Raw Initial Estimate'] = pd.to_numeric(
    raw_estimates_df['Raw Initial Estimate'], errors='raise'
)
raw_estimates_df['Raw Final Estimate'] = pd.to_numeric(
    raw_estimates_df['Raw Final Estimate'], errors='raise'
)

correction_df = correction_df.merge(
    raw_estimates_df,
    on=['Participant ID', 'Condition', 'Statement ID'],
    how='left'
)

# merge displayed reference values
statements = pd.read_csv('../data/statements.csv')
correction_df = correction_df.merge(
    statements[['Statement ID', 'Reference']],
    on='Statement ID',
    how='left'
)

correction_df['Reference'] = pd.to_numeric(correction_df['Reference'], errors='raise')

# influence score for treatment conditions only, computed in raw agreement space
social_mask = correction_df['Condition'] != 'Individual'
denom = correction_df['Reference'] - correction_df['Raw Initial Estimate']

correction_df['Influence Score'] = np.nan
correction_df.loc[social_mask, 'Influence Score'] = (
    (correction_df.loc[social_mask, 'Raw Final Estimate'] - correction_df.loc[social_mask, 'Raw Initial Estimate']) /
    denom.loc[social_mask]
)

# zero-gap items are undefined and excluded from participant-level means
zero_gap_mask = social_mask & (denom == 0)
correction_df.loc[zero_gap_mask, 'Influence Score'] = np.nan

# participant-level summary for the main Figure 4 analyses
participant_summary = correction_df.groupby(
    ['Participant ID', 'Condition', 'Group'],
    as_index=False
).agg({
    'MAR': 'mean',
    'Revised?': 'mean',
    'Influence Score': 'mean'
})

participant_summary['Revision Rate'] = participant_summary['Revised?'] * 100

# number of participants excluded from the participant-level influence analysis
participants_excluded_from_influence = participant_summary[
    (participant_summary['Condition'] != 'Individual') &
    (participant_summary['Influence Score'].isna())
]['Participant ID'].nunique()

print("\n" + "="*80)
print("Figure 4: Condition-Specific Data Prepared")
print("="*80)
print(f"Participants excluded from the participant-level influence analysis: {participants_excluded_from_influence}")

print("\nStatement-level correction_df")
print(correction_df.head().to_string(index=False))

print("\nParticipant-level participant_summary")
print(participant_summary.head().to_string(index=False))

# =============================================================================
# purpose: summarize revision rate by condition for Figure 4 Panel A
# tests:
#   - one-way ANOVA across all conditions
#   - Kruskal-Wallis test as a non-parametric robustness check
#   - Tukey HSD post-hoc comparisons if the ANOVA is significant
# output:
#   - figure4a.csv with condition-level participant revision-rate summaries
# =============================================================================
print("\n" + "="*80)
print("Figure 4 Panel A: Revision Rate by Condition")
print("="*80)

# condition-level participant revision-rate summaries
figure_4a_df = pd.DataFrame(columns=['Condition', 'Revision Rate', 'N', 'lower', 'upper'])

for condition in participant_summary['Condition'].unique():
    condition_df = participant_summary[participant_summary['Condition'] == condition]
    revision_rate_stats = mean_std_ci(condition_df['Revision Rate'])

    figure_4a_df.loc[len(figure_4a_df)] = [
        condition,
        revision_rate_stats[0],
        condition_df['Participant ID'].nunique(),
        revision_rate_stats[2][0],
        revision_rate_stats[2][1]
    ]

    print(
        f"Condition: {condition} - Revision Rate: {revision_rate_stats[0]:.2f}%, "
        f"95% CI: ({revision_rate_stats[2][0]:.2f}, {revision_rate_stats[2][1]:.2f}), "
        f"N = {condition_df['Participant ID'].nunique()}"
    )

figure_4a_df.to_csv('../output/figure4a.csv', index=False)

# omnibus participant-level tests across all conditions
revision_rate_groups = [
    participant_summary[participant_summary['Condition'] == condition]['Revision Rate']
    for condition in participant_summary['Condition'].unique()
]

f_stat, p_value = stats.f_oneway(*revision_rate_groups)
print(f"\nOne-way ANOVA: F = {f_stat:.2f}, P-value = {p_value:.4f}")

h_stat, p_value_kw = stats.kruskal(*revision_rate_groups)
print(f"Kruskal-Wallis H-statistic: {h_stat:.2f}, P-value = {p_value_kw:.4f}")

# post-hoc Tukey HSD if the ANOVA is significant
from statsmodels.stats.multicomp import pairwise_tukeyhsd

if p_value < 0.05:
    tukey = pairwise_tukeyhsd(
        endog=participant_summary['Revision Rate'],
        groups=participant_summary['Condition'],
        alpha=0.05
    )
    print("\nTukey HSD Post-hoc Comparisons:")
    print(tukey)

# =============================================================================
# purpose: summarize MAR by condition for Figure 4 Panel B
# tests:
#   - one-way ANOVA across all conditions
#   - Kruskal-Wallis test as a non-parametric robustness check
#   - Tukey HSD post-hoc comparisons if the ANOVA is significant
#   - Welch t-tests comparing each treatment condition with control
#     (BH-FDR corrected across the four treatment-versus-control tests)
# output:
#   - figure4b.csv with condition-level participant MAR summaries
# =============================================================================
print("\n" + "="*80)
print("Figure 4 Panel B: Mean Absolute Revision by Condition")
print("="*80)

# condition-level participant MAR summaries
figure_4b_df = pd.DataFrame(columns=['Condition', 'MAR', 'N', 'lower', 'upper'])

for condition in participant_summary['Condition'].unique():
    condition_df = participant_summary[participant_summary['Condition'] == condition]
    mar_stats = mean_std_ci(condition_df['MAR'])

    figure_4b_df.loc[len(figure_4b_df)] = [
        condition,
        mar_stats[0],
        condition_df['Participant ID'].nunique(),
        mar_stats[2][0],
        mar_stats[2][1]
    ]

    print(
        f"Condition: {condition} - MAR: {mar_stats[0]:.2f}, "
        f"95% CI: ({mar_stats[2][0]:.2f}, {mar_stats[2][1]:.2f}), "
        f"N = {condition_df['Participant ID'].nunique()}"
    )

# omnibus participant-level tests across all conditions
mar_groups = [
    participant_summary[participant_summary['Condition'] == condition]['MAR']
    for condition in participant_summary['Condition'].unique()
]

f_stat, p_value = stats.f_oneway(*mar_groups)
print(f"\nOne-way ANOVA: F = {f_stat:.2f}, P-value = {p_value:.4f}")

h_stat, p_value_kw = stats.kruskal(*mar_groups)
print(f"Kruskal-Wallis H-statistic: {h_stat:.2f}, P-value = {p_value_kw:.4f}")

# post-hoc Tukey HSD if the ANOVA is significant
from statsmodels.stats.multicomp import pairwise_tukeyhsd

if p_value < 0.05:
    tukey = pairwise_tukeyhsd(
        endog=participant_summary['MAR'],
        groups=participant_summary['Condition'],
        alpha=0.05
    )
    print("\nTukey HSD Post-hoc Comparisons:")
    print(tukey)

# Welch treatment-versus-control tests with BH-FDR correction
from statsmodels.stats.multitest import multipletests

control_mar = participant_summary[
    participant_summary['Condition'] == 'Individual'
]['MAR']

social_conditions = ['Season', 'Prestige', 'Conformity', 'Age']

welch_rows = []
raw_pvals = []

for condition in social_conditions:
    treatment_mar = participant_summary[
        participant_summary['Condition'] == condition
    ]['MAR']

    t_stat, p_raw = stats.ttest_ind(
        treatment_mar,
        control_mar,
        equal_var=False
    )

    welch_rows.append({
        'Condition': condition,
        't_statistic': t_stat,
        'p_raw': p_raw
    })
    raw_pvals.append(p_raw)

reject_bh, pvals_bh, _, _ = multipletests(raw_pvals, alpha=0.05, method='fdr_bh')

for i in range(len(welch_rows)):
    welch_rows[i]['p_bh_fdr'] = pvals_bh[i]
    welch_rows[i]['reject_bh_fdr'] = reject_bh[i]

    if pvals_bh[i] < 0.001:
        welch_rows[i]['stars_bh_fdr'] = '***'
    elif pvals_bh[i] < 0.01:
        welch_rows[i]['stars_bh_fdr'] = '**'
    elif pvals_bh[i] < 0.05:
        welch_rows[i]['stars_bh_fdr'] = '*'
    else:
        welch_rows[i]['stars_bh_fdr'] = ''

mar_vs_control_df = pd.DataFrame(welch_rows)

print("\nWelch t-tests: treatment vs control (BH-FDR corrected)")
print(mar_vs_control_df.to_string(index=False))

# add Welch-vs-control BH-FDR results into figure4b.csv without changing the
# existing condition-level summary values
figure_4b_df = figure_4b_df.merge(
    mar_vs_control_df,
    on='Condition',
    how='left'
)

figure_4b_df.to_csv('../output/figure4b.csv', index=False)

# =============================================================================
# purpose: summarize influence score by condition for Figure 4 Panel C
# note:
#   - Control is included in the output as a blank row for figure layout
#   - statistical tests are conducted only across the four treatment conditions
# tests:
#   - one-way ANOVA across treatment conditions
#   - Kruskal-Wallis test as a non-parametric robustness check
#   - Tukey HSD post-hoc comparisons if the ANOVA is significant
# output:
#   - figure4c.csv with condition-level participant influence summaries
# =============================================================================
print("\n" + "="*80)
print("Figure 4 Panel C: Influence Score by Condition")
print("="*80)

conditions_order = ['Individual', 'Age', 'Conformity', 'Prestige', 'Season']

# condition-level participant influence summaries
figure_4c_df = pd.DataFrame(columns=['Condition', 'Influence Score', 'N', 'lower', 'upper'])

for condition in conditions_order:
    condition_df = participant_summary[participant_summary['Condition'] == condition]

    if condition == 'Individual':
        figure_4c_df.loc[len(figure_4c_df)] = [condition, np.nan, condition_df['Participant ID'].nunique(), np.nan, np.nan]
        print(f"Condition: {condition} - Influence Score: not defined for control")
    else:
        valid_influence = condition_df['Influence Score'].dropna()

        influence_stats = mean_std_ci(valid_influence)

        figure_4c_df.loc[len(figure_4c_df)] = [
            condition,
            influence_stats[0],
            valid_influence.shape[0],
            influence_stats[2][0],
            influence_stats[2][1]
        ]

        print(
            f"Condition: {condition} - Influence Score: {influence_stats[0]:.2f}, "
            f"95% CI: ({influence_stats[2][0]:.2f}, {influence_stats[2][1]:.2f}), "
            f"N = {valid_influence.shape[0]}"
        )

figure_4c_df.to_csv('../output/figure4c.csv', index=False)

# omnibus participant-level tests across treatment conditions only
treatment_conditions = ['Age', 'Conformity', 'Prestige', 'Season']
influence_groups = [
    participant_summary[
        (participant_summary['Condition'] == condition) &
        (participant_summary['Influence Score'].notna())
    ]['Influence Score']
    for condition in treatment_conditions
]

f_stat, p_value = stats.f_oneway(*influence_groups)
print(f"\nOne-way ANOVA (treatments only): F = {f_stat:.2f}, P-value = {p_value:.4f}")

h_stat, p_value_kw = stats.kruskal(*influence_groups)
print(f"Kruskal-Wallis H-statistic (treatments only): {h_stat:.2f}, P-value = {p_value_kw:.4f}")

# post-hoc Tukey HSD if the ANOVA is significant
from statsmodels.stats.multicomp import pairwise_tukeyhsd

if p_value < 0.05:
    tukey_df = participant_summary[
        (participant_summary['Condition'] != 'Individual') &
        (participant_summary['Influence Score'].notna())
    ].copy()

    tukey = pairwise_tukeyhsd(
        endog=tukey_df['Influence Score'],
        groups=tukey_df['Condition'],
        alpha=0.05
    )
    print("\nTukey HSD Post-hoc Comparisons:")
    print(tukey)

#####################################################
################ Estimation Error ###################
#####################################################
# =============================================================================
# purpose: prepare statement-level and participant-level estimation-error data
#          for Figure 5
# measures:
#   - Initial Error = absolute difference between the initial estimate and the
#     actual norm for that statement
#   - Final Error = absolute difference between the final estimate and the
#     actual norm for that statement
#   - Estimation Error Change = Final Error - Initial Error
# interpretation:
#   - negative values indicate improvement in estimation accuracy
#   - positive values indicate worse accuracy after revision
# note:
#   - Figure 5 is computed in raw agreement space, not the recoded
#     gender-inequity space used for Figure 1 misperception
#   - both overestimation and underestimation count as inaccuracy
# =============================================================================
# raw actual norm for each statement as the observed proportion agreeing
# with the statement as worded
raw_agreement_df = assessment_df[['Statement ID']].copy()
raw_agreement_df['Agreement'] = (
    pd.to_numeric(assessment_df['Strongly agree'], errors='raise') +
    pd.to_numeric(assessment_df['Agree'], errors='raise')
)

# start from raw agreement-space estimates
correction_df = estimates_df[
    ['Participant ID', 'Condition', 'Statement ID', 'Initial Estimate', 'Final Estimate']
].copy()

# pooled control versus treatment grouping
correction_df['Group'] = correction_df['Condition'].apply(
    lambda x: 'control' if x == 'Individual' else 'treatment'
)

# convert raw estimates to numeric values
correction_df['Initial Estimate'] = pd.to_numeric(
    correction_df['Initial Estimate'], errors='raise'
)
correction_df['Final Estimate'] = pd.to_numeric(
    correction_df['Final Estimate'], errors='raise'
)

# merge raw observed agreement with each statement
correction_df = correction_df.merge(
    raw_agreement_df[['Statement ID', 'Agreement']],
    on='Statement ID',
    how='left'
)

correction_df['Agreement'] = pd.to_numeric(
    correction_df['Agreement'], errors='raise'
)

# statement-level revision measures
correction_df['MAR'] = abs(
    correction_df['Final Estimate'] - correction_df['Initial Estimate']
)
correction_df['Revised?'] = (correction_df['MAR'] > 0).astype(int)

# statement-level estimation error relative to the actual norm
correction_df['Initial Error'] = abs(
    correction_df['Initial Estimate'] - correction_df['Agreement']
)
correction_df['Final Error'] = abs(
    correction_df['Final Estimate'] - correction_df['Agreement']
)

# negative values indicate improvement and positive values indicate getting worse
correction_df['Estimation Error Change'] = (
    correction_df['Final Error'] - correction_df['Initial Error']
)

# participant-level summary for the main Figure 5 analyses
participant_summary = correction_df.groupby(
    ['Participant ID', 'Condition', 'Group'],
    as_index=False
).agg({
    'Initial Error': 'mean',
    'Final Error': 'mean',
    'Estimation Error Change': 'mean'
})

print("\n" + "="*80)
print("Figure 5: Estimation-Error Data Prepared")
print("="*80)
print("Statement-level correction_df")
print(correction_df.head().to_string(index=False))

print("\nParticipant-level participant_summary")
print(participant_summary.head().to_string(index=False))

# =============================================================================
# purpose: save participant-level plotting data for Figure 5
# output:
#   - figure5.csv with initial and final estimation error by condition
# =============================================================================
figure_5_df = participant_summary[
    ['Participant ID', 'Condition', 'Initial Error', 'Final Error', 'Estimation Error Change']
].copy()

figure_5_df.to_csv('../output/figure5.csv', index=False)

print("\n" + "="*80)
print("Figure 5 Output: Participant-Level Initial and Final Estimation Error")
print("="*80)
print(figure_5_df.head().to_string(index=False))

# =============================================================================
# purpose: compare control versus pooled treatment on participant-level
#          change in estimation error, as specified in the main Methods
# measure:
#   - Estimation Error Change = Final Error - Initial Error
# test:
#   - Welch t-test; the contrast is control minus pooled treatment
# note:
#   - this adds printed output only and does not change the Figure 5 data
# =============================================================================
print("\n" + "="*80)
print("Figure 5: Control vs. Pooled Treatment on Estimation Error Change")
print("="*80)

pooled_control_change = participant_summary[
    participant_summary['Group'] == 'control'
]['Estimation Error Change']
pooled_treatment_change = participant_summary[
    participant_summary['Group'] == 'treatment'
]['Estimation Error Change']

pooled_error_test = stats.ttest_ind(
    pooled_control_change, pooled_treatment_change, equal_var=False
)
print(f"Control: Mean = {pooled_control_change.mean():.2f}, N = {len(pooled_control_change)}")
print(f"Pooled treatment: Mean = {pooled_treatment_change.mean():.2f}, N = {len(pooled_treatment_change)}")
print(f"Mean difference (control - treatment): {pooled_control_change.mean() - pooled_treatment_change.mean():.2f}")
print(f"Welch t-statistic: {pooled_error_test.statistic:.2f}, P-value: {pooled_error_test.pvalue:.3e}")

# =============================================================================
# purpose: test whether estimation error changes from initial
#          to final within each condition for Figure 5
# tests:
#   - paired t-test within each condition
#   - Wilcoxon signed-rank test as a non-parametric robustness check
# effect size:
#   - Cohen's d for paired differences
# =============================================================================
print("\n" + "="*80)
print("Figure 5: Within-Condition Change in Estimation Error")
print("="*80)

for condition in participant_summary['Condition'].unique():
    condition_df = participant_summary[participant_summary['Condition'] == condition].copy()

    initial_error_stats = mean_std_ci(condition_df['Initial Error'])
    final_error_stats = mean_std_ci(condition_df['Final Error'])
    change_stats = mean_std_ci(condition_df['Estimation Error Change'])

    # paired t-test
    t_stat, p_value = stats.ttest_rel(
        condition_df['Initial Error'],
        condition_df['Final Error']
    )

    # Wilcoxon signed-rank test
    w_stat, p_value_w = stats.wilcoxon(
        condition_df['Initial Error'],
        condition_df['Final Error']
    )

    # Cohen's d for paired differences
    diff = condition_df['Final Error'] - condition_df['Initial Error']
    cohen_d_paired = diff.mean() / diff.std()

    print(f"\nCondition: {condition}")
    print(
        'Initial Error: {:.2f}, SD={:.2f}, 95% CI=({:.2f}, {:.2f})'.format(
            initial_error_stats[0], initial_error_stats[1],
            initial_error_stats[2][0], initial_error_stats[2][1]
        )
    )
    print(
        'Final Error: {:.2f}, SD={:.2f}, 95% CI=({:.2f}, {:.2f})'.format(
            final_error_stats[0], final_error_stats[1],
            final_error_stats[2][0], final_error_stats[2][1]
        )
    )
    print(
        'Estimation Error Change: {:.2f}, SD={:.2f}, 95% CI=({:.2f}, {:.2f})'.format(
            change_stats[0], change_stats[1],
            change_stats[2][0], change_stats[2][1]
        )
    )
    print(f"Paired t-statistic: {t_stat:.2f}, P-value: {p_value:.4f}")
    print(f"Wilcoxon statistic: {w_stat:.2f}, P-value: {p_value_w:.4f}")
    print(f"Cohen's d: {cohen_d_paired:.2f}")

# =============================================================================
# purpose: compare estimation error change across all
#          conditions for Figure 5
# tests:
#   - one-way ANOVA across all conditions
#   - Kruskal-Wallis test as a non-parametric robustness check
#   - Tukey HSD post-hoc comparisons if the ANOVA is significant
# =============================================================================
print("\n" + "="*80)
print("Figure 5: Between-Condition Comparison of Estimation Error Change")
print("="*80)

# condition-level participant summaries
for condition in participant_summary['Condition'].unique():
    condition_df = participant_summary[participant_summary['Condition'] == condition]
    change_stats = mean_std_ci(condition_df['Estimation Error Change'])

    print(
        f"Condition: {condition} - Estimation Error Change: {change_stats[0]:.2f}, "
        f"95% CI: ({change_stats[2][0]:.2f}, {change_stats[2][1]:.2f}), "
        f"N = {condition_df['Participant ID'].nunique()}"
    )

# omnibus participant-level tests across all conditions
error_change_groups = [
    participant_summary[participant_summary['Condition'] == condition]['Estimation Error Change']
    for condition in participant_summary['Condition'].unique()
]

f_stat, p_value = stats.f_oneway(*error_change_groups)
print(f"\nOne-way ANOVA: F = {f_stat:.2f}, P-value = {p_value:.4f}")

h_stat, p_value_kw = stats.kruskal(*error_change_groups)
print(f"Kruskal-Wallis H-statistic: {h_stat:.2f}, P-value = {p_value_kw:.4f}")

# post-hoc Tukey HSD if the ANOVA is significant
from statsmodels.stats.multicomp import pairwise_tukeyhsd

if p_value < 0.05:
    tukey = pairwise_tukeyhsd(
        endog=participant_summary['Estimation Error Change'],
        groups=participant_summary['Condition'],
        alpha=0.05
    )
    print("\nTukey HSD Post-hoc Comparisons (all five conditions):")
    print(tukey)

# =============================================================================
# purpose: compare estimation error change across the four
#          treatment conditions and then compare each treatment with control
# tests:
#   - one-way ANOVA across treatment conditions only
#   - Tukey HSD post-hoc comparisons across treatment conditions
#   - independent-samples t-tests comparing each treatment with control
# effect size:
#   - Cohen's d for each treatment-versus-control comparison
# =============================================================================
print("\n" + "="*80)
print("Figure 5: Treatment-Focused Follow-Up Comparisons")
print("="*80)

# treatment-only omnibus comparison
treatment_data = participant_summary[
    participant_summary['Condition'] != 'Individual'
].copy()

treatment_conditions = treatment_data['Condition'].unique()
treatment_groups = [
    treatment_data[treatment_data['Condition'] == condition]['Estimation Error Change']
    for condition in treatment_conditions
]

f_stat_treat, p_value_treat = stats.f_oneway(*treatment_groups)
print(f"\nOne-way ANOVA (treatments only): F = {f_stat_treat:.2f}, P-value = {p_value_treat:.4f}")

# Tukey HSD across treatment conditions
from statsmodels.stats.multicomp import pairwise_tukeyhsd

if p_value_treat < 0.05:
    tukey_treat = pairwise_tukeyhsd(
        endog=treatment_data['Estimation Error Change'],
        groups=treatment_data['Condition'],
        alpha=0.05
    )
    print("\nTukey HSD (treatments only; main-text accuracy comparisons):")
    print(tukey_treat)

print("\nTreatment vs Control Comparisons:")

control_change = participant_summary[
    participant_summary['Condition'] == 'Individual'
]['Estimation Error Change']

for condition in ['Season', 'Conformity', 'Age', 'Prestige']:
    treatment_change = participant_summary[
        participant_summary['Condition'] == condition
    ]['Estimation Error Change']

    # independent-samples t-test
    t_stat, p_value = stats.ttest_ind(treatment_change, control_change, equal_var=False)

    # Cohen's d
    pooled_std = np.sqrt(
        (control_change.std() ** 2 + treatment_change.std() ** 2) / 2
    )
    cohen_d_between = (
        (treatment_change.mean() - control_change.mean()) / pooled_std
    )

    print(f"\n{condition} vs Control:")
    print(f"Mean difference: {treatment_change.mean() - control_change.mean():.2f}")
    print(f"Welch t-statistic: {t_stat:.2f}, P-value: {p_value:.4f}")
    print(f"Cohen's d: {cohen_d_between:.2f}")

################################################################################
######################## Additional Analysis Data Exports #######################
###################################################################################
# =============================================================================
# purpose: regenerate the legacy analysis csv files in original formats
#          while keeping all measurements consistent with the current
#          main_analysis.py workflow
# files:
#   - ../data/complete-anova.csv
#   - ../data/randomization-check.csv
#   - ../data/robustness-check.csv
# note:
#   - robustness-data.csv is not regenerated because it is identical to
#     robustness-check.csv
#   - original column names are preserved
#   - robustness-check.csv retains all original columns and appends three
#     new signed misperception measures from the recoded gender-inequity space
# =============================================================================

demographics_export_df = pd.read_csv('../data/demographics.csv')
statements_export_df = pd.read_csv('../data/statements.csv')

############################################################################
########################## complete-anova.csv ###############################
############################################################################
# =============================================================================
# purpose: regenerate complete-anova.csv in its original format
# measure source:
#   - uses the current Figure 5 raw agreement-space estimation-error logic
# columns:
#   - Participant ID
#   - Condition
#   - Error Reduction
#   - MAR
#   - Final Error
# note:
#   - the legacy column name "Error Reduction" is preserved, but its values
#     follow the current main_analysis.py definition:
#         Final Error - Initial Error
#   - negative values indicate improved accuracy
# =============================================================================

complete_anova_df = correction_df[
    ['Participant ID', 'Condition', 'Estimation Error Change', 'MAR', 'Final Error']
].copy()

complete_anova_df = complete_anova_df.rename(columns={
    'Estimation Error Change': 'Error Reduction'
})

complete_anova_df = complete_anova_df[
    ['Participant ID', 'Condition', 'Error Reduction', 'MAR', 'Final Error']
].copy()

complete_anova_df.to_csv('../data/complete-anova.csv', index=False)

print("\n" + "=" * 80)
print("complete-anova.csv regenerated")
print("=" * 80)
print(complete_anova_df.head().to_string(index=False))


############################################################################
######################## randomization-check.csv ############################
############################################################################
# =============================================================================
# purpose: regenerate randomization-check.csv in its original repeated format
# structure:
#   - one row per participant-statement observation, with participant-level
#     demographics repeated across statements
# columns:
#   - Participant ID
#   - Condition
#   - Gender
#   - Age
#   - Education
#   - Marital Status
#   - Race
#   - Political Affiliation
# =============================================================================

randomization_check_df = estimates_df[
    ['Participant ID', 'Condition']
].copy()

randomization_check_df = randomization_check_df.merge(
    demographics_export_df[
        ['Participant ID', 'Gender', 'Age', 'Education',
         'Marital Status', 'Race', 'Political Affiliation']
    ],
    on='Participant ID',
    how='left'
)

randomization_check_df = randomization_check_df[
    ['Participant ID', 'Condition', 'Gender', 'Age', 'Education',
     'Marital Status', 'Race', 'Political Affiliation']
].copy()

randomization_check_df.to_csv('../data/randomization-check.csv', index=False)

print("\n" + "=" * 80)
print("randomization-check.csv regenerated")
print("=" * 80)
print(randomization_check_df.head().to_string(index=False))


###########################################################################
######################### robustness-check.csv #############################
###########################################################################
# =============================================================================
# purpose: regenerate robustness-check.csv in its original format and append
#          three new signed misperception measures
# original columns preserved:
#   - Participant ID
#   - Condition
#   - Statement ID
#   - Initial Estimate
#   - Final Estimate
#   - MAR
#   - Agreement
#   - Reference
#   - Initial Error
#   - Final Error
#   - Error Reduction
#   - Influence Score
#   - Gender
#   - Age
#   - Political Affiliation
# new columns appended:
#   - Initial Misperception
#   - Final Misperception
#   - Misperception Change
# measure source:
#   - raw-space variables follow the current Figure 4 / Figure 5 logic
#   - signed misperception variables follow the current recoded
#     gender-inequity-space logic
# =============================================================================

# start from the current Figure 5 raw agreement-space dataframe
robustness_check_df = correction_df[
    ['Participant ID', 'Condition', 'Statement ID',
     'Initial Estimate', 'Final Estimate', 'MAR',
     'Agreement', 'Initial Error', 'Final Error', 'Estimation Error Change']
].copy()

robustness_check_df = robustness_check_df.rename(columns={
    'Estimation Error Change': 'Error Reduction'
})

# merge displayed reference values
robustness_check_df = robustness_check_df.merge(
    statements_export_df[['Statement ID', 'Reference']],
    on='Statement ID',
    how='left'
)

robustness_check_df['Reference'] = pd.to_numeric(
    robustness_check_df['Reference'],
    errors='raise'
)

# influence score follows the current Figure 4 raw agreement-space logic
social_mask = robustness_check_df['Condition'] != 'Individual'
denom = robustness_check_df['Reference'] - robustness_check_df['Initial Estimate']

robustness_check_df['Influence Score'] = np.nan
robustness_check_df.loc[social_mask, 'Influence Score'] = (
    (robustness_check_df.loc[social_mask, 'Final Estimate'] -
     robustness_check_df.loc[social_mask, 'Initial Estimate']) /
    denom.loc[social_mask]
)

zero_gap_mask = social_mask & (denom == 0)
robustness_check_df.loc[zero_gap_mask, 'Influence Score'] = np.nan

# append selected demographics
robustness_check_df = robustness_check_df.merge(
    demographics_export_df[
        ['Participant ID', 'Gender', 'Age', 'Political Affiliation']
    ],
    on='Participant ID',
    how='left'
)

# append the new signed misperception measures from the current recoded
# gender-inequity-space workflow
misperception_append_df = misperception_df[
    ['Participant ID', 'Condition', 'Statement ID', 'Initial Estimate', 'Final Estimate', 'Agreement']
].copy()

misperception_append_df['Initial Misperception'] = (
    misperception_append_df['Initial Estimate'] - misperception_append_df['Agreement']
)
misperception_append_df['Final Misperception'] = (
    misperception_append_df['Final Estimate'] - misperception_append_df['Agreement']
)
misperception_append_df['Misperception Change'] = (
    misperception_append_df['Final Misperception'] -
    misperception_append_df['Initial Misperception']
)

misperception_append_df = misperception_append_df[
    ['Participant ID', 'Condition', 'Statement ID',
     'Initial Misperception', 'Final Misperception', 'Misperception Change']
].copy()

robustness_check_df = robustness_check_df.merge(
    misperception_append_df,
    on=['Participant ID', 'Condition', 'Statement ID'],
    how='left'
)

robustness_check_df = robustness_check_df[
    ['Participant ID', 'Condition', 'Statement ID',
     'Initial Estimate', 'Final Estimate', 'MAR',
     'Agreement', 'Reference', 'Initial Error', 'Final Error',
     'Error Reduction', 'Influence Score',
     'Gender', 'Age', 'Political Affiliation',
     'Initial Misperception', 'Final Misperception', 'Misperception Change']
].copy()

robustness_check_df.to_csv('../data/robustness-check.csv', index=False)

print("\n" + "=" * 80)
print("robustness-check.csv regenerated")
print("=" * 80)
print(robustness_check_df.head().to_string(index=False))


# =============================================================================
# purpose: document that robustness-data.csv is not regenerated because it is
#          identical to robustness-check.csv in the original workflow
# =============================================================================
print("\n" + "=" * 80)
print("robustness-data.csv not regenerated")
print("=" * 80)
print("robustness-data.csv was not regenerated because it duplicates robustness-check.csv.")