import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from patsy import build_design_matrices
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib.table import Table


# =============================================================================
# purpose: load cleaned participant-statement estimates, participant-level
#          demographics, and statement-level reference values
# inputs:
#   - ../data/estimates.csv
#   - ../data/demographics.csv
#   - ../data/statements.csv
# output:
#   - ../output/figure2.csv
# =============================================================================

estimates_df = pd.read_csv('../data/estimates.csv')
demographics_df = pd.read_csv('../data/demographics.csv')
statements_df = pd.read_csv('../data/statements.csv')


# =============================================================================
# purpose: define helpers for agreement parsing, significance coding, p-value
#          formatting, and non-parametric robustness checks
# =============================================================================

def _stars(p):
    if p is None or pd.isna(p):
        return ""
    p = float(p)
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""

def _format_p(p):
    if p is None or pd.isna(p):
        return ""
    p = float(p)
    if p < 0.001:
        return "<0.001"
    return f"{p:.3f}"

def _agreement_response_to_binary(v):
    """
    Map raw agreement responses to binary agreement:
      - 1 if Agree / Strongly agree
      - 0 otherwise
    Supports either text responses or numeric 1..5 coding.
    """
    if pd.isna(v):
        return np.nan

    s = str(v).strip().lower()

    # numeric fallback
    try:
        x = float(s)
        if x in [1, 2, 3]:
            return 0.0
        if x in [4, 5]:
            return 1.0
    except Exception:
        pass

    mapping = {
        "strongly disagree": 0.0,
        "disagree": 0.0,
        "neither agree nor disagree": 0.0,
        "neither": 0.0,
        "neutral": 0.0,
        "agree": 1.0,
        "strongly agree": 1.0,
    }
    return mapping.get(s, np.nan)

def _kw_test(frame, group_col, outcome_col):
    sub = frame[[group_col, outcome_col]].copy()
    sub[outcome_col] = pd.to_numeric(sub[outcome_col], errors='coerce')
    sub = sub.dropna(subset=[group_col, outcome_col])

    groups = []
    labels = []

    for label, g in sub.groupby(group_col, observed=False):
        vals = g[outcome_col].to_numpy(dtype=float)
        if len(vals) > 0:
            groups.append(vals)
            labels.append(label)

    if len(groups) < 2:
        return np.nan, np.nan, len(groups), labels

    h_stat, p_value = stats.kruskal(*groups)
    return float(h_stat), float(p_value), len(groups), labels


# =============================================================================
# purpose: compute statement-level actual support as the observed proportion
#          agreeing with each statement as worded
# measure:
#   - Actual agreement (%) = percent of participants who agree with the
#     statement as worded
# =============================================================================

if 'Agreement-Text' in estimates_df.columns:
    agreement_source_col = 'Agreement-Text'
elif 'Agreement' in estimates_df.columns:
    agreement_source_col = 'Agreement'
else:
    raise ValueError(
        "estimates.csv must contain either 'Agreement-Text' or 'Agreement' "
        "to compute statement-level actual agreement."
    )

actual_agreement_df = (
    estimates_df.groupby('Statement ID', as_index=False)
    .agg(
        **{
            'Actual Agreement': (
                agreement_source_col,
                lambda x: pd.Series(x).apply(_agreement_response_to_binary).mean() * 100.0
            )
        }
    )
)


# =============================================================================
# purpose: recode baseline estimates and actual norms to a common
#          gender-inequity scale, then compute participant-level baseline
#          overestimation / misperception
# measure:
#   - Misperception = perceived support for gender inequity minus actual support
#     for gender inequity, in percentage points
# coding:
#   - statements 1-6 are equitable as worded and are reversed
#   - statements 7-12 are inequitable as worded and are left unchanged
# =============================================================================

equitable_ids = [1, 2, 3, 4, 5, 6]

baseline_df = estimates_df[
    ['Participant ID', 'Statement ID', 'Initial Estimate']
].copy()

baseline_df['Initial Estimate'] = pd.to_numeric(
    baseline_df['Initial Estimate'],
    errors='raise'
)

baseline_df = baseline_df.merge(
    actual_agreement_df,
    on='Statement ID',
    how='left'
)

mask_equitable = baseline_df['Statement ID'].isin(equitable_ids)

baseline_df['Initial Inequity'] = baseline_df['Initial Estimate']
baseline_df['Actual Inequity'] = baseline_df['Actual Agreement']

baseline_df.loc[mask_equitable, 'Initial Inequity'] = (
    100.0 - baseline_df.loc[mask_equitable, 'Initial Inequity']
)
baseline_df.loc[mask_equitable, 'Actual Inequity'] = (
    100.0 - baseline_df.loc[mask_equitable, 'Actual Inequity']
)

baseline_df['Misperception'] = (
    baseline_df['Initial Inequity'] - baseline_df['Actual Inequity']
)

participant_over_df = (
    baseline_df.groupby('Participant ID', as_index=False)['Misperception']
    .mean()
    .rename(columns={'Misperception': 'overestimation_score_pp'})
)


# =============================================================================
# purpose: merge participant-level baseline overestimation with demographics
#          and derive the demographic predictors used in the baseline model
# predictors:
#   - Gender
#   - age_group (18-29, 30-44, 45-64)
#   - edu3
# note:
#   - education grouping:
#       low    = Less than high school + High school diploma/equivalent
#                + Some college (no degree)
#       middle = Associate degree + Bachelor's degree
#       high   = Graduate / professional degree
# =============================================================================

reg_df = participant_over_df.merge(
    demographics_df,
    on='Participant ID',
    how='inner'
)

reg_df['Age'] = pd.to_numeric(reg_df['Age'], errors='coerce')
reg_df['age_group'] = pd.cut(
    reg_df['Age'],
    bins=[17, 29, 44, 64],
    labels=['18–29 years', '30–44 years', '45–64 years']
)

def _collapse_edu3(x):
    s = str(x).strip()
    if s == "" or s.lower() == "nan":
        return np.nan
    if s in {
        'Less than high school',
        'High school diploma or equivalent',
        'Some college (no degree)'
    }:
        return 'low'
    if s in {
        'Associate degree',
        "Bachelor's degree"
    }:
        return 'middle'
    if s == 'Graduate or professional degree (Masters, PhD, JD, MD, etc.)':
        return 'high'
    return np.nan

reg_df['edu3'] = reg_df['Education'].apply(_collapse_edu3)
reg_df['edu3'] = pd.Categorical(
    reg_df['edu3'],
    categories=['low', 'middle', 'high'],
    ordered=True
)


# =============================================================================
# purpose: check validity of the three demographic predictors used in the model
#          and restrict the regression sample to participants with valid values
# valid responses:
#   - Gender: Male or Female only
#   - age_group: must map into 18–29, 30–44, or 45–64
#   - edu3: must map into low, middle, or high
# output:
#   - printed counts of invalid responses for each demographic predictor
#   - printed raw invalid response values for inspection
#   - filtered reg_df containing only valid analytic cases
# =============================================================================

valid_gender = ['Male', 'Female']

invalid_gender_mask = ~reg_df['Gender'].isin(valid_gender)
invalid_age_mask = reg_df['age_group'].isna()
invalid_edu_mask = reg_df['edu3'].isna()

print("\n" + "=" * 80)
print("Demographic Validity Checks Before Baseline Regression")
print("=" * 80)

# Gender
invalid_gender_values = (
    reg_df.loc[invalid_gender_mask, 'Gender']
    .astype('string')
    .fillna('<missing>')
    .value_counts(dropna=False)
)

print(f"\nInvalid Gender responses: {int(invalid_gender_mask.sum())}")
if len(invalid_gender_values) > 0:
    print("Raw invalid Gender values:")
    print(invalid_gender_values.to_string())
else:
    print("Raw invalid Gender values: none")

# Age
invalid_age_values = (
    reg_df.loc[invalid_age_mask, 'Age']
    .astype('string')
    .fillna('<missing>')
    .value_counts(dropna=False)
)

print(f"\nInvalid Age responses for model grouping: {int(invalid_age_mask.sum())}")
if len(invalid_age_values) > 0:
    print("Raw invalid Age values:")
    print(invalid_age_values.to_string())
else:
    print("Raw invalid Age values: none")

# Education
invalid_education_values = (
    reg_df.loc[invalid_edu_mask, 'Education']
    .astype('string')
    .fillna('<missing>')
    .value_counts(dropna=False)
)

print(f"\nInvalid Education responses for model grouping: {int(invalid_edu_mask.sum())}")
if len(invalid_education_values) > 0:
    print("Raw invalid Education values:")
    print(invalid_education_values.to_string())
else:
    print("Raw invalid Education values: none")

# restrict to valid analytic cases
valid_model_mask = (
    reg_df['Gender'].isin(valid_gender) &
    reg_df['age_group'].notna() &
    reg_df['edu3'].notna()
)

print(f"\nParticipants retained for regression: {int(valid_model_mask.sum())} / {len(reg_df)}")
print(f"Participants excluded from regression: {int((~valid_model_mask).sum())}")

reg_df = reg_df.loc[valid_model_mask].copy()


# =============================================================================
# purpose: fit the participant-level baseline demographic regression
# outcome:
#   - overestimation_score_pp
# model:
#   - OLS with HC3 robust standard errors
# predictors:
#   - Gender
#   - age_group
#   - edu3
# =============================================================================

formula = (
    "overestimation_score_pp ~ "
    "C(Gender, Treatment(reference='Male')) + "
    "C(age_group, Treatment(reference='18–29 years')) + "
    "C(edu3, Treatment(reference='low'))"
)

model = smf.ols(formula, data=reg_df).fit()
robust_model = model.get_robustcov_results(cov_type='HC3')

print("\n" + "=" * 80)
print("Figure 2: Baseline Demographic Regression on Initial Overestimation")
print("=" * 80)
print(f"N used in model: {int(model.nobs)}")
print(f"Adjusted R-squared: {model.rsquared_adj:.4f}")

coef_names = list(robust_model.model.exog_names)
coef_tcrit = float(stats.t.ppf(0.975, df=robust_model.df_resid))

coef_df = pd.DataFrame({
    'term': coef_names,
    'coef': robust_model.params,
    'se_hc3': robust_model.bse,
    't_hc3': robust_model.tvalues,
    'p_hc3': robust_model.pvalues
})

coef_df['ci_lower'] = coef_df['coef'] - coef_tcrit * coef_df['se_hc3']
coef_df['ci_upper'] = coef_df['coef'] + coef_tcrit * coef_df['se_hc3']
coef_df['stars'] = coef_df['p_hc3'].apply(_stars)

print("\nRegression coefficients (HC3 robust SEs):")
print(coef_df.to_string(index=False))


# =============================================================================
# purpose: compute HC3-based adjusted predicted means and confidence intervals
#          for the Figure 2 demographic figure input
# figure structure:
#   - Gender panel: Male, Female
#   - Age panel: 18-29, 30-44, 45-64
#   - Education panel: Low, Middle, High
# output:
#   - ../output/figure2.csv
# =============================================================================

design_info = model.model.data.design_info
cov = robust_model.cov_params()
params = np.asarray(robust_model.params, dtype=float)
df_resid = float(robust_model.df_resid)
tcrit = float(stats.t.ppf(0.975, df=df_resid))

def _term_pvalue(term):
    names = list(robust_model.model.exog_names)
    if term == 'Intercept' and term not in names and 'const' in names:
        term = 'const'
    if term not in names:
        return np.nan
    idx = names.index(term)
    return float(robust_model.pvalues[idx])

def _marginal_mean_ci(df_base, factor, level):
    new = df_base.copy()
    new[factor] = level
    Xp = build_design_matrices([design_info], new, return_type='dataframe')[0]
    xbar = Xp.mean(axis=0).to_numpy(dtype=float)
    mean = float(xbar @ params)
    se = float(np.sqrt(xbar @ cov @ xbar))
    lower = mean - tcrit * se
    upper = mean + tcrit * se
    return mean, lower, upper

panels = [
    {
        'panel': 'Gender',
        'factor': 'Gender',
        'levels': ['Male', 'Female'],
        'reference': 'Male',
        'term': lambda lvl: f"C(Gender, Treatment(reference='Male'))[T.{lvl}]",
        'display_map': {}
    },
    {
        'panel': 'Age group',
        'factor': 'age_group',
        'levels': ['18–29 years', '30–44 years', '45–64 years'],
        'reference': '18–29 years',
        'term': lambda lvl: f"C(age_group, Treatment(reference='18–29 years'))[T.{lvl}]",
        'display_map': {}
    },
    {
        'panel': 'Education',
        'factor': 'edu3',
        'levels': ['low', 'middle', 'high'],
        'reference': 'low',
        'term': lambda lvl: f"C(edu3, Treatment(reference='low'))[T.{lvl}]",
        'display_map': {'low': 'Low', 'middle': 'Middle', 'high': 'High'}
    }
]

figure_rows = []

for panel_index, panel in enumerate(panels, start=1):
    factor = panel['factor']
    levels = panel['levels']
    reference = panel['reference']

    for level_index, level in enumerate(levels, start=1):
        mean, lower, upper = _marginal_mean_ci(reg_df, factor, level)
        n_level = int(reg_df[factor].astype(str).eq(str(level)).sum())

        if level == reference:
            p_value = np.nan
            stars = ""
        else:
            p_value = _term_pvalue(panel['term'](level))
            stars = _stars(p_value)

        figure_rows.append({
            'panel': panel['panel'],
            'panel_order': panel_index,
            'factor': factor,
            'level': level,
            'level_display': panel['display_map'].get(level, level),
            'level_order': level_index,
            'reference_group': reference,
            'n': n_level,
            'adjusted_mean': mean,
            'ci_lower': lower,
            'ci_upper': upper,
            'p_value_vs_reference': p_value,
            'stars': stars
        })

figure_df = pd.DataFrame(figure_rows)
figure_df.to_csv('../output/figure2.csv', index=False)

print("\n" + "=" * 80)
print("Figure 2 input saved to ../output/figure2.csv")
print("=" * 80)
print(figure_df.to_string(index=False))


# =============================================================================
# purpose: print a formatted regression table for the current baseline model
#          with coefficient, confidence interval, and exact p-value
# note:
#   - this does not replace any existing output
#   - this is an additional printed table only
# =============================================================================

def _get_formatted_coef_row(term):
    row = coef_df.loc[coef_df['term'] == term]
    if row.empty:
        return "", "", ""
    b_val = float(row['coef'].iloc[0])
    ci_low = float(row['ci_lower'].iloc[0])
    ci_high = float(row['ci_upper'].iloc[0])
    p_val = float(row['p_hc3'].iloc[0])
    stars = str(row['stars'].iloc[0])
    return f"{b_val:.2f}{stars}", f"{ci_low:.2f}, {ci_high:.2f}", _format_p(p_val)

intercept_b, intercept_ci, intercept_p = _get_formatted_coef_row('Intercept')
female_b, female_ci, female_p = _get_formatted_coef_row(
    "C(Gender, Treatment(reference='Male'))[T.Female]"
)
age_30_44_b, age_30_44_ci, age_30_44_p = _get_formatted_coef_row(
    "C(age_group, Treatment(reference='18–29 years'))[T.30–44 years]"
)
age_45_64_b, age_45_64_ci, age_45_64_p = _get_formatted_coef_row(
    "C(age_group, Treatment(reference='18–29 years'))[T.45–64 years]"
)
edu_middle_b, edu_middle_ci, edu_middle_p = _get_formatted_coef_row(
    "C(edu3, Treatment(reference='low'))[T.middle]"
)
edu_high_b, edu_high_ci, edu_high_p = _get_formatted_coef_row(
    "C(edu3, Treatment(reference='low'))[T.high]"
)

regression_table_df = pd.DataFrame([
    {
        'Predictor': 'Intercept',
        'Level': '',
        'B': intercept_b,
        '95% CI': intercept_ci,
        'p-value': intercept_p
    },
    {
        'Predictor': 'Age group (reference, 18–29 years)',
        'Level': '30–44 years',
        'B': age_30_44_b,
        '95% CI': age_30_44_ci,
        'p-value': age_30_44_p
    },
    {
        'Predictor': '',
        'Level': '45–64 years',
        'B': age_45_64_b,
        '95% CI': age_45_64_ci,
        'p-value': age_45_64_p
    },
    {
        'Predictor': 'Education (reference, low)',
        'Level': 'Middle',
        'B': edu_middle_b,
        '95% CI': edu_middle_ci,
        'p-value': edu_middle_p
    },
    {
        'Predictor': '',
        'Level': 'High',
        'B': edu_high_b,
        '95% CI': edu_high_ci,
        'p-value': edu_high_p
    },
    {
        'Predictor': 'Gender (reference, Male)',
        'Level': 'Female',
        'B': female_b,
        '95% CI': female_ci,
        'p-value': female_p
    },
    {
        'Predictor': 'N',
        'Level': '',
        'B': f"{int(model.nobs)}",
        '95% CI': '',
        'p-value': ''
    },
    {
        'Predictor': 'Adjusted R-squared',
        'Level': '',
        'B': f"{model.rsquared_adj:.4f}",
        '95% CI': '',
        'p-value': ''
    }
])

print("\n" + "=" * 80)
print("Formatted Regression Table")
print("=" * 80)
print(regression_table_df.to_string(index=False))


# =============================================================================
# purpose: run non-parametric robustness checks for the baseline outcome across
#          the demographic predictors used in the regression
# tests:
#   - Kruskal-Wallis by Gender
#   - Kruskal-Wallis by age_group
#   - Kruskal-Wallis by edu3
# =============================================================================

kw_rows = []

for group_col, label in [
    ('Gender', 'Gender'),
    ('age_group', 'Age group'),
    ('edu3', 'Education')
]:
    h_stat, p_value, n_groups, group_labels = _kw_test(
        reg_df,
        group_col=group_col,
        outcome_col='overestimation_score_pp'
    )

    kw_rows.append({
        'predictor': label,
        'group_column': group_col,
        'n_groups': n_groups,
        'h_statistic': h_stat,
        'p_value': p_value
    })

kw_df = pd.DataFrame(kw_rows)

print("\n" + "=" * 80)
print("Kruskal-Wallis Robustness Checks")
print("=" * 80)
print(kw_df.to_string(index=False))


# =============================================================================
# purpose: prepare participant-level post-treatment outcomes and additional
#          demographic predictors for a broader formatted regression table
# note:
#   - this section does not change the baseline Figure 2 workflow
#   - it only creates an additional printed table and a PDF export
#   - condition is retained in post-treatment models to stay aligned with the
#     experimental design of the study
# =============================================================================

analysis_df = estimates_df[
    ['Participant ID', 'Condition', 'Statement ID', 'Initial Estimate', 'Final Estimate']
].copy()

analysis_df['Initial Estimate'] = pd.to_numeric(analysis_df['Initial Estimate'], errors='raise')
analysis_df['Final Estimate'] = pd.to_numeric(analysis_df['Final Estimate'], errors='raise')

analysis_df = analysis_df.merge(
    actual_agreement_df.rename(columns={'Actual Agreement': 'Agreement'}),
    on='Statement ID',
    how='left'
)

analysis_df = analysis_df.merge(
    statements_df[['Statement ID', 'Reference']],
    on='Statement ID',
    how='left'
)

analysis_df['Agreement'] = pd.to_numeric(analysis_df['Agreement'], errors='raise')
analysis_df['Reference'] = pd.to_numeric(analysis_df['Reference'], errors='raise')

analysis_df['Initial Inequity'] = analysis_df['Initial Estimate']
analysis_df['Final Inequity'] = analysis_df['Final Estimate']
analysis_df['Actual Inequity'] = analysis_df['Agreement']

mask_equitable = analysis_df['Statement ID'].isin(equitable_ids)

analysis_df.loc[mask_equitable, 'Initial Inequity'] = (
    100.0 - analysis_df.loc[mask_equitable, 'Initial Inequity']
)
analysis_df.loc[mask_equitable, 'Final Inequity'] = (
    100.0 - analysis_df.loc[mask_equitable, 'Final Inequity']
)
analysis_df.loc[mask_equitable, 'Actual Inequity'] = (
    100.0 - analysis_df.loc[mask_equitable, 'Actual Inequity']
)

analysis_df['Initial Misperception'] = (
    analysis_df['Initial Inequity'] - analysis_df['Actual Inequity']
)
analysis_df['Final Misperception'] = (
    analysis_df['Final Inequity'] - analysis_df['Actual Inequity']
)

analysis_df['MAR'] = np.abs(
    analysis_df['Final Inequity'] - analysis_df['Initial Inequity']
)
analysis_df['Revised?'] = (analysis_df['MAR'] > 0).astype(int)

analysis_df['Influence Score Item'] = np.nan
social_mask = analysis_df['Condition'] != 'Individual'
denom = analysis_df['Reference'] - analysis_df['Initial Estimate']

analysis_df.loc[social_mask, 'Influence Score Item'] = (
    (analysis_df.loc[social_mask, 'Final Estimate'] -
     analysis_df.loc[social_mask, 'Initial Estimate']) /
    denom.loc[social_mask]
)

zero_gap_mask = social_mask & (denom == 0)
analysis_df.loc[zero_gap_mask, 'Influence Score Item'] = np.nan

participant_post_df = analysis_df.groupby(
    ['Participant ID', 'Condition'],
    as_index=False
).agg({
    'Final Misperception': 'mean',
    'MAR': 'mean',
    'Revised?': 'mean',
    'Influence Score Item': 'mean'
}).rename(columns={
    'Final Misperception': 'overestimation_after_pp',
    'MAR': 'absolute_revision_pp',
    'Revised?': 'revision_rate_prop',
    'Influence Score Item': 'influence_score'
})

participant_post_df['revision_rate'] = participant_post_df['revision_rate_prop'] * 100.0

broad_df = participant_over_df.merge(
    participant_post_df,
    on='Participant ID',
    how='inner'
).merge(
    demographics_df,
    on='Participant ID',
    how='inner'
)

broad_df['Age'] = pd.to_numeric(broad_df['Age'], errors='coerce')
broad_df['age_group'] = pd.cut(
    broad_df['Age'],
    bins=[17, 29, 44, 64],
    labels=['18–29 years', '30–44 years', '45–64 years']
)

broad_df['edu3'] = broad_df['Education'].apply(_collapse_edu3)
broad_df['edu3'] = pd.Categorical(
    broad_df['edu3'],
    categories=['low', 'middle', 'high'],
    ordered=True
)

def _collapse_marital2(x):
    s = str(x).strip()
    if s == "" or s.lower() == "nan":
        return np.nan
    if s == 'Widowed':
        return 'Divorced/separated/widowed'
    if s == 'Divorced or separated':
        return 'Divorced/separated/widowed'
    if s in {
        'Single, never married',
        'Married',
        'In relationship, but not married'
    }:
        return s
    return np.nan

broad_df['marital2'] = broad_df['Marital Status'].apply(_collapse_marital2)
broad_df['marital2'] = pd.Categorical(
    broad_df['marital2'],
    categories=[
        'Single, never married',
        'Married',
        'In relationship, but not married',
        'Divorced/separated/widowed'
    ],
    ordered=False
)

def _collapse_race_group(x):
    s = str(x).strip()
    if s == "" or s.lower() == "nan":
        return np.nan

    parts = [p.strip() for p in s.split(',') if p.strip()]
    if len(parts) == 0:
        return np.nan
    if 'Prefer not to say' in parts:
        return np.nan

    core_levels = {
        'White',
        'Black or African American',
        'Hispanic or Latino/a/x',
        'Asian'
    }

    if len(parts) >= 2:
        return 'Multiple'

    single = parts[0]
    if single in core_levels:
        return single
    return 'Other (pooled)'

broad_df['race_group'] = broad_df['Race'].apply(_collapse_race_group)
broad_df['race_group'] = pd.Categorical(
    broad_df['race_group'],
    categories=[
        'White',
        'Black or African American',
        'Hispanic or Latino/a/x',
        'Asian',
        'Multiple',
        'Other (pooled)'
    ],
    ordered=False
)

def _collapse_political_group(x):
    s = str(x).strip()
    if s == "" or s.lower() == "nan":
        return np.nan

    parts = [p.strip() for p in s.split(',') if p.strip()]
    if len(parts) == 0:
        return np.nan
    if 'Prefer not to say' in parts:
        return np.nan

    if len(parts) == 1 and parts[0] in {'Democratic', 'Republican', 'Independent'}:
        return parts[0]
    return 'Other / third party'

broad_df['political_group'] = broad_df['Political Affiliation'].apply(_collapse_political_group)
broad_df['political_group'] = pd.Categorical(
    broad_df['political_group'],
    categories=[
        'Democratic',
        'Republican',
        'Independent',
        'Other / third party'
    ],
    ordered=False
)

def _collapse_season_group(x):
    s = str(x).strip()
    if s == "" or s.lower() == "nan":
        return np.nan
    if s in {'Spring', 'Summer', 'Autumn', 'Winter'}:
        return s
    return np.nan

broad_df['season_group'] = broad_df['Favorite Season'].apply(_collapse_season_group)
broad_df['season_group'] = pd.Categorical(
    broad_df['season_group'],
    categories=['Spring', 'Summer', 'Autumn', 'Winter'],
    ordered=False
)

broad_df['Condition'] = pd.Categorical(
    broad_df['Condition'],
    categories=['Individual', 'Age', 'Conformity', 'Prestige', 'Season'],
    ordered=False
)

valid_broad_gender = ['Male', 'Female']

valid_broad_mask = (
    broad_df['Gender'].isin(valid_broad_gender) &
    broad_df['age_group'].notna() &
    broad_df['edu3'].notna() &
    broad_df['marital2'].notna() &
    broad_df['race_group'].notna() &
    broad_df['political_group'].notna() &
    broad_df['season_group'].notna()
)

broad_df = broad_df.loc[valid_broad_mask].copy()


# =============================================================================
# purpose: fit additional participant-level models for a broader formatted table
# outcomes:
#   - overestimation_score_pp
#   - overestimation_after_pp
#   - revision_rate
#   - absolute_revision_pp
#   - influence_score
# note:
#   - condition is retained for post-treatment outcomes to remain consistent
#     with the experimental design
# =============================================================================

broad_formula_demogs = (
    "C(Gender, Treatment(reference='Male')) + "
    "C(age_group, Treatment(reference='18–29 years')) + "
    "C(edu3, Treatment(reference='low')) + "
    "C(marital2, Treatment(reference='Single, never married')) + "
    "C(race_group, Treatment(reference='White')) + "
    "C(political_group, Treatment(reference='Democratic')) + "
    "C(season_group, Treatment(reference='Autumn'))"
)

broad_model_specs = {
    'Overestimation (initial)': {
        'formula': f"overestimation_score_pp ~ {broad_formula_demogs}",
        'data': broad_df.copy()
    },
    'Overestimation (after)': {
        'formula': (
            "overestimation_after_pp ~ "
            "C(Condition, Treatment(reference='Individual')) + "
            "overestimation_score_pp + "
            f"{broad_formula_demogs}"
        ),
        'data': broad_df.copy()
    },
    'Revision rate': {
        'formula': (
            "revision_rate ~ "
            "C(Condition, Treatment(reference='Individual')) + "
            f"{broad_formula_demogs}"
        ),
        'data': broad_df.copy()
    },
    'MAR': {
        'formula': (
            "absolute_revision_pp ~ "
            "C(Condition, Treatment(reference='Individual')) + "
            f"{broad_formula_demogs}"
        ),
        'data': broad_df.copy()
    },
    'Influence score': {
        'formula': (
            "influence_score ~ "
            "C(Condition, Treatment(reference='Prestige')) + "
            f"{broad_formula_demogs}"
        ),
        'data': broad_df.loc[
            broad_df['Condition'].isin(['Age', 'Conformity', 'Prestige', 'Season'])
        ].copy()
    }
}

# =============================================================================
# purpose: retain OLS estimates while avoiding undefined HC3 uncertainty when
#          an observation has leverage one (e.g., a one-person category)
# note:
#   - HC3 divides residuals by 1 - leverage, which is zero in this case
#   - only coefficients with zero weight on these observations have defined
#     HC3 uncertainty; affected confidence intervals and p-values are NA
#   - no observations, predictors, or coefficient estimates are changed
# =============================================================================
def _hc3_with_leverage_guard(fitted_model):
    design = np.asarray(fitted_model.model.wexog, dtype=float)
    pinv = np.asarray(fitted_model.model.pinv_wexog, dtype=float)
    leverage = np.sum(design * pinv.T, axis=1)
    unit_leverage = np.isclose(leverage, 1.0, rtol=0.0, atol=1e-10)

    with np.errstate(divide='ignore', invalid='ignore'):
        robust_res = fitted_model.get_robustcov_results(cov_type='HC3')
    robust_res.hc3_unavailable_terms = []
    if not unit_leverage.any():
        return robust_res

    # Compute the defined covariance block without the unit-leverage rows.
    # Their variance is not assumed zero: affected coefficients are marked NA.
    valid = ~unit_leverage
    adjusted_residuals = np.asarray(fitted_model.wresid)[valid] / (1.0 - leverage[valid])
    weighted_pinv = pinv[:, valid] * adjusted_residuals
    covariance = weighted_pinv @ weighted_pinv.T
    affected = np.any(np.abs(pinv[:, unit_leverage]) > 1e-10, axis=1)
    covariance[affected, :] = np.nan
    covariance[:, affected] = np.nan

    robust_res.cov_params_default = covariance
    robust_res._cache.clear()
    robust_res.hc3_unavailable_terms = [
        term for term, flag in zip(robust_res.model.exog_names, affected) if flag
    ]
    return robust_res


broad_models = {}

for name, spec in broad_model_specs.items():
    fitted_model = smf.ols(spec['formula'], data=spec['data']).fit()
    fitted_robust = _hc3_with_leverage_guard(fitted_model)
    broad_models[name] = {
        'model': fitted_model,
        'robust': fitted_robust
    }


# =============================================================================
# purpose: print a broader formatted regression table with condition retained
#          in post-treatment models and export it as a PDF
# note:
#   - this is printed output plus PDF export only
#   - it does not modify figure2.csv or the baseline Figure 2 workflow
# =============================================================================

def _term_stats_from_robust(robust_res, term):
    names = list(robust_res.model.exog_names)
    if term == 'Intercept' and term not in names and 'const' in names:
        term = 'const'
    if term not in names:
        return None

    idx = names.index(term)
    coef = float(robust_res.params[idx])
    se = float(robust_res.bse[idx])
    pval = float(robust_res.pvalues[idx])
    df_resid_local = float(robust_res.df_resid)
    tcrit_local = float(stats.t.ppf(0.975, df=df_resid_local))
    ci_low = coef - tcrit_local * se
    ci_high = coef + tcrit_local * se

    return coef, ci_low, ci_high, pval

def _format_b_ci_p(robust_res, term):
    stats_out = _term_stats_from_robust(robust_res, term)
    if stats_out is None:
        return "", "", ""
    coef, ci_low, ci_high, pval = stats_out
    coef_str = f"{coef:.2f}{_stars(pval)}"
    ci_str = f"{ci_low:.2f}, {ci_high:.2f}" if np.isfinite([ci_low, ci_high]).all() else 'NA'
    p_str = _format_p(pval) if np.isfinite(pval) else 'NA'
    return coef_str, ci_str, p_str

broad_outcomes = [
    'Overestimation (initial)',
    'Overestimation (after)',
    'Revision rate',
    'MAR',
    'Influence score'
]

broad_rows = []

def _add_block(label):
    broad_rows.append({'Variable': label, 'row_type': 'block', 'term_map': {}})

def _add_term(label, term_map):
    broad_rows.append({'Variable': label, 'row_type': 'term', 'term_map': term_map})

def _add_spacer():
    broad_rows.append({'Variable': '', 'row_type': 'spacer', 'term_map': {}})

_add_term('Intercept', {outcome: 'Intercept' for outcome in broad_outcomes})
_add_spacer()

_add_block('Condition (reference: Individual; Influence score reference: Prestige)')
for level in ['Age', 'Conformity', 'Prestige', 'Season']:
    term_map = {}
    for outcome in broad_outcomes:
        if outcome in ['Overestimation (after)', 'Revision rate', 'MAR']:
            term_map[outcome] = f"C(Condition, Treatment(reference='Individual'))[T.{level}]"
        elif outcome == 'Influence score':
            if level != 'Prestige':
                term_map[outcome] = f"C(Condition, Treatment(reference='Prestige'))[T.{level}]"
            else:
                term_map[outcome] = None
        else:
            term_map[outcome] = None
    _add_term(level, term_map)
_add_spacer()

_add_block('Baseline covariate')
_add_term('Overestimation (initial)', {
    'Overestimation (initial)': None,
    'Overestimation (after)': 'overestimation_score_pp',
    'Revision rate': None,
    'MAR': None,
    'Influence score': None
})
_add_spacer()

_add_block('Age group (reference: 18–29 years)')
for level in ['30–44 years', '45–64 years']:
    _add_term(level, {
        outcome: f"C(age_group, Treatment(reference='18–29 years'))[T.{level}]"
        for outcome in broad_outcomes
    })
_add_spacer()

_add_block('Education (reference: low)')
for level in ['middle', 'high']:
    _add_term(level.capitalize(), {
        outcome: f"C(edu3, Treatment(reference='low'))[T.{level}]"
        for outcome in broad_outcomes
    })
_add_spacer()

_add_block('Gender (reference: Male)')
_add_term('Female', {
    outcome: "C(Gender, Treatment(reference='Male'))[T.Female]"
    for outcome in broad_outcomes
})
_add_spacer()

_add_block('Marital status (reference: Single, never married)')
for level in ['Married', 'In relationship, but not married', 'Divorced/separated/widowed']:
    _add_term(level, {
        outcome: f"C(marital2, Treatment(reference='Single, never married'))[T.{level}]"
        for outcome in broad_outcomes
    })
_add_spacer()

_add_block('Race / ethnicity (reference: White)')
for level in [
    'Black or African American',
    'Hispanic or Latino/a/x',
    'Asian',
    'Multiple',
    'Other (pooled)'
]:
    _add_term(level, {
        outcome: f"C(race_group, Treatment(reference='White'))[T.{level}]"
        for outcome in broad_outcomes
    })
_add_spacer()

_add_block('Political affiliation (reference: Democratic)')
for level in ['Republican', 'Independent', 'Other / third party']:
    _add_term(level, {
        outcome: f"C(political_group, Treatment(reference='Democratic'))[T.{level}]"
        for outcome in broad_outcomes
    })
_add_spacer()

_add_block('Favorite season (reference: Autumn)')
for level in ['Spring', 'Summer', 'Winter']:
    _add_term(level, {
        outcome: f"C(season_group, Treatment(reference='Autumn'))[T.{level}]"
        for outcome in broad_outcomes
    })
_add_spacer()

_add_term('N', {outcome: '__N__' for outcome in broad_outcomes})
_add_term('Adjusted R-squared', {outcome: '__R2__' for outcome in broad_outcomes})

broad_table_rows = []

for row in broad_rows:
    out_row = {
        'Variable': row['Variable'],
        'row_type': row['row_type']
    }

    if row['row_type'] in {'block', 'spacer'}:
        for outcome in broad_outcomes:
            out_row[f'{outcome} | B'] = ''
            out_row[f'{outcome} | 95% CI'] = ''
            out_row[f'{outcome} | p-value'] = ''
        broad_table_rows.append(out_row)
        continue

    for outcome in broad_outcomes:
        term = row['term_map'].get(outcome)

        if term is None:
            out_row[f'{outcome} | B'] = ''
            out_row[f'{outcome} | 95% CI'] = ''
            out_row[f'{outcome} | p-value'] = ''
            continue

        if term == '__N__':
            out_row[f'{outcome} | B'] = f"{int(broad_models[outcome]['model'].nobs)}"
            out_row[f'{outcome} | 95% CI'] = ''
            out_row[f'{outcome} | p-value'] = ''
            continue

        if term == '__R2__':
            out_row[f'{outcome} | B'] = f"{broad_models[outcome]['model'].rsquared_adj:.4f}"
            out_row[f'{outcome} | 95% CI'] = ''
            out_row[f'{outcome} | p-value'] = ''
            continue

        coef_str, ci_str, p_str = _format_b_ci_p(
            broad_models[outcome]['robust'],
            term
        )
        out_row[f'{outcome} | B'] = coef_str
        out_row[f'{outcome} | 95% CI'] = ci_str
        out_row[f'{outcome} | p-value'] = p_str

    broad_table_rows.append(out_row)

broad_regression_table_df = pd.DataFrame(broad_table_rows)

print("\n" + "=" * 80)
print("Additional Formatted Regression Table")
print("=" * 80)
print(broad_regression_table_df.to_string(index=False))
print("\nNA: HC3 confidence intervals and p-values are not estimable for coefficients "
      "affected by leverage-one observations; OLS coefficients are retained.")


# =============================================================================
# purpose: export the additional formatted regression table as a PDF
# output:
#   - ../output/demographic regression with full measure.pdf
# note:
#   - this PDF export mirrors the printed broad regression table
# =============================================================================

pdf_colnames = ['Variable']
for outcome in broad_outcomes:
    pdf_colnames.extend([f'{outcome} | B', f'{outcome} | 95% CI', f'{outcome} | p-value'])

pdf_table_df = broad_regression_table_df[pdf_colnames].copy()

n_body = len(pdf_table_df)
n_head = 2
n_out = len(broad_outcomes)

fig_w = 28
fig_h = max(9, 0.33 * (n_body + n_head + 5))
fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=200)
ax.axis('off')

tbl = Table(ax, bbox=[0, 0.12, 1, 0.88])

w_var = 0.17
w_sub = (1.0 - w_var) / (3 * n_out)
col_widths = [w_var] + [w_sub] * (3 * n_out)
row_h = 1.0 / (n_body + n_head + 2)

header1 = ['Variable']
for outcome in broad_outcomes:
    header1.extend([outcome, '', ''])

header2 = ['']
for _ in broad_outcomes:
    header2.extend(['B', '95% CI', 'p-value'])

for j, lab in enumerate(header1):
    cell = tbl.add_cell(
        -2, j,
        width=col_widths[j],
        height=row_h,
        text=lab,
        loc='center',
        facecolor='#f0f0f0',
        edgecolor='black'
    )
    cell.get_text().set_weight('bold')

for j, lab in enumerate(header2):
    cell = tbl.add_cell(
        -1, j,
        width=col_widths[j],
        height=row_h,
        text=lab,
        loc='center',
        facecolor='#f0f0f0',
        edgecolor='black'
    )
    cell.get_text().set_weight('bold')

for i in range(n_body):
    variable_label = broad_regression_table_df.iloc[i]['Variable']
    row_type = broad_regression_table_df.iloc[i]['row_type']
    vals = pdf_table_df.iloc[i].tolist()

    is_block = (row_type == 'block')
    is_spacer = (row_type == 'spacer')

    for j, v in enumerate(vals):
        cell = tbl.add_cell(
            i, j,
            width=col_widths[j],
            height=row_h,
            text=str(v),
            loc=('left' if j == 0 else 'center'),
            facecolor='white',
            edgecolor='black'
        )

        if is_block:
            if j == 0:
                cell.get_text().set_weight('bold')
                cell.get_text().set_fontstyle('italic')
            else:
                cell.get_text().set_text('')
        elif is_spacer:
            cell.get_text().set_text('')

ax.add_table(tbl)
tbl.auto_set_font_size(False)
tbl.set_fontsize(8.6)

fig.text(
    0.01, 0.04,
    'Significance: *** p<0.001; ** p<0.01; * p<0.05\n'
    'NA: HC3 confidence intervals and p-values are not estimable for coefficients affected by leverage-one observations; OLS coefficients are retained.',
    ha='left',
    va='bottom',
    fontsize=11,
    fontweight='bold'
)

fig.savefig(
    '../output/demographic regression with full measure.pdf',
    format='pdf',
    bbox_inches='tight',
    pad_inches=0.02
)

plt.close(fig)