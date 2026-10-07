import pandas as pd

# =============================================================================
# purpose: calculate pilot agreement rates and verify the displayed references
# inputs:
#   - ../data/pilot_cleaned.csv
#   - ../data/statements.csv
# measure:
#   - percentage of pilot participants who agree or strongly agree with each item
# output:
#   - ../output/pilot_reference_values.csv
# note:
#   - statements are retained as worded; no reverse coding is applied
#   - reference values are rounded to the nearest whole percentage point
#   - neither input file is modified
# =============================================================================
pilot_df = pd.read_csv('../data/pilot_cleaned.csv')
statements_df = pd.read_csv('../data/statements.csv')

# check the required columns and participant-statement records
required_cols = ['Participant ID', 'Statement ID', 'Agreement-Text', 'Agreement-Value']
missing_cols = [col for col in required_cols if col not in pilot_df.columns]
if missing_cols:
    raise ValueError(f"Missing required columns in pilot_cleaned.csv: {missing_cols}")

if pilot_df[required_cols].isna().any().any():
    raise ValueError('Pilot participant-statement records must be complete.')
if pilot_df.duplicated(['Participant ID', 'Statement ID']).any():
    raise ValueError('Duplicate participant-statement records in pilot_cleaned.csv.')

for col in ['Statement ID', 'Agreement-Value']:
    pilot_df[col] = pd.to_numeric(pilot_df[col], errors='raise')

statement_ids = list(range(1, 13))
if sorted(pilot_df['Statement ID'].unique()) != statement_ids:
    raise ValueError('Pilot Statement ID values must be 1 through 12.')
if not pilot_df.groupby('Participant ID')['Statement ID'].nunique().eq(12).all():
    raise ValueError('Each pilot participant must have all 12 statements.')
if pilot_df['Participant ID'].nunique() != 879:
    raise ValueError('The released pilot dataset must contain 879 participants.')

# use the same agreement coding as estimates.csv
agreement_map = {
    'Strongly disagree': 1,
    'Disagree': 2,
    'Neither agree nor disagree': 3,
    'Agree': 4,
    'Strongly agree': 5
}
mapped_agreement = pilot_df['Agreement-Text'].map(agreement_map)
if mapped_agreement.isna().any() or not mapped_agreement.eq(pilot_df['Agreement-Value']).all():
    raise ValueError('Pilot agreement text and numeric values do not match.')

# check the existing reference table before comparing values
required_reference_cols = ['Statement ID', 'Reference']
missing_reference_cols = [
    col for col in required_reference_cols if col not in statements_df.columns
]
if missing_reference_cols:
    raise ValueError(f"Missing required columns in statements.csv: {missing_reference_cols}")

for col in required_reference_cols:
    statements_df[col] = pd.to_numeric(statements_df[col], errors='raise')
if statements_df[required_reference_cols].isna().any().any():
    raise ValueError('Statement reference records must be complete.')
if sorted(statements_df['Statement ID'].tolist()) != statement_ids:
    raise ValueError('statements.csv must contain one reference for each of the 12 statements.')

# calculate agreement with each statement as worded
pilot_df['Agreed'] = pilot_df['Agreement-Value'].isin([4, 5]).astype(int)
reference_df = pilot_df.groupby('Statement ID', as_index=False).agg(
    N=('Participant ID', 'nunique'),
    **{'Agreement Count': ('Agreed', 'sum')}
)
reference_df['Agreement (%)'] = reference_df['Agreement Count'] / reference_df['N'] * 100.0
reference_df['Calculated Reference'] = reference_df['Agreement (%)'].round().astype(int)
reference_df = reference_df.merge(
    statements_df[required_reference_cols],
    on='Statement ID',
    how='left',
    validate='one_to_one'
)
reference_df['Matches'] = reference_df['Calculated Reference'].eq(reference_df['Reference'])

if not reference_df['Matches'].all():
    mismatched_ids = reference_df.loc[~reference_df['Matches'], 'Statement ID'].tolist()
    raise ValueError(f"Pilot references do not match statements.csv for items: {mismatched_ids}")

# save the calculation without replacing the displayed reference table
reference_df.to_csv('../output/pilot_reference_values.csv', index=False)

print('\n' + '='*80)
print('Pilot Agreement Rates and Displayed Reference Values')
print('='*80)
print(f"Participants: {pilot_df['Participant ID'].nunique()}")
print(reference_df.to_string(index=False, float_format=lambda value: f'{value:.4f}'))
print('\nAll 12 reference values match statements.csv.')
print("Output saved to '../output/pilot_reference_values.csv'.")
