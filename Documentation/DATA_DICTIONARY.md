# Data dictionary

## 1. Shared conventions

CSV files use UTF-8 encoding, one header row, and no saved dataframe index. Column names are case-sensitive and are retained as used by the scripts. Main-study participant IDs are `P0001`–`P0903`; pilot IDs are `PIL0001`–`PIL0879`. These are separate identifier systems for separate samples. Join main-study files on `Participant ID`; join statement text and references on `Statement ID`.

`Statement ID` ranges from 1 to 12 and identifies content, not presentation order. Statements 1–6 are equitable as worded; statements 7–12 are inequitable. Full wording is stored once in `statements.csv`. Statements pair by domain: sexual autonomy 1/7, education 2/8, family autonomy 3/9, professional equality 4/10, political participation 5/11, and marital autonomy 6/12.

Condition labels are `Individual`, `Age`, `Conformity`, `Prestige`, and `Season`. `Individual` is the no-feedback control and is displayed as “Control” in most manuscript figures. Other labels identify the attributed source, not the participant’s demographic category. `Age` refers to a source aged 40 and above; the participant-age regression categories are different.

## 2. `data/estimates.csv`

One row per main-study participant and statement: 10,836 rows, 903 participants, and 12 statements each. All seven columns are complete.

| Column | Definition and coding |
|---|---|
| `Participant ID` | Main-study anonymous participant identifier. |
| `Condition` | Assigned experimental condition; constant within participant. |
| `Statement ID` | Integer 1–12, linking to `statements.csv`. |
| `Agreement-Text` | Participant’s personal agreement with the statement as worded. |
| `Agreement-Value` | Matching integer agreement code, defined below. |
| `Initial Estimate` | Estimated percentage agreeing before the revision task; 0–100. |
| `Final Estimate` | Estimated percentage agreeing after the revision task; 0–100. May equal the initial value. |

Personal agreement uses the same coding in the main and released pilot datasets:

| `Agreement-Text` | `Agreement-Value` |
|---|---:|
| Strongly disagree | 1 |
| Disagree | 2 |
| Neither agree nor disagree | 3 |
| Agree | 4 |
| Strongly agree | 5 |

Agreement for a reference or observed norm means a response of 4 or 5. Neutral responses remain in the denominator and do not count as agreement. The numerical estimate columns are not Likert responses. The stored values are not reverse-coded.

## 3. `data/demographics.csv`

One row per main-study participant: 903 rows. All eight columns are complete; “Prefer not to say” is an explicit response rather than an empty cell.

| Column | Definition and observed coding |
|---|---|
| `Participant ID` | Key linking demographics to the main-study files. |
| `Gender` | `Male`, `Female`, `Other`, or `Prefer not to say`. |
| `Age` | Age in whole years; observed range 18–64. |
| `Education` | Highest level attended: `Less than high school`; `High school diploma or equivalent`; `Some college (no degree)`; `Associate degree`; `Bachelor's degree`; or `Graduate or professional degree (Masters, PhD, JD, MD, etc.)`. |
| `Marital Status` | `Single, never married`; `Married`; `In relationship, but not married`; `Divorced or separated`; or `Widowed`. |
| `Race` | Recorded race/ethnicity selections. Multiple selections are stored together, separated by commas. |
| `Political Affiliation` | Recorded affiliation, including combined selections. Main single labels include `Democratic`, `Republican`, `Independent`, `Green Party`, `Libertarian`, and `Prefer not to say`. |
| `Favorite Season` | `Spring`, `Summer`, `Autumn`, or `Winter`. |

The core regression uses age groups 18–29, 30–44, and 45–64. Low education combines less than high school, high school/equivalent, and some college without a degree. Middle education combines associate and bachelor’s degrees. High education is graduate/professional education. The core regression uses the 900 participants recorded as Male or Female.

The extended models also code marital status, race, political affiliation, and season. Divorced/separated and widowed responses are combined. Race is coded as White, Black or African American, Hispanic or Latino/a/x, Asian, Multiple, or Other (pooled); refusal is not a modeled race category. Exact single Democratic, Republican, and Independent responses retain those groups; other non-refusal political responses are grouped as Other / third party. The extended models use 872 complete valid records, or 695 treatment records for influence. `regressions.py` contains the exact coding and model restrictions.

For balance tests, `si-randomization.py` treats each recorded category string as a category. In the political robustness check, `si-robustness.py` instead assigns strings containing “Democratic” to Democrat, then strings containing “Republican” to Republican, and exact “Independent” to Independent. These analysis-specific definitions should not be substituted for one another. Similarly, counting any White selection differs from counting White-only responses.

## 4. `data/statements.csv`

One row for each of 12 statements; all columns complete.

| Column | Definition |
|---|---|
| `Statement ID` | Integer 1–12 used throughout the release. |
| `Statement Text` | Full wording used to identify the item. |
| `Reference` | Whole-percentage-point pilot agreement value displayed in the four feedback conditions. |

References in statement order are 93, 96, 91, 60, 71, 95, 10, 5, 3, 15, 15, and 8. They are rounded pilot agreement rates, not subgroup estimates, not averages of pilot estimates, and not main-study observed agreement. The same values are used across all four feedback conditions; control receives no reference value.

## 5. `data/pilot_cleaned.csv`

One row per retained pilot participant and relevant statement: 10,548 rows, 879 participants, and 12 items each. Its four columns are an exact-name subset of the main `estimates.csv` schema: `Participant ID`, `Statement ID`, `Agreement-Text`, and `Agreement-Value`. Only the personal agreement responses needed for the references are released. All four columns are complete.

The pilot was the earlier 18-item survey described in Chen’s May 2025 thesis, *Peer Support for Inequitable Gender Norms: A Mixed-Methods Study of Perception and Learning*. The release includes the 12 items used by the present experiment, not the entire thesis dataset. There is no treatment condition or final-estimate column in this extract. Pilot IDs are anonymous release IDs and do not link to main-study IDs.

| Release Statement ID | Original pilot item |
|---:|---|
| 1 | G3 |
| 2 | G5 |
| 3 | G7 |
| 4 | G15 |
| 5 | G17 |
| 6 | G9 |
| 7 | G4 |
| 8 | G6 |
| 9 | G14 |
| 10 | G16 |
| 11 | G18 |
| 12 | G10 |

For each item, divide the number of Agree/Strongly agree responses by 879, multiply by 100, and round to the nearest whole percentage point. Do not reverse-code equitable items when calculating references. `pilot-reference-values.py` verifies this calculation against `statements.csv` without replacing either source file.

## 6. `data/confidence.csv`

One overall response per main-study participant: 903 rows, three complete columns.

| Column | Definition |
|---|---|
| `Participant ID` | Main-study identifier. |
| `Condition` | Assigned condition; retained for linkage, not evidence that confidence was measured after feedback. |
| `Confidence` | Overall confidence in initial estimates: `Not at all confident`, `A little confident`, `Somewhat confident`, `Quite confident`, or `Very confident`. |

Confidence was collected after initial estimates and before the revision stage, as shown in the supplementary procedure diagram. This is not an item-specific measure. It is shared as an auxiliary measure and is not read by the six main-analysis scripts.

## 7. `data/ssi.csv`

One complete four-item response record per participant: 902 rows and seven complete columns. One main-study participant with an incomplete item response is not present. The six main-analysis scripts do not use this file.

| Column | Definition |
|---|---|
| `Participant ID` | Main-study identifier. |
| `Condition` | Assigned condition. |
| `Q1` | “I often consult other people to help choose the best alternative available from a product class.” |
| `Q2` | “To make sure I buy the right product or brand, I often observe what others are buying and using.” |
| `Q3` | “If I have little experience with a product, I often ask my friends about the product.” |
| `Q4` | “I frequently gather information from friends or family about a product before I buy.” |
| `Score` | Saved mean of the four items after mapping their responses to the 0–1 scale below. |

The numeric mapping that reproduces every saved score is Strongly disagree = 0, Somewhat disagree = 0.25, Neither agree nor disagree = 0.50, Somewhat agree = 0.75, and Strongly agree = 1. Q3/Q4 store the highest category as `Strong agree`; Q1/Q2 use `Strongly agree`. Both labels map to 1. No reverse coding is used in the saved composite. The item wording is reproduced from the survey export; this dictionary does not infer an instrument citation or additional psychometric properties.

## 8. `data/qualitative_coding_final.xlsx`

The workbook has two nonoverlapping worksheets containing all 903 main-study participant IDs. `pilot_coding` has 181 codebook-development records. `final_coding` has 722 subsequent records and final consensus classifications. The latter worksheet alone supplies theme frequencies for the current sharing plan.

| Column | Definition |
|---|---|
| `Participant ID` | Main-study identifier; joins to `estimates.csv`. |
| `Condition` | Assigned experimental condition. |
| `Revised` | 1 if at least one initial estimate was changed; 0 if all 12 were unchanged. This is not the participant’s percentage revision rate. |
| `Reason` | Response to the open-ended explanation prompt, retained as supplied. One cell is blank; its coding record remains present. |
| `Coder_1`, `Coder_2` | Independent categorical assignments before consensus. |
| `Consensus` | Final agreed category; present only in `final_coding`. |

The final-coding condition counts are Age 143, Conformity 147, Individual 142, Prestige 146, and Season 144. Count each record once, including records with `Revised = 0`; a blank response cell is not a reason to remove a record that has a retained consensus classification.

To calculate a theme percentage, count records assigned that theme in `Consensus` within a condition, divide by that condition’s final-coding count, and multiply by 100. Zero-count themes are zero, whether displayed as 0% or a dash. Do not infer pilot consensus from matching independent codes or combine the 181 pilot-coded records into these denominators.

For unweighted Cohen’s kappa, use `Coder_1` and `Coder_2` separately within each worksheet, before consensus. Let observed agreement be the fraction of identical assignments. Expected agreement is the sum, across all categories, of the product of the two coders’ marginal proportions. Kappa is (observed agreement − expected agreement) / (1 − expected agreement). Treat category identifiers consistently as labels, not numerical scores. Consensus codes are not used to calculate inter-rater kappa.

General codes are 1 Normative Alignment; 2 Discrepancy Magnitude Sensitivity; 3 Confidence-Weighted Updating; 4 Reflective Reconsideration; 5 Prior Overgeneralized Pessimism; 6 Population Scope Adjustment; and 7 Other. Condition-specific codes are A1 Generational Ideology Assumption; A2 Age-Based Knowledge Attribution; C1 Majority Accuracy Belief; C2 Majority Distrust; I1 Absence of External Cue; E1 Elite–Mass Disconnect; E2 Elite Deference; S1 Cue Irrelevance Assessment; and S2 Random Sample Inference. The materials PDF reproduces the coding definitions. `S1` and `S2` here are qualitative codes, not numeric statement IDs.

## 9. Derived quantitative measures

These measures are calculated by the scripts; they are not additional source responses. Let A be the main-study agreement percentage for a statement, R its pilot reference, E0 the initial estimate, and Ef the final estimate.

| Measure | Definition |
|---|---|
| Signed misperception | For statements 1–6, A − E; for statements 7–12, E − A. Positive values indicate overestimation of gender-inequitable support on the recoded scale. |
| MAR | Absolute value of Ef − E0; averaged over 12 items for each participant. |
| Revision rate | 100 times the fraction of a participant’s 12 estimates with Ef different from E0. |
| Initial/final estimation error | Absolute value of E0 − A or Ef − A; averaged over 12 items. |
| Change in estimation error | Final error minus initial error. Negative values indicate improved accuracy. The generated files retain the existing column name `Error Reduction` for this quantity. |
| Influence score | (Ef − E0) / (R − E0), calculated in raw agreement space. Undefined for control and for items with E0 = R; those items are omitted from the participant’s mean. |
| Misperception change | Final signed misperception minus initial signed misperception; distinct from change in absolute estimation error. |

For equitable items, the common-scale calculation uses 100 − agreement and 100 − the estimate. This complement includes neutral and disagreeing responses; it must not be replaced with the percentage explicitly disagreeing.

The three generated datasets are `complete-anova.csv` (10,836 rows; Participant ID, Condition, Error Reduction, MAR, Final Error), `randomization-check.csv` (10,836 rows; participant/condition and repeated demographics), and `robustness-check.csv` (10,836 rows; 18 columns containing estimates, references, observed agreement, outcomes, and selected demographics). `si-anova.py` averages to participant level; `si-randomization.py` removes repeated participant rows. These row counts do not represent additional participants.

In generated `figure1c-d.csv`, `Agreement` and `Initial Estimate` have already been put on the common inequity scale. In `robustness-check.csv`, `Agreement` is observed agreement with the statement as worded. Do not reverse either file again without checking its role. Undefined influence values are stored as empty CSV cells.

## 10. Pilot calculation output

`output/pilot_reference_values.csv` contains one row per statement and the columns `Statement ID`, `N`, `Agreement Count`, `Agreement (%)`, `Calculated Reference`, `Reference`, and `Matches`. `Reference` is read from the source statement table; `Calculated Reference` is derived from the pilot. All 12 `Matches` values must be True. This output is a verification table, not a replacement source dataset.
