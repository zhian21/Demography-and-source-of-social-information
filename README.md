# Demography and social learning impact norm misperception and belief revision following intervention

## Overview

This repository contains cleaned data, analysis code, qualitative coding records, and documented study materials for the U.S. gender-norm misperception experiment. The main quantitative dataset contains 903 participants, each with personal agreement, initial estimates, and final estimates for 12 statements. A separate pilot dataset contains the agreement responses from 879 participants used to calculate the displayed reference values.

Replication starts from the cleaned datasets in `data/`. The main-study and pilot participants are separate samples. Main-study IDs begin with `P`; pilot IDs begin with `PIL`. They must not be joined as if they identify the same people.

Preregistration: https://doi.org/10.17605/OSF.IO/P6EX3

## Repository files

| File | Contents or purpose |
|---|---|
| `data/estimates.csv` | Main-study participant-by-statement agreement and estimates; 10,836 rows. |
| `data/demographics.csv` | Demographic information for 903 main-study participants. |
| `data/statements.csv` | Full text and displayed reference value for each of the 12 statements. |
| `data/confidence.csv` | Overall confidence in initial estimates; 903 records. |
| `data/ssi.csv` | Four auxiliary social-influence items and the saved composite; 902 complete records. |
| `data/qualitative_coding_final.csv` | Response text, independent coder assignments, and final consensus codes for all 903 participants. |
| `data/pilot_cleaned.csv` | The 12 relevant agreement responses for each of 879 pilot participants; 10,548 rows. |
| `code/pilot-reference-values.py` | Calculate pilot agreement percentages and check all 12 displayed references. |
| `code/main-analysis.py` | Main quantitative outcomes, tests, figure inputs, and derived supplementary-analysis datasets. |
| `code/regressions.py` | Core demographic and extended regressions; Figure 2 input. |
| `code/si-anova.py` | Participant-level ANOVAs corresponding to Tables S11–S13. |
| `code/si-randomization.py` | Demographic balance tests and descriptive breakdowns. |
| `code/si-robustness.py` | Gender, initial-error, statement, outlier, and political-affiliation checks. |
| `code/figures.py` | Figures 1–5 from the generated figure-input CSVs. |
| `DATA_DICTIONARY.md` | Column definitions, response coding, joins, derived measures, and qualitative calculation procedures. |
| `requirements.txt` | Exact versions of the direct analysis dependencies. |
| `materials/study_materials_and_protocol.pdf` | Statement bank, available question wording, source framing, pilot reference derivation, and documented procedures. |

This repository contains 18 source and documentation files. The supplementary information will be available with the published article on the publisher’s website and is not included in this repository.
## Environment

The scripts were tested with Python 3.13.5 and the package versions in `requirements.txt`. Jinja2 is included because the existing pandas LaTeX export requires it. The qualitative CSV can be opened in spreadsheet software or read as UTF-8 text; none of the seven scripts reads this file. No qualitative-analysis script is supplied.

The tested replication environment is distinct from the historical analysis environment described in the manuscript. No network access is needed during analysis once the dependencies have been installed.

## Run the analyses

Keep `code/`, `data/`, and `materials/` together. Do not rename the files listed above: the existing scripts use relative paths. Run all analysis commands from `code/`, not from the repository root.

The following commands assume a macOS or Linux terminal and an installed Python 3.13.5 interpreter. From the repository root:

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
mkdir -p output
export MPLBACKEND=Agg
export PYTHONIOENCODING=utf-8
cd code
python pilot-reference-values.py > ../output/pilot-reference-values.log
python main-analysis.py > ../output/main-analysis.log
python regressions.py > ../output/regressions.log
python si-anova.py > ../output/si-anova.log
python si-randomization.py > ../output/si-randomization.log
python si-robustness.py > ../output/si-robustness.log
python figures.py > ../output/figures.log
```

`MPLBACKEND=Agg` saves figures without opening interactive windows. Omit the output redirection to display the printed results in the terminal. Existing noninteractive-display and pandas future-default warnings do not prevent the verified outputs in the tested environment.

`main-analysis.py` creates `data/complete-anova.csv`, `data/randomization-check.csv`, and `data/robustness-check.csv`. The supplementary scripts read these generated files, so they do not need to be supplied as separate inputs. Rerunning the workflow replaces these derived datasets and the files in `output/`; it does not change the source datasets.

## Results and outputs

| Analysis or manuscript display | Script and output |
|---|---|
| Pilot-derived references and statement reference table | `pilot-reference-values.py`; `output/pilot_reference_values.csv`. |
| Figure 1 and Table S8 | `main-analysis.py`; `figure1a-b.csv`, `figure1c-d.csv`, and `t_test_results.tex`; `figures.py` draws Figure 1. The `c-d` filename is retained from the existing workflow and supplies current panel C. |
| Figure 2 and Table S2 | `regressions.py`; printed core regression and `figure2.csv`; `figures.py` draws Figure 2. |
| Tables S3–S4 | `regressions.py`; printed extended table and `demographic regression with full measure.pdf`. |
| Figures 3–5 and main condition comparisons | `main-analysis.py`; `figure3a.csv`, `figure3b.csv`, `figure4a.csv`, `figure4b.csv`, `figure4c.csv`, and `figure5.csv`; `figures.py` draws the figures. |
| Table S9 and demographic breakdowns | `si-randomization.py`; printed counts/proportions and `balance_check_results.csv`. Full S7/S10 publication-format tables are not exported. |
| Tables S11–S13 | `si-anova.py`; printed ANOVA tables, descriptive statistics, and LaTeX text. |
| Tables S14–S19 | `si-robustness.py`; printed sensitivity analyses and `statement_level_results.csv` for the statement-level table. |
| Qualitative theme frequencies | `data/qualitative_coding_final.csv` and the procedures in `DATA_DICTIONARY.md`; no automatic qualitative output is generated. |

All listed output filenames are relative to `output/`, except the three generated datasets explicitly placed in `data/`.

For a basic check, baseline misperception rounds to 18.27 percentage points. MAR rounds to 2.72 in control and 10.49 in pooled treatments; revision rates round to 20.56% and 54.39%. The original, winsorized, and trimmed error-change ANOVAs round to F = 45.98, 48.94, and 50.08. The pilot script reports that all 12 references match `statements.csv`.

The extended regression tables retain the OLS coefficients but print `NA` for HC3 confidence intervals and p-values of the one-observation “Other (pooled)” race category. The table note explains this limitation. Undefined uncertainty is not a zero effect or a non-significant test result.

## Qualitative records

`data/qualitative_coding_final.csv` contains one record for each of the 903 main-study participants. A randomly selected subset of 181 responses was used to develop and refine the codebook. The finalized codebook was applied to all 903 responses, including the development subset. Theme frequencies use the `Consensus` column for the full sample, with the number of records within each condition as the denominator.

The `Coder 1` and `Coder 2` columns retain the independent assignments. Use the coding-stage participant IDs documented in `DATA_DICTIONARY.md` to reproduce stage-specific Cohen’s kappa. The code definitions and hierarchical coding rule are described in the study-materials document and supplementary methods. The dictionary gives the column definitions and calculation steps.

## Documentation scope

The study-materials PDF is a source-based compilation of documented procedures and available question wording, not a runnable Qualtrics survey or a reconstruction of unavailable survey programming. The original recruitment exports and raw-to-clean processing notebooks are not inputs to this cleaned-data replication workflow. Reference-file content is checked against the pilot responses; it is not estimated from the main-study sample.
