# =============================================================================
# Figure 1
# purpose:
#   build Figure 1 from the CSV outputs produced by main_analysis.py
# panels:
#   - Panel A: Likert distributions for Statements 1–6
#   - Panel B: Likert distributions for Statements 7–12
#   - Panel C: observed norm and condition-specific estimated support for
#              gender inequitable norms across Statements 1–12
# output:
#   - ../output/Figure 1.pdf
# =============================================================================

from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib.offsetbox import AnchoredOffsetbox, VPacker, HPacker, TextArea, DrawingArea
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ----------------------------
# 0. Load figure input CSVs
# ----------------------------
assessment_df = pd.read_csv('../output/figure1a-b.csv')
misperception_df = pd.read_csv('../output/figure1c-d.csv')

# ----------------------------
# 1. Global font sizes
# ----------------------------
FS_YLABEL = 16
FS_YLABEL_AB = 22
FS_XLABEL = 22
FS_XTICKS = 23
FS_YTICKS = 16
FS_LEGEND = 14
FS_INBAR = 17

# ----------------------------
# 2. Constants and helpers
# ----------------------------
STATEMENT_IDS = list(range(1, 13))
LEFT_IDS = [1, 2, 3, 4, 5, 6]
RIGHT_IDS = [7, 8, 9, 10, 11, 12]

STATEMENT_ID_TO_SHORT = {
    1:  "A woman can refuse sex under any circumstance.",
    2:  "It is important for girls and women to be educated.",
    3:  "A woman can leave her husband even if he disagrees.",
    4:  "Men and women are equally suitable for all professions.",
    5:  "Women should have more political opportunities.",
    6:  "A woman can live successfully without marrying a man.",
    7:  "Women exaggerate sexual violence to gain advantage.",
    8:  "University education matters more for boys than girls.",
    9:  "A mistreated mother should stay for the children.",
    10: "Women losing fairly to men often claim discrimination.",
    11: "Men make better political leaders than women do.",
    12: "A woman has to have children in order to be fulfilled.",
}

CAT_ORDER = ['SD', 'D', 'N', 'A', 'SA']
COLOR_MAP = {
    'SD': '#7b0f0f',
    'D':  '#d67c7c',
    'N':  '#d2d2d2',
    'A':  '#6baed6',
    'SA': '#2171b5',
}
LEGEND_LABELS = {
    'SD': 'Strongly disagree',
    'D':  'Disagree',
    'N':  'Neutral',
    'A':  'Agree',
    'SA': 'Strongly agree',
}

LIKERT_COL_MAP = {
    'Strongly disagree': 'SD',
    'Disagree': 'D',
    'Neither agree nor disagree': 'N',
    'Agree': 'A',
    'Strongly agree': 'SA',
}

def ceil_to_5(x):
    if not np.isfinite(x):
        return 0.0
    return float(np.ceil(x / 5.0) * 5.0)

def floor_to_5(x):
    if not np.isfinite(x):
        return 0.0
    return float(np.floor(x / 5.0) * 5.0)

# ----------------------------
# 3. Panel A/B data: Likert distributions
# ----------------------------
assessment_df = assessment_df.copy()
assessment_df['Statement ID'] = pd.to_numeric(assessment_df['Statement ID'], errors='raise')
assessment_df = assessment_df.sort_values('Statement ID').reset_index(drop=True)

# make sure all expected Likert columns exist
for full_col in LIKERT_COL_MAP.keys():
    if full_col not in assessment_df.columns:
        assessment_df[full_col] = 0.0

rows_f1 = []
for sid in STATEMENT_IDS:
    row = assessment_df.loc[assessment_df['Statement ID'] == sid]
    if row.empty:
        continue
    row = row.iloc[0]

    rows_f1.append({
        'Statement ID': sid,
        'num': str(sid),
        'SD': float(pd.to_numeric(row['Strongly disagree'], errors='coerce')),
        'D':  float(pd.to_numeric(row['Disagree'], errors='coerce')),
        'N':  float(pd.to_numeric(row['Neither agree nor disagree'], errors='coerce')),
        'A':  float(pd.to_numeric(row['Agree'], errors='coerce')),
        'SA': float(pd.to_numeric(row['Strongly agree'], errors='coerce')),
    })

likert_df_f1 = pd.DataFrame(rows_f1).set_index('Statement ID').loc[STATEMENT_IDS]
left_df_f1 = likert_df_f1.loc[LEFT_IDS].copy()
right_df_f1 = likert_df_f1.loc[RIGHT_IDS].copy()

# ----------------------------
# 4. Panel C data: unified gender-inequity scale
# ----------------------------
misperception_df = misperception_df.copy()
misperception_df['Statement ID'] = pd.to_numeric(misperception_df['Statement ID'], errors='raise')
misperception_df['Agreement'] = pd.to_numeric(misperception_df['Agreement'], errors='raise')
misperception_df['Initial Estimate'] = pd.to_numeric(misperception_df['Initial Estimate'], errors='raise')

cond_order_f1 = ['Individual', 'Prestige', 'Conformity', 'Age', 'Season']
cond_colors_f1 = {
    'Individual': '#7f8c8d',
    'Prestige': '#E67E22',
    'Conformity': '#008000',
    'Age': '#3498DB',
    'Season': '#9B59B6',
}

actual_f1 = (
    misperception_df[['Statement ID', 'Agreement']]
    .drop_duplicates(subset=['Statement ID'])
    .sort_values('Statement ID')['Agreement']
    .to_numpy(dtype=float)
)

cond_means_f1 = []
for cond in cond_order_f1:
    cond_df = (
        misperception_df[misperception_df['Condition'] == cond]
        .sort_values('Statement ID')
    )
    cond_df = cond_df.set_index('Statement ID').reindex(STATEMENT_IDS)
    cond_means_f1.append(cond_df['Initial Estimate'].to_numpy(dtype=float))

cond_means_f1 = np.vstack(cond_means_f1)

# ----------------------------
# 5. Figure building: top row A/B, bottom row C only
# ----------------------------
fig_f1 = plt.figure(figsize=(18, 16), dpi=300)
gs = fig_f1.add_gridspec(
    nrows=2, ncols=2,
    height_ratios=[3.2, 2.1],
    hspace=0.22,
    wspace=0.25
)

axA = fig_f1.add_subplot(gs[0, 0])
axB = fig_f1.add_subplot(gs[0, 1])
axC = fig_f1.add_subplot(gs[1, :])

fig_f1.subplots_adjust(
    left=0.08,
    right=0.90,
    top=0.95,
    bottom=0.08,
    hspace=0.22,
    wspace=0.25
)

# ----------------------------
# Panel A: Likert bars (1–6)
# ----------------------------
data = left_df_f1
y = np.arange(len(data))[::-1]
cum = np.zeros(len(data))

for cat in CAT_ORDER:
    vals = data[cat].to_numpy(dtype=float)
    axA.barh(
        y, vals, left=cum,
        color=COLOR_MAP[cat],
        edgecolor='white', linewidth=0.4, height=0.6
    )
    for i, w in enumerate(vals):
        if w >= 5:
            axA.text(
                cum[i] + w / 2, y[i], f"{w:.0f}",
                ha='center', va='center',
                fontsize=FS_INBAR, color='white'
            )
    cum += vals

axA.set_xlim(0, 100)
axA.set_xticks([0, 25, 50, 75, 100])
axA.set_yticks(y)
axA.set_yticklabels([])
axA.tick_params(axis='y', left=False)
axA.set_xticklabels([0, 25, 50, 75, 100], fontsize=FS_XTICKS)
axA.tick_params(axis='x', labelsize=FS_XTICKS)
for side in ['top', 'right', 'left']:
    axA.spines[side].set_visible(False)

for i, sid in enumerate(LEFT_IDS):
    label = f"{sid}. {STATEMENT_ID_TO_SHORT[sid]}"
    axA.text(
        0.0, y[i] + 0.35, label,
        ha='left', va='bottom',
        fontsize=FS_YTICKS, color='black',
        clip_on=True
    )

axA.set_ylabel("Statement (Equitable belief)", fontsize=FS_YLABEL_AB, fontweight='bold')
axA.set_xlabel("% of participants", fontsize=FS_XLABEL, fontweight='bold')

# ----------------------------
# Panel B: Likert bars (7–12)
# ----------------------------
data = right_df_f1
y = np.arange(len(data))[::-1]
cum = np.zeros(len(data))

for cat in CAT_ORDER:
    vals = data[cat].to_numpy(dtype=float)
    axB.barh(
        y, vals, left=cum,
        color=COLOR_MAP[cat],
        edgecolor='white', linewidth=0.4, height=0.6
    )
    for i, w in enumerate(vals):
        if w >= 5:
            axB.text(
                cum[i] + w / 2, y[i], f"{w:.0f}",
                ha='center', va='center',
                fontsize=FS_INBAR, color='white'
            )
    cum += vals

axB.set_xlim(0, 100)
axB.set_xticks([0, 25, 50, 75, 100])
axB.set_yticks(y)
axB.set_yticklabels([])
axB.tick_params(axis='y', left=False)
axB.set_xticklabels([0, 25, 50, 75, 100], fontsize=FS_XTICKS)
axB.tick_params(axis='x', labelsize=FS_XTICKS)
for side in ['top', 'right', 'left']:
    axB.spines[side].set_visible(False)

for i, sid in enumerate(RIGHT_IDS):
    label = f"{sid}. {STATEMENT_ID_TO_SHORT[sid]}"
    axB.text(
        0.0, y[i] + 0.35, label,
        ha='left', va='bottom',
        fontsize=FS_YTICKS, color='black',
        clip_on=True
    )

axB.set_ylabel("Statement (Inequitable belief)", fontsize=FS_YLABEL_AB, fontweight='bold')
axB.set_xlabel("% of participants", fontsize=FS_XLABEL, fontweight='bold')

likert_handles = [Patch(facecolor=COLOR_MAP[c]) for c in CAT_ORDER]
likert_labels = [LEGEND_LABELS[c] for c in CAT_ORDER]
axB.legend(
    likert_handles, likert_labels,
    frameon=True, fancybox=True, framealpha=1.0,
    fontsize=FS_LEGEND,
    loc='upper left',
    bbox_to_anchor=(1.03, 0.95),
    borderaxespad=0.0
)

# ----------------------------
# Panel C: unified scatter (1–12)
# ----------------------------
xC = np.arange(1, 13)
axC.scatter(xC, actual_f1, color='red', marker='s', s=100, label='Observed norm')

offsets = np.linspace(-0.30, 0.30, len(cond_order_f1))
for off, cond, row in zip(offsets, cond_order_f1, cond_means_f1):
    axC.scatter(
        xC + off, row,
        color=cond_colors_f1[cond],
        marker='d', s=100, label=cond
    )

axC.set_xticks(xC)
axC.set_xticklabels([str(i) for i in range(1, 13)], fontsize=FS_XTICKS)

vals_C = np.r_[actual_f1, cond_means_f1.ravel()]
min_C = float(np.nanmin(vals_C))
max_C = float(np.nanmax(vals_C))
ymin_C = max(0.0, floor_to_5(min_C - 10.0))
ymax_C = ceil_to_5(max_C + 10.0)

axC.set_ylim(ymin_C, ymax_C)
axC.set_xlabel("Statement", fontsize=FS_XLABEL, fontweight='bold')
axC.set_ylabel("% estimated to support gender inequitable belief", fontsize=13.5, fontweight='bold')
axC.grid(axis='y', linestyle=':', alpha=0.35)
axC.tick_params(axis='x', labelsize=FS_XTICKS)
axC.tick_params(axis='y', labelsize=FS_YTICKS)

# ----------------------------
# Legend for Panel C
# ----------------------------
def _marker_box(marker, face, edge, size=8):
    da = DrawingArea(18, 16, 0, 0)
    ln = Line2D([9], [8], marker=marker, linestyle='None',
                markersize=size, markerfacecolor=face,
                markeredgecolor=edge, color=edge)
    da.add_artist(ln)
    return da

IND_PX = 22

row_actual_hdr = TextArea("Observed Norm", textprops={"size": FS_LEGEND})
row_all = HPacker(
    children=[
        DrawingArea(IND_PX, 1, 0, 0),
        _marker_box("s", "red", "red", size=8),
        TextArea("All", textprops={"size": FS_LEGEND}),
    ],
    align='center',
    pad=0,
    sep=6
)

row_spacer = TextArea(" ", textprops={"size": FS_LEGEND})
row_est_hdr = TextArea("Estimated Norm", textprops={"size": FS_LEGEND})

rows_conds = []
for c in cond_order_f1:
    rows_conds.append(
        HPacker(
            children=[
                DrawingArea(IND_PX, 1, 0, 0),
                _marker_box("D", cond_colors_f1[c], cond_colors_f1[c], size=8),
                TextArea(c, textprops={"size": FS_LEGEND}),
            ],
            align='center',
            pad=0,
            sep=6
        )
    )

legend_box = VPacker(
    children=[row_actual_hdr, row_all, row_spacer, row_est_hdr] + rows_conds,
    align='left',
    pad=0,
    sep=4
)

anch = AnchoredOffsetbox(
    loc='upper left',
    child=legend_box,
    frameon=True,
    bbox_to_anchor=(1.02, 1.0),
    bbox_transform=axC.transAxes,
    borderpad=0.6
)
axC.add_artist(anch)

# ----------------------------
# Subplot tags
# ----------------------------
fig_f1.canvas.draw()
TAG_DY = 0.017
TAG_DX = 0.015

def _place_tag(fig, ax, text, dy=TAG_DY, dx=TAG_DX, fs=18, weight='bold'):
    bb = ax.get_position()
    x = max(0.0, bb.x0 - dx)
    y = bb.y1 + dy
    fig.text(x, y, text, ha='left', va='bottom', fontsize=fs, fontweight=weight)

_place_tag(fig_f1, axA, "A)", dy=0.005)
_place_tag(fig_f1, axB, "B)", dy=0.005)
_place_tag(fig_f1, axC, "C)")

fig_f1.savefig(
    '../output/Figure 1.pdf',
    format='pdf',
    bbox_inches='tight'
)

plt.show()

# =============================================================================
# Figure 2
# purpose:
#   build Figure 2 from the CSV output produced by regressions.py
# panels:
#   - Panel A: predicted initial overestimation by age group
#   - Panel B: predicted initial overestimation by education group
#   - Panel C: predicted initial overestimation by gender
# output:
#   - ../output/Figure 2.pdf
# =============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

figure2_df = pd.read_csv('../output/figure2.csv')

required_cols = [
    'panel', 'panel_order', 'level_display', 'level_order',
    'n', 'adjusted_mean', 'ci_lower', 'ci_upper', 'stars'
]
missing_cols = [c for c in required_cols if c not in figure2_df.columns]
if missing_cols:
    raise ValueError(f"Missing required columns in figure2.csv: {missing_cols}")

figure2_df['panel_order'] = pd.to_numeric(figure2_df['panel_order'], errors='raise')
figure2_df['level_order'] = pd.to_numeric(figure2_df['level_order'], errors='raise')
figure2_df['n'] = pd.to_numeric(figure2_df['n'], errors='raise')
figure2_df['adjusted_mean'] = pd.to_numeric(figure2_df['adjusted_mean'], errors='raise')
figure2_df['ci_lower'] = pd.to_numeric(figure2_df['ci_lower'], errors='raise')
figure2_df['ci_upper'] = pd.to_numeric(figure2_df['ci_upper'], errors='raise')
figure2_df['stars'] = figure2_df['stars'].fillna('').astype(str)

# revise displayed gender labels only
figure2_df['level_display'] = figure2_df['level_display'].replace({
    'Male': 'Men',
    'Female': 'Women',
    '18–29 years': '18–29',
    '30–44 years': '30–44',
    '45–64 years': '45–64',
})

figure2_df = (
    figure2_df.sort_values(['panel_order', 'level_order'])
    .reset_index(drop=True)
)

# keep the figure panel order fixed as Age, Education, Gender
panel_names = ['Age group', 'Education', 'Gender']

available_panels = set(figure2_df['panel'].dropna().unique().tolist())
missing_panels = [p for p in panel_names if p not in available_panels]
if missing_panels:
    raise ValueError(
        f"Figure 2 expects panels {panel_names}, but these were missing from figure2.csv: {missing_panels}"
    )

all_ci = pd.concat([figure2_df['ci_lower'], figure2_df['ci_upper']], axis=0)
xmin = float(all_ci.min())
xmax = float(all_ci.max())
pad = 0.10 * (xmax - xmin) if xmax > xmin else 1.0
xlim = (xmin - pad, xmax + pad)

FS_TITLE = 18
FS_AXLAB = 17
FS_TICKS = 15
FS_ANN = 14

level_color_map = {
    '18–29 years': '#7f7f7f',
    '30–44 years': '#1f77b4',
    '45–64 years': '#2ca02c',
    '18–29': '#7f7f7f',
    '30–44': '#1f77b4',
    '45–64': '#2ca02c',
    'Low': '#7f7f7f',
    'Middle': '#1f77b4',
    'High': '#2ca02c',
    'Men': '#7f7f7f',
    'Women': '#1f77b4',
    'Male': '#7f7f7f',
    'Female': '#1f77b4',
}

fig, axes = plt.subplots(ncols=3, nrows=1, figsize=(14, 4.2), dpi=300)

for ax, panel_name in zip(axes, panel_names):
    panel_df = (
        figure2_df[figure2_df['panel'] == panel_name]
        .sort_values('level_order')
        .reset_index(drop=True)
    )

    y_pos = np.arange(len(panel_df))[::-1]
    means = panel_df['adjusted_mean'].to_numpy(dtype=float)
    ci_low = panel_df['ci_lower'].to_numpy(dtype=float)
    ci_high = panel_df['ci_upper'].to_numpy(dtype=float)
    labels = panel_df['level_display'].astype(str).tolist()
    colors = [level_color_map.get(lbl, '#7f7f7f') for lbl in labels]

    xerr = np.vstack([means - ci_low, ci_high - means])

    ax.barh(
        y_pos,
        means,
        xerr=xerr,
        color=colors,
        edgecolor='black',
        linewidth=1.0,
        capsize=4
    )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=FS_TICKS)

    title_text = 'Age(years)' if panel_name == 'Age group' else panel_name
    ax.set_title(title_text, fontsize=FS_TITLE, fontweight='bold', pad=8)

    ax.set_xlim(*xlim)
    ax.tick_params(axis='x', labelsize=FS_TICKS)
    ax.tick_params(axis='y', labelsize=FS_TICKS)

    for yy, (_, row) in zip(y_pos, panel_df.iterrows()):
        star_txt = f"; {row['stars']}" if row['stars'] != "" else ""
        ann = f"(N = {int(row['n'])}{star_txt})"
        ax.text(
            float(row['ci_upper']) + (xlim[1] - xlim[0]) * 0.01,
            yy,
            ann,
            ha='left',
            va='center',
            fontsize=FS_ANN
        )

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

fig.text(
    0.5, 0.02,
    'Predicted baseline overestimation',
    ha='center',
    fontsize=FS_AXLAB,
    fontweight='bold'
)

fig.text(
    0.92, 0.02,
    '*** p<0.001; ** p<0.01; * p<0.05',
    ha='right',
    fontsize=14,
    bbox=dict(facecolor='white', edgecolor='black', boxstyle='square,pad=0.25')
)

plt.tight_layout(rect=[0.03, 0.06, 0.98, 1.0])

fig.savefig(
    '../output/Figure 2.pdf',
    format='pdf',
    bbox_inches='tight'
)

plt.show()

# =============================================================================
# Figure 3
# purpose:
#   build Figure 3 from the CSV outputs produced by main_analysis.py
# panels:
#   - Panel A: mean absolute revision (MAR) for control vs treatment
#   - Panel B: revision rate for control vs treatment
# output:
#   - ../output/Figure 3.pdf
# =============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

figure3a_df = pd.read_csv('../output/figure3a.csv')
figure3b_df = pd.read_csv('../output/figure3b.csv')

required_cols_a = ['Group', 'MAR']
required_cols_b = ['Group', 'Revision Rate']

missing_a = [c for c in required_cols_a if c not in figure3a_df.columns]
missing_b = [c for c in required_cols_b if c not in figure3b_df.columns]

if missing_a:
    raise ValueError(f"Missing required columns in figure3a.csv: {missing_a}")
if missing_b:
    raise ValueError(f"Missing required columns in figure3b.csv: {missing_b}")

figure3a_df['MAR'] = pd.to_numeric(figure3a_df['MAR'], errors='raise')
figure3b_df['Revision Rate'] = pd.to_numeric(figure3b_df['Revision Rate'], errors='raise')

group_order = ['control', 'treatment']
group_labels = ['Control', 'Treatment']
group_colors = ['#0072B2', '#D55E00']

data_A = [
    figure3a_df.loc[figure3a_df['Group'] == 'control', 'MAR'].dropna(),
    figure3a_df.loc[figure3a_df['Group'] == 'treatment', 'MAR'].dropna()
]

rev_rate_control = figure3b_df.loc[
    figure3b_df['Group'] == 'control', 'Revision Rate'
].dropna().astype(float)

rev_rate_treatment = figure3b_df.loc[
    figure3b_df['Group'] == 'treatment', 'Revision Rate'
].dropna().astype(float)

n_control = int(rev_rate_control.shape[0])
n_treatment = int(rev_rate_treatment.shape[0])

mean_rate_control = float(rev_rate_control.mean())
mean_rate_treatment = float(rev_rate_treatment.mean())

sd_rate_control = float(rev_rate_control.std(ddof=1)) if n_control > 1 else np.nan
sd_rate_treatment = float(rev_rate_treatment.std(ddof=1)) if n_treatment > 1 else np.nan

se_rate_control = sd_rate_control / np.sqrt(n_control) if n_control > 1 else np.nan
se_rate_treatment = sd_rate_treatment / np.sqrt(n_treatment) if n_treatment > 1 else np.nan

tcrit_control = stats.t.ppf(0.975, df=n_control - 1) if n_control > 1 else np.nan
tcrit_treatment = stats.t.ppf(0.975, df=n_treatment - 1) if n_treatment > 1 else np.nan

ci_control = tcrit_control * se_rate_control if n_control > 1 else np.nan
ci_treatment = tcrit_treatment * se_rate_treatment if n_treatment > 1 else np.nan

t_rate, p_rate = stats.ttest_ind(
    rev_rate_control,
    rev_rate_treatment,
    equal_var=False
)

def _stars_from_p(p):
    if pd.isna(p):
        return ""
    p = float(p)
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    if p < 0.10:
        return "†"
    return ""

stars_rate = _stars_from_p(p_rate)

fig_f3, (axA, axB) = plt.subplots(ncols=2, nrows=1, figsize=(12, 5.5), dpi=600)

# ---------------------------
# Panel A: MAR boxplot
# ---------------------------
positions = np.arange(2) + 1

bpA = axA.boxplot(
    data_A,
    positions=positions,
    widths=0.6,
    patch_artist=True,
    showfliers=False,
    medianprops=dict(color='black', linewidth=1.3),
    boxprops=dict(linewidth=1.0, edgecolor='black'),
    whiskerprops=dict(linewidth=1.0, color='black'),
    capprops=dict(linewidth=1.0, color='black'),
)

for box, col in zip(bpA['boxes'], group_colors):
    box.set_facecolor(col)

means_A = [float(data_A[0].mean()), float(data_A[1].mean())]
axA.plot(
    positions,
    means_A,
    linestyle='None',
    marker='D',
    markersize=7,
    markerfacecolor='white',
    markeredgecolor='black',
    zorder=5
)

axA.set_xticks(positions)
axA.set_xticklabels(group_labels, fontsize=18, fontweight='bold')
axA.set_ylabel('MAR (percentage points)', fontsize=17)
axA.tick_params(axis='y', labelsize=15)

axA.legend(
    [
        plt.Line2D([0], [0], color='black', linewidth=1.3),
        plt.Line2D(
            [0], [0],
            marker='D',
            linestyle='None',
            markerfacecolor='white',
            markeredgecolor='black',
            markersize=7,
            color='black'
        ),
    ],
    ['Median', 'Mean'],
    loc='upper left',
    frameon=True,
    fontsize=14
)

axA.text(
    -0.15, 1.02, 'A)',
    transform=axA.transAxes,
    fontsize=18,
    fontweight='bold',
    ha='left',
    va='bottom'
)

# ---------------------------
# Panel B: Revision rate
# ---------------------------
heights = [mean_rate_control, mean_rate_treatment]
yerr = [ci_control, ci_treatment]

axB.bar(
    positions,
    heights,
    yerr=yerr,
    capsize=5,
    width=0.6,
    color=group_colors,
    edgecolor='black',
    linewidth=1.0
)

axB.set_xticks(positions)
axB.set_xticklabels(group_labels, fontsize=18, fontweight='bold')
axB.set_ylabel('Revision Rate (%)', fontsize=17)
axB.tick_params(axis='y', labelsize=15)
axB.set_ylim(0, 100)

label_pad = 3
for x_pos, h, n, e in zip(positions, heights, [n_control, n_treatment], yerr):
    if np.isnan(h):
        continue
    e_use = 0.0 if (e is None or np.isnan(e)) else float(e)
    axB.text(
        x_pos,
        h + e_use + label_pad,
        f"{h:.1f}% (N={n})",
        ha='center',
        va='bottom',
        fontsize=14
    )

if stars_rate != "":
    x1, x2 = positions
    y_max = max(
        heights[0] + (0.0 if np.isnan(yerr[0]) else float(yerr[0])),
        heights[1] + (0.0 if np.isnan(yerr[1]) else float(yerr[1]))
    )
    y_br = y_max + 11.0
    h_br = 2.0

    axB.plot(
        [x1, x1, x2, x2],
        [y_br, y_br + h_br, y_br + h_br, y_br],
        color='black',
        linewidth=1.2
    )

    axB.text(
        (x1 + x2) / 2,
        y_br + h_br + 1.2,
        stars_rate,
        ha='center',
        va='bottom',
        fontsize=18,
        fontweight='bold'
    )

axB.text(
    -0.15, 1.02, 'B)',
    transform=axB.transAxes,
    fontsize=18,
    fontweight='bold',
    ha='left',
    va='bottom'
)

plt.tight_layout(w_pad=2.5)

fig_f3.savefig(
    '../output/Figure 3.pdf',
    format='pdf',
    bbox_inches='tight'
)

plt.show()

# =============================================================================
# Figure 4
# purpose:
#   build Figure 4 from the CSV outputs produced by main_analysis.py
# panels:
#   - Panel A: revision rate by condition
#   - Panel B: mean absolute revision (MAR) by condition
#   - Panel C: influence score by condition
# inputs:
#   - ../output/figure4a.csv
#   - ../output/figure4b.csv
#   - ../output/figure4c.csv
# output:
#   - ../output/Figure 4.pdf
# note:
#   - Panel C keeps the control row as an empty row for layout
# =============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------
# 0. Load figure inputs
# ---------------------------
fig4a_df = pd.read_csv('../output/figure4a.csv')
fig4b_df = pd.read_csv('../output/figure4b.csv')
fig4c_df = pd.read_csv('../output/figure4c.csv')

# ---------------------------
# 1. Condition order and colours
# ---------------------------
cond_order = ["Age", "Conformity", "Prestige", "Season", "Individual"]

cond_colors = {
    "Individual": "#7f7f7f",
    "Age":        "#1f77b4",
    "Conformity": "#2ca02c",
    "Prestige":   "#ff7f0e",
    "Season":     "#9467bd",
}

label_map = {"Individual": "Control"}

def disp(lbl: str) -> str:
    return label_map.get(lbl, lbl)

# ---------------------------
# 2. Font sizes (consistent with original figure style)
# ---------------------------
FS_XLAB = 18
FS_TICKS = 18
FS_TAG = 18
FS_BAR_TEXT_A = 13
FS_BAR_TEXT_C = 17
FS_YTICK_A = FS_TICKS - 2

# ---------------------------
# 3. Clean and reorder input tables
# ---------------------------
for df_ in [fig4a_df, fig4b_df, fig4c_df]:
    df_['Condition'] = pd.Categorical(df_['Condition'], categories=cond_order, ordered=True)

for col in ['Revision Rate', 'N', 'lower', 'upper']:
    if col in fig4a_df.columns:
        fig4a_df[col] = pd.to_numeric(fig4a_df[col], errors='coerce')

for col in ['MAR', 'N', 'lower', 'upper', 't_statistic', 'p_raw', 'p_bh_fdr']:
    if col in fig4b_df.columns:
        fig4b_df[col] = pd.to_numeric(fig4b_df[col], errors='coerce')

for col in ['Influence Score', 'N', 'lower', 'upper']:
    if col in fig4c_df.columns:
        fig4c_df[col] = pd.to_numeric(fig4c_df[col], errors='coerce')

fig4a_df = fig4a_df.sort_values('Condition').reset_index(drop=True)
fig4b_df = fig4b_df.sort_values('Condition').reset_index(drop=True)
fig4c_df = fig4c_df.sort_values('Condition').reset_index(drop=True)

# ---------------------------
# 4. Panel A data: Revision Rate
# ---------------------------
prob_vals = np.array([
    fig4a_df.loc[fig4a_df['Condition'] == cond, 'Revision Rate'].iloc[0]
    for cond in cond_order
], dtype=float)

N_by_cond = {
    cond: int(fig4a_df.loc[fig4a_df['Condition'] == cond, 'N'].iloc[0])
    for cond in cond_order
}

colors_prob = [cond_colors[c] for c in cond_order]

# ---------------------------
# 5. Panel B data: MAR
# ---------------------------
mar_vals = np.array([
    fig4b_df.loc[fig4b_df['Condition'] == cond, 'MAR'].iloc[0]
    for cond in cond_order
], dtype=float)

ci_low_b = np.array([
    fig4b_df.loc[fig4b_df['Condition'] == cond, 'lower'].iloc[0]
    for cond in cond_order
], dtype=float)

ci_hi_b = np.array([
    fig4b_df.loc[fig4b_df['Condition'] == cond, 'upper'].iloc[0]
    for cond in cond_order
], dtype=float)

xerr_b = np.vstack([mar_vals - ci_low_b, ci_hi_b - mar_vals])
colors_b = [cond_colors[c] for c in cond_order]

# Welch-vs-control BH-FDR stars are stored only for the four social conditions
stars_b_dict = {}
if 'stars_bh_fdr' in fig4b_df.columns:
    for _, row in fig4b_df.iterrows():
        if pd.notna(row['stars_bh_fdr']):
            stars_b_dict[str(row['Condition'])] = str(row['stars_bh_fdr'])

# ---------------------------
# 6. Panel C data: Influence Score
# ---------------------------
inf_vals = np.array([
    fig4c_df.loc[fig4c_df['Condition'] == cond, 'Influence Score'].iloc[0]
    for cond in cond_order
], dtype=float)

ci_low_c = np.array([
    fig4c_df.loc[fig4c_df['Condition'] == cond, 'lower'].iloc[0]
    for cond in cond_order
], dtype=float)

ci_hi_c = np.array([
    fig4c_df.loc[fig4c_df['Condition'] == cond, 'upper'].iloc[0]
    for cond in cond_order
], dtype=float)

plot_vals_c = np.where(np.isnan(inf_vals), 0.0, inf_vals)
xerr_c = np.where(
    np.isnan(inf_vals),
    0.0,
    np.vstack([inf_vals - ci_low_c, ci_hi_c - inf_vals])
)

colors_c = [cond_colors[c] for c in cond_order]

# ---------------------------
# 7. Build figure
# ---------------------------
fig, (axA, axB, axC) = plt.subplots(ncols=3, figsize=(17, 5), dpi=300)
y_pos = np.arange(len(cond_order))

# ---- Panel A: Revision Rate ----
axA.barh(y_pos, prob_vals, color=colors_prob, edgecolor="black", linewidth=1.0)
axA.set_yticks(y_pos)
axA.set_yticklabels([disp(c) for c in cond_order], fontsize=FS_YTICK_A)

for t in axA.get_yticklabels():
    t.set_fontweight("bold")

axA.set_xlabel("Revision Rate (%)", fontsize=FS_XLAB, fontweight='bold')
axA.set_xlim(0, 100)

for y, cond, val in zip(y_pos, cond_order, prob_vals):
    if np.isnan(val):
        continue
    axA.text(
        val + 2,
        y,
        f"{val:.1f}% (N={N_by_cond[cond]})",
        ha="left",
        va="center",
        fontsize=FS_BAR_TEXT_A
    )

axA.text(
    -0.15, 1.03, "A)",
    transform=axA.transAxes,
    fontsize=FS_TAG,
    fontweight="bold"
)

# ---- Panel B: MAR ----
axB.barh(
    y_pos,
    mar_vals,
    xerr=xerr_b,
    color=colors_b,
    edgecolor="black",
    linewidth=1.0,
    capsize=4
)
axB.set_yticks(y_pos)
axB.set_yticklabels([])
axB.set_xlabel("MAR (percentage points)", fontsize=FS_XLAB, fontweight='bold')

max_x_b = np.nanmax(ci_hi_b)
axB.set_xlim(0, max_x_b * 1.35)

y_ind = cond_order.index("Individual")
bracket_order = ["Season", "Prestige", "Conformity", "Age"]
base_x = max_x_b * 1.05
step_x = max_x_b * 0.075
tick = max_x_b * 0.02

for i, cond in enumerate(bracket_order):
    stars_txt = stars_b_dict.get(cond, "")
    if stars_txt == "":
        continue

    y_soc = cond_order.index(cond)
    xb = min(base_x + step_x * i, max_x_b * 1.30)

    axB.plot([xb, xb], [y_soc, y_ind], color="black", linewidth=0.8)
    axB.plot([xb - tick, xb], [y_soc, y_soc], color="black", linewidth=0.8)
    axB.plot([xb - tick, xb], [y_ind, y_ind], color="black", linewidth=0.8)

    axB.text(
        xb + tick * 1.8,
        (y_soc + y_ind) / 2,
        "\n".join(list(stars_txt)),
        ha="center",
        va="center",
        fontsize=FS_TAG,
        fontweight="bold",
        linespacing=0.9
    )

axB.text(
    -0.15, 1.03, "B)",
    transform=axB.transAxes,
    fontsize=FS_TAG,
    fontweight="bold"
)

# ---- Panel C: Influence Score ----
bars = axC.barh(
    y_pos,
    plot_vals_c,
    xerr=xerr_c,
    color=colors_c,
    edgecolor="black",
    linewidth=1.0,
    capsize=4
)

# make the control row visually empty
bars[cond_order.index("Individual")].set_alpha(0)

axC.set_yticks(y_pos)
axC.set_yticklabels([])
axC.set_xlabel("Influence score", fontsize=FS_XLAB, fontweight='bold')

max_x_c = np.nanmax(ci_hi_c)
axC.set_xlim(0, max_x_c * 1.45)

for y, val in zip(y_pos, inf_vals):
    if np.isnan(val):
        continue
    axC.text(
        val + max_x_c * 0.18,
        y,
        f"{val:.2f}",
        ha="left",
        va="center",
        fontsize=FS_BAR_TEXT_C
    )

axC.text(
    -0.15, 1.03, "C)",
    transform=axC.transAxes,
    fontsize=FS_TAG,
    fontweight="bold"
)

# ---- Ticks ----
for ax in (axA, axB, axC):
    ax.tick_params(axis="both", labelsize=FS_TICKS)

plt.tight_layout(w_pad=3)

fig.savefig(
    '../output/Figure 4.pdf',
    format='pdf',
    bbox_inches='tight'
)

plt.show()

# =============================================================================
# Figure 5
# purpose:
#   build Figure 5 from the CSV output produced by main_analysis.py
# content:
#   - density plots of initial and final estimation error
#     within each condition
# inputs:
#   - ../output/figure5.csv
# output:
#   - ../output/Figure 5.pdf
# =============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

# ---------------------------
# 0. Load figure input
# ---------------------------
figure5_df = pd.read_csv('../output/figure5.csv')

for col in ['Initial Error', 'Final Error', 'Estimation Error Change']:
    if col in figure5_df.columns:
        figure5_df[col] = pd.to_numeric(figure5_df[col], errors='coerce')

# ---------------------------
# 1. Condition order and labels
# ---------------------------
conditions_order = ["Individual", "Age", "Conformity", "Prestige", "Season"]

label_map = {"Individual": "Control"}

def disp(lbl: str) -> str:
    return label_map.get(lbl, lbl)

# ---------------------------
# 2. Compute condition-level paired error arrays and paired effect size (dz)
#    note:
#      - dz is based on (Final Error - Initial Error), so negative values
#        indicate improvement in estimation accuracy
# ---------------------------
cond_errors_before = {}
cond_errors_after = {}
cond_effect_dz = {}

for cond in conditions_order:
    sub = figure5_df[figure5_df["Condition"].eq(cond)].copy()

    b = pd.to_numeric(sub["Initial Error"], errors="coerce").dropna().to_numpy(dtype=float)
    a = pd.to_numeric(sub["Final Error"], errors="coerce").dropna().to_numpy(dtype=float)

    # align in case of any unexpected missingness pattern
    n_pair = min(len(b), len(a))
    b = b[:n_pair]
    a = a[:n_pair]

    cond_errors_before[cond] = b
    cond_errors_after[cond] = a

    if n_pair > 1:
        diff = a - b
        sd_diff = diff.std(ddof=1)
        dz = float(diff.mean() / sd_diff) if sd_diff > 0 else np.nan
    else:
        dz = np.nan

    cond_effect_dz[cond] = dz

# ---------------------------
# 3. Global axis limits and KDEs
# ---------------------------
all_before = np.concatenate([v for v in cond_errors_before.values() if v.size > 0])
all_after = np.concatenate([v for v in cond_errors_after.values() if v.size > 0])

y_min = 0.0
y_max = float(np.nanmax(np.concatenate([all_before, all_after]))) if all_before.size else 0.0
y_max = np.ceil(y_max / 5.0) * 5.0

y_grid = np.linspace(y_min, y_max, 400)

kde_before = {}
kde_after = {}
max_density = 0.0

for cond in conditions_order:
    eb = cond_errors_before[cond]
    ea = cond_errors_after[cond]

    if eb.size > 1:
        kde_b = gaussian_kde(eb, bw_method="scott")
        kde_before[cond] = kde_b
        max_density = max(max_density, kde_b(y_grid).max())
    else:
        kde_before[cond] = None

    if ea.size > 1:
        kde_a = gaussian_kde(ea, bw_method="scott")
        kde_after[cond] = kde_a
        max_density = max(max_density, kde_a(y_grid).max())
    else:
        kde_after[cond] = None

x_lim = float(max_density * 1.05) if max_density > 0 else 0.05
x_min, x_max = -x_lim, x_lim

# ---------------------------
# 4. Build figure
# ---------------------------
fig, axes = plt.subplots(
    ncols=len(conditions_order),
    nrows=1,
    figsize=(14, 5),
    dpi=300,
    sharey=False,
    sharex=True
)

before_line = "#8C4C23"
before_fill = "#C49A6C"
after_line = "#F4A259"
after_fill = "#F9C892"

# ---------------------------
# 5. Font sizes
# ---------------------------
FS_TITLE = 14
FS_D = 12
FS_TICKS = 13.5
FS_AXLAB = 13
FS_LEG = 10

TAG_Y_TITLE = 1.10
TAG_Y_D = 1.085

# ---------------------------
# 6. Plot each condition
# ---------------------------
for ax, cond in zip(axes, conditions_order):
    kde_b = kde_before.get(cond)
    kde_a = kde_after.get(cond)

    ax.axvline(0.0, color="black", linewidth=1.0)

    # Initial Error (left side)
    if kde_b is not None:
        dens_b = kde_b(y_grid)
        ax.fill_betweenx(y_grid, 0, -dens_b, color=before_fill, alpha=0.7)
        ax.plot(-dens_b, y_grid, color=before_line, linewidth=2)
        ax.hlines(
            cond_errors_before[cond].mean(),
            x_min, 0.0,
            colors=before_line,
            linestyles="--",
            linewidth=1.5
        )

    # Final Error (right side)
    if kde_a is not None:
        dens_a = kde_a(y_grid)
        ax.fill_betweenx(y_grid, 0, dens_a, color=after_fill, alpha=0.7)
        ax.plot(dens_a, y_grid, color=after_line, linewidth=2)
        ax.hlines(
            cond_errors_after[cond].mean(),
            0.0, x_max,
            colors=after_line,
            linestyles="--",
            linewidth=1.5
        )

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    ax.yaxis.tick_right()
    ax.spines["top"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["right"].set_visible(True)

    ax.set_xticks([x_min, 0.0, x_max])
    ax.set_xticklabels([f"{abs(x_min):.2f}", "0.00", f"{abs(x_max):.2f}"])

    y_ticks = np.linspace(0, y_max, 6)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([f"{yt:.0f}" for yt in y_ticks])

    ax.tick_params(axis="both", labelsize=FS_TICKS)

    dz = cond_effect_dz.get(cond, np.nan)
    dz_txt = "" if np.isnan(dz) else f"(d = {dz:.2f})"

    ax.text(
        0.5, TAG_Y_TITLE, disp(cond),
        transform=ax.transAxes,
        fontsize=FS_TITLE,
        fontweight="bold",
        ha="center",
        va="bottom"
    )
    ax.text(
        0.5, TAG_Y_D, dz_txt,
        transform=ax.transAxes,
        fontsize=FS_D,
        ha="center",
        va="top"
    )

# ---------------------------
# 7. Shared axis labels
# ---------------------------
fig.text(
    0.45, 0.03, "Density",
    ha="center",
    fontsize=FS_AXLAB,
    fontweight="bold"
)

y_label_center = axes[-1].get_position().y0 + axes[-1].get_position().height / 2 - 0.01

fig.text(
    0.98, y_label_center, "Estimation error (percentage points)",
    va="center",
    rotation="vertical",
    fontsize=11,
    fontweight="bold"
)

# ---------------------------
# 8. Legend
# ---------------------------
legend_handles = [
    plt.Line2D([0], [0], color=before_line, linestyle="--", linewidth=2),
    plt.Line2D([0], [0], color=after_line, linestyle="--", linewidth=2),
]
legend_labels = ["Baseline Error", "Post-intervention Error"]

fig.legend(
    legend_handles, legend_labels,
    loc="lower center",
    bbox_to_anchor=(0.63, 0),
    ncol=2,
    frameon=True,
    fontsize=FS_LEG,
    handlelength=2.5,
    columnspacing=1.4
)

plt.tight_layout(rect=[0.06, 0.08, 0.98, 0.93])

fig.savefig(
    '../output/Figure 5.pdf',
    format='pdf',
    bbox_inches='tight'
)

plt.show()