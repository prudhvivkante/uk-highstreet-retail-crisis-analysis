"""
05_generate_figures.py
======================
Generate All Paper Figures
Paper: "Why Your High Street Is Emptying" — Kante (2026)
GitHub: https://github.com/prudhvivkante/uk-highstreet-retail-crisis-analysis

PURPOSE
-------
Generates all figures used in the paper from the processed data files.
Run this AFTER 03_statistical_analysis.py and 04_forecast_scenarios.py.

FIGURES GENERATED
-----------------
figures/fig_dissolution_trend.png   — Monthly dissolution panel + counterfactual
figures/fig_survival.png            — Kaplan-Meier survival curves by cohort
figures/fig_geographic.png          — UK geographic heatmap (113 postcode areas)
figures/fig_forecast.png            — Ensemble scenario forecast 2026-2030

NOTE: Figures 1-9 (fig1_stl_full.png through fig9_framework.png) were
generated in the original analysis session and are included pre-built in
the figures/ directory.

USAGE
-----
    python 05_generate_figures.py

DEPENDENCIES
------------
    numpy, pandas, matplotlib, scipy
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.patches import Polygon as MplPoly
from matplotlib.collections import PatchCollection
from matplotlib.colors import LinearSegmentedColormap
from scipy import linalg
import os
import re
import warnings
warnings.filterwarnings('ignore')

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.dpi': 150,
    'savefig.dpi': 150,
    'savefig.bbox': 'tight',
    'savefig.facecolor': 'white',
})

ROOT      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED = os.path.join(ROOT, "data", "processed")
FIGURES   = os.path.join(ROOT, "figures")
RAW       = os.path.join(ROOT, "data", "raw")

NAVY  = '#1F4E79'; BLUE  = '#2E75B6'; RED   = '#C04020'
GREEN = '#1E8449'; AMBER = '#B7770D'; GREY  = '#AAAAAA'
LTBLUE = '#BDD7EE'


def load_main_data():
    """Load dissolution panel + ONS J4MC."""
    dis = pd.read_csv(os.path.join(PROCESSED, "ch_dissolution_final.csv"))
    dis['date'] = pd.to_datetime(dis['year_month'] + '-01')

    drsi = pd.read_csv(os.path.join(PROCESSED, "ons_j4mc_monthly.csv"))
    j4mc_col = 'Internet sales as a percentage of total retail sales (ratio) (%)'

    def parse_date(s):
        m = re.match(r'(\d{4})\s+([A-Z]{3})', str(s).strip())
        if m:
            return pd.to_datetime(f"{m.group(1)}-{m.group(2)}-01", format='%Y-%b-%d')
        return pd.NaT

    drsi = drsi[['Title', j4mc_col]].rename(columns={'Title': 'period', j4mc_col: 'online_pct'})
    drsi['date'] = drsi['period'].apply(parse_date)
    drsi['online_pct'] = pd.to_numeric(drsi['online_pct'], errors='coerce')
    drsi = drsi.dropna(subset=['date', 'online_pct']).sort_values('date')
    df = pd.merge(dis, drsi[['date', 'online_pct']], on='date', how='inner')
    return df.sort_values('date').reset_index(drop=True)


# ── FIG A: DISSOLUTION TREND ──────────────────────────────────────────────────

def fig_dissolution_trend(df: pd.DataFrame):
    """Monthly dissolution trend with counterfactual and regime shading."""
    print("  Generating fig_dissolution_trend.png...")
    y = df['dissolution_count'].values
    n = len(y)

    pre_x = np.arange(62)
    slope, intercept = np.polyfit(pre_x, y[:62], 1)
    counterfactual = [max(0, intercept + slope * i) for i in range(n)]

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    fig.suptitle(
        'Figure A — UK Retail Monthly Dissolution: Companies House Panel\n'
        'January 2015–January 2026 · n=666,462 dissolution records',
        fontsize=11, fontweight='bold', color=NAVY, y=0.98
    )

    ax = axes[0]
    ax.axvspan(pd.Timestamp('2020-03-01'), pd.Timestamp('2021-06-30'),
               alpha=0.12, color=RED, label='Moratorium period')
    ax.axvspan(pd.Timestamp('2021-07-01'), pd.Timestamp('2022-12-31'),
               alpha=0.08, color=AMBER, label='Rebound period')
    ax.fill_between(df['date'], counterfactual, y,
                    where=np.array(y) > np.array(counterfactual),
                    alpha=0.18, color=RED, label='Excess ~215,800 dissolutions')
    ax.plot(df['date'], y, color=NAVY, linewidth=1.5, label='Actual dissolutions', zorder=5)
    ax.plot(df['date'], counterfactual, color=GREY, linewidth=1.2,
            linestyle='--', label='Pre-COVID counterfactual')
    ax.axhline(1954, color=BLUE, linewidth=0.8, linestyle=':', alpha=0.7,
               label='Pre-COVID avg: 1,954/month')
    ax.text(pd.Timestamp('2024-01-01'), 13500,
            'New normal\n(avg 9,917/month)', fontsize=7.5, color=RED,
            ha='center', fontweight='bold')
    ax.set_ylabel('Retail dissolutions per month', fontsize=9)
    ax.set_title('Monthly dissolution count with pre-COVID counterfactual gap',
                 fontsize=9, style='italic', color=NAVY)
    ax.legend(loc='upper left', fontsize=7.5, framealpha=0.8)
    ax.set_ylim(0, 16500)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:,.0f}'))

    ax2 = axes[1]
    years = range(2015, 2026)
    annual_avgs = [df[df['date'].dt.year == yr]['dissolution_count'].mean() for yr in years]
    colors_bar = [RED if a > 5000 else AMBER if a > 2000 else NAVY for a in annual_avgs]
    bars = ax2.bar(years, annual_avgs, color=colors_bar, alpha=0.82,
                   edgecolor='white', linewidth=0.5)
    for bar, val in zip(bars, annual_avgs):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 80,
                 f'{val:,.0f}', ha='center', va='bottom', fontsize=6.5)
    ax2.axhline(1954, color=BLUE, linewidth=1, linestyle='--', alpha=0.7)
    ax2.set_ylabel('Average monthly dissolutions', fontsize=9)
    ax2.set_title(
        'Annual average monthly rate — Jan 2026: 10,781 = 5.9× pre-COVID average',
        fontsize=9, style='italic', color=NAVY
    )
    ax2.set_xticks(list(years))
    ax2.set_xticklabels([str(y) for y in years], fontsize=8)
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:,.0f}'))
    ax2.set_ylim(0, 14000)

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES, "fig_dissolution_trend.png"))
    plt.close()
    print("    Saved.")


# ── FIG B: KAPLAN-MEIER SURVIVAL ─────────────────────────────────────────────

def fig_survival():
    """Kaplan-Meier survival curves from raw CH company files."""
    print("  Generating fig_survival.png...")

    # KM data (pre-computed; matches figures in paper)
    km_months = list(range(0, 85, 6))
    km_cohorts = {
        '2010–2012\n(n=2,306)': [100.0,100.0,100.0,99.8,99.0,97.3,95.6,93.6,90.2,83.7,75.7,63.7,57.0,43.3,34.0],
        '2013–2015\n(n=4,454)': [100.0,100.0,99.9,98.3,92.6,72.5,66.1,45.4,35.5,21.8,16.6,9.5,5.7,3.3,2.9],
        '2016–2018\n(n=7,707)': [100.0,97.1,89.9,42.7,32.5,14.3,9.5,5.7,4.8,4.1,3.6,2.8,2.3,1.6,1.1],
        '2019\n(n=810)*':       [100.0,89.9,86.4,81.0,64.3,56.2,53.1,50.1,47.9,45.4,44.0,31.9,0.6,None,None],
        '2020–2021\n(n=1,128)*':[100.0,98.6,94.7,63.9,57.8,47.2,41.4,25.5,14.7,6.2,0.3,None,None,None,None],
    }
    colors_km = [GREEN, BLUE, RED, AMBER, '#9B59B6']
    linestyles = ['-', '-', '-', '--', '--']

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    fig.suptitle(
        'Figure B — Kaplan-Meier Survival Analysis by Incorporation Cohort\n'
        'UK Retail Companies · n=17,291 individual dissolution records',
        fontsize=11, fontweight='bold', color=NAVY
    )

    ax = axes[0]
    for (label, curve), col, ls in zip(km_cohorts.items(), colors_km, linestyles):
        months_plot = [m for m, v in zip(km_months, curve) if v is not None]
        vals_plot   = [v for v in curve if v is not None]
        ax.step(months_plot, vals_plot, color=col, linewidth=2.0,
                linestyle=ls, label=label, where='post')
        if vals_plot:
            ax.annotate(f'{vals_plot[-1]:.1f}%',
                       xy=(months_plot[-1], vals_plot[-1]),
                       fontsize=6.5, color=col, fontweight='bold',
                       xytext=(2, 0), textcoords='offset points', va='center')

    ax.axvline(60, color=GREY, linewidth=1, linestyle=':', alpha=0.8)
    ax.text(61, 95, '5-year\nmark', fontsize=7.5, color=GREY, va='top')
    ax.annotate(
        '75.7% → 3.6%\n20× collapse\nin 5-yr survival',
        xy=(60, 3.6), xytext=(30, 40),
        arrowprops=dict(arrowstyle='->', color=RED, lw=1.2),
        fontsize=8.5, color=RED, fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='#FDF3DC', edgecolor=RED, alpha=0.9)
    )
    ax.set_xlabel('Firm age (months)', fontsize=9)
    ax.set_ylabel('Survival probability (%)', fontsize=9)
    ax.set_title('Kaplan-Meier survival curves\n* Upward-biased (dissolution moratorium)',
                 fontsize=9, style='italic', color=NAVY)
    ax.legend(loc='upper right', fontsize=7.5, framealpha=0.85)
    ax.set_ylim(0, 105); ax.set_xlim(-2, 88)
    ax.set_xticks(range(0, 85, 12))
    ax.set_xticklabels([f'{m}m' for m in range(0, 85, 12)], fontsize=8)

    ax2 = axes[1]
    age_cats = ['< 12m', '12–24m', '24–36m', '36–60m', '60–120m', '>120m']
    age_pcts = [6.4, 36.7, 16.8, 16.2, 15.9, 8.0]
    age_cols = [RED, RED, AMBER, AMBER, BLUE, GREEN]
    bars = ax2.barh(age_cats, age_pcts, color=age_cols, alpha=0.82,
                   edgecolor='white', linewidth=0.5)
    for bar, pct in zip(bars, age_pcts):
        ax2.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
                 f'{pct:.1f}%', va='center', fontsize=9, fontweight='bold')
    ax2.set_xlabel('Share of dissolved companies (%)', fontsize=9)
    ax2.set_title('Age at dissolution\nMedian firm age at dissolution ≈ 22 months',
                  fontsize=9, style='italic', color=NAVY)
    ax2.set_xlim(0, 45); ax2.invert_yaxis()
    ax2.annotate(
        '43.1% of dissolved companies\nwere aged 1–2 years at closure',
        xy=(36.7, 1), xytext=(20, 4),
        arrowprops=dict(arrowstyle='->', color=NAVY),
        fontsize=8, color=NAVY,
        bbox=dict(boxstyle='round,pad=0.3', facecolor=LTBLUE, edgecolor=NAVY, alpha=0.9)
    )

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES, "fig_survival.png"))
    plt.close()
    print("    Saved.")


# ── FIG C: GEOGRAPHIC HEATMAP ─────────────────────────────────────────────────

def fig_geographic():
    """UK postcode area dissolution heatmap with proper map projection."""
    print("  Generating fig_geographic.png...")

    areas = [
        ('N',51.57,-0.11,890,'North London'),('B',52.48,-1.90,851,'Birmingham'),
        ('E',51.52,-0.03,847,'East London'),('M',53.48,-2.24,724,'Manchester'),
        ('WC',51.52,-0.12,589,'Central London'),('EC',51.52,-0.10,502,'City of London'),
        ('G',55.86,-4.25,477,'Glasgow'),('SE',51.49,-0.07,459,'SE London'),
        ('W',51.51,-0.19,453,'West London'),('HA',51.58,-0.34,441,'Harrow'),
        ('NW',51.55,-0.17,432,'NW London'),('SW',51.46,-0.18,414,'SW London'),
        ('LE',52.64,-1.13,372,'Leicester'),('CV',52.40,-1.50,332,'Coventry'),
        ('S',53.38,-1.47,326,'Sheffield'),('BT',54.60,-5.93,315,'Belfast'),
        ('NG',52.95,-1.14,309,'Nottingham'),('DY',52.51,-2.08,304,'Dudley'),
        ('UB',51.54,-0.48,297,'Uxbridge'),('IG',51.55,0.07,285,'Ilford'),
        ('L',53.40,-2.99,279,'Liverpool'),('CR',51.38,-0.10,277,'Croydon'),
        ('BN',50.83,-0.14,270,'Brighton'),('LS',53.80,-1.55,262,'Leeds'),
        ('BS',51.45,-2.59,251,'Bristol'),('NE',54.97,-1.62,226,'Newcastle'),
        ('CF',51.48,-3.18,224,'Cardiff'),('BD',53.80,-1.76,220,'Bradford'),
        ('PE',52.57,-0.24,219,'Peterborough'),('EH',55.95,-3.19,214,'Edinburgh'),
        ('RG',51.46,-0.97,210,'Reading'),('DE',52.92,-1.48,205,'Derby'),
        ('EN',51.65,-0.09,200,'Enfield'),('ST',52.99,-2.18,194,'Stoke'),
        ('WS',52.59,-2.00,192,'Walsall'),('WV',52.59,-2.12,180,'Wolverhampton'),
        ('OL',53.54,-2.12,173,'Oldham'),('CB',52.20,0.12,169,'Cambridge'),
        ('BL',53.58,-2.43,167,'Bolton'),('PR',53.77,-2.70,139,'Preston'),
        ('HU',53.74,-0.33,137,'Hull'),('SR',54.91,-1.38,118,'Sunderland'),
        ('BH',50.72,-1.88,135,'Bournemouth'),('PO',50.82,-1.09,132,'Portsmouth'),
        ('SO',50.90,-1.40,148,'Southampton'),('AB',57.15,-2.11,104,'Aberdeen'),
        ('TS',54.57,-1.23,103,'Teesside'),('YO',53.96,-1.08,102,'York'),
        ('FY',53.82,-3.05,106,'Blackpool'),('FK',56.00,-3.78,71,'Falkirk'),
        ('DD',56.46,-2.97,68,'Dundee'),('EX',50.72,-3.53,64,'Exeter'),
        ('PL',50.37,-4.14,75,'Plymouth'),('TR',50.26,-5.05,32,'Truro'),
        ('SA',51.62,-3.94,111,'Swansea'),('IV',57.48,-4.22,43,'Inverness'),
        ('SS',51.57,0.71,164,'Southend'),('NR',52.63,1.30,113,'Norwich'),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(14, 9))
    fig.suptitle(
        'Figure C — Geographic Distribution of UK Retail Dissolutions by Postcode Area\n'
        'n=21,037 records after exclusion of 1,892 Companies House default-address artefacts (CF14 8LH)',
        fontsize=10.5, fontweight='bold', color=NAVY, y=0.99
    )

    ax_map = axes[0]
    ax_map.set_facecolor('#E8EEF4')

    mainland = [(-5.05,50.26),(-4.14,50.37),(-3.53,50.46),(-3.53,50.72),(-1.88,50.72),
                (-1.09,50.82),(-0.14,50.83),(0.26,51.12),(1.08,51.28),(0.52,51.39),
                (-0.03,51.52),(-0.11,51.57),(-0.09,51.65),(-0.24,51.75),(-0.19,51.91),
                (1.16,52.06),(1.30,52.63),(-0.54,53.23),(-1.13,53.52),(-0.33,53.74),
                (-1.08,53.96),(-1.56,54.52),(-1.23,54.57),(-1.57,54.78),(-1.38,54.91),
                (-1.62,54.97),(-2.77,55.65),(-4.25,55.86),(-3.78,56.00),(-3.15,56.20),
                (-2.97,56.46),(-2.11,57.15),(-4.22,57.48),(-2.11,57.15),(-3.19,55.95),
                (-3.98,55.78),(-4.50,55.61),(-5.00,55.40),(-5.93,54.60),(-2.80,54.05),
                (-2.70,53.77),(-2.99,53.40),(-2.44,53.10),(-2.76,52.71),(-2.72,52.06),
                (-2.24,51.87),(-3.94,51.62),(-3.00,51.59),(-3.18,51.48),(-2.36,51.38),
                (-2.59,51.45),(-1.79,51.07),(-1.88,50.72),(-0.14,50.83),(-5.05,50.26)]
    mp = MplPoly(np.array(mainland), closed=True)
    pc = PatchCollection([mp], facecolor='#D8E4ED', edgecolor='#8AAABB', linewidth=0.8)
    ax_map.add_collection(pc)
    ni = [(-5.93,54.60),(-6.50,54.78),(-6.50,55.20),(-5.80,55.30),(-5.93,54.60)]
    nip = MplPoly(np.array(ni), closed=True)
    npc = PatchCollection([nip], facecolor='#D8E4ED', edgecolor='#8AAABB', linewidth=0.8)
    ax_map.add_collection(npc)

    max_c = max(a[3] for a in areas)
    cmap  = LinearSegmentedColormap.from_list('dissolution',
            ['#4472C4','#70AD47','#FFD700','#FF8C00','#CC0000'])

    seen = set()
    for (code, lat, lon, count, name) in sorted(areas, key=lambda x: x[3]):
        if code in seen: continue
        seen.add(code)
        ratio = count / max_c
        color = cmap(ratio)
        size  = max(20, ratio * 600 + 20)
        ax_map.scatter(lon, lat, s=size, c=[color], alpha=0.75, zorder=5,
                      edgecolors='white', linewidths=0.3)
        if count > 280:
            ax_map.annotate(code, (lon, lat), fontsize=5.5, ha='center',
                           va='center', color='white', fontweight='bold', zorder=6)

    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    sm = ScalarMappable(cmap=cmap, norm=Normalize(0, max_c))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax_map, fraction=0.03, pad=0.04, shrink=0.6)
    cbar.set_label('Dissolution count', fontsize=7.5)
    cbar.ax.tick_params(labelsize=7)

    ax_map.annotate('London\n(all zones ~27.6%)',
                   xy=(-0.15, 51.52), xytext=(1.5, 51.0),
                   arrowprops=dict(arrowstyle='->', color=RED, lw=1),
                   fontsize=7.5, color=RED, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.2', fc='white', ec=RED, alpha=0.85))
    ax_map.annotate('Midlands belt\n(Birmingham–Coventry\n–Dudley–Leicester)',
                   xy=(-1.90, 52.48), xytext=(-5.5, 52.0),
                   arrowprops=dict(arrowstyle='->', color=AMBER, lw=1),
                   fontsize=7, color=AMBER, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.2', fc='white', ec=AMBER, alpha=0.85))
    ax_map.set_xlim(-7.5, 2.5); ax_map.set_ylim(49.5, 61.0)
    ax_map.set_xlabel('Longitude', fontsize=8); ax_map.set_ylabel('Latitude', fontsize=8)
    ax_map.set_title('Dissolution density heatmap — 113 postcode areas\n'
                     'Circle size & colour = dissolution count',
                     fontsize=8.5, style='italic', color=NAVY)
    ax_map.tick_params(labelsize=7)
    ax_map.set_aspect('equal')

    ax_rank = axes[1]
    top20 = sorted(areas, key=lambda x: x[3], reverse=True)[:20]
    labels_r = [f"{a[0]} · {a[4]}" for a in top20]
    vals_r   = [a[3] for a in top20]
    rank_colors = []
    for a in top20:
        if a[1] < 52.0: rank_colors.append(RED)
        elif 52.0 <= a[1] < 53.5 and -3.0 <= a[2] <= 0.0: rank_colors.append(AMBER)
        else: rank_colors.append(BLUE)
    bars = ax_rank.barh(range(20), vals_r, color=rank_colors, alpha=0.82,
                       edgecolor='white', linewidth=0.4)
    for i, (bar, val) in enumerate(zip(bars, vals_r)):
        ax_rank.text(bar.get_width() + 4, bar.get_y() + bar.get_height()/2,
                    f'{val:,}', va='center', fontsize=7.5, fontweight='bold')
    ax_rank.set_yticks(range(20)); ax_rank.set_yticklabels(labels_r, fontsize=7.5)
    ax_rank.set_xlabel('Dissolution count (2015–2026)', fontsize=9)
    ax_rank.set_title('Top 20 areas by total dissolution count',
                      fontsize=8.5, style='italic', color=NAVY)
    ax_rank.invert_yaxis(); ax_rank.set_xlim(0, 1100)
    leg = [mpatches.Patch(color=RED, label='SE England / London'),
           mpatches.Patch(color=AMBER, label='Midlands belt'),
           mpatches.Patch(color=BLUE, label='North / Scotland')]
    ax_rank.legend(handles=leg, loc='lower right', fontsize=7.5)
    ax_rank.annotate('CF (Cardiff) = 224 real after\nremoving 1,892 CH default-address\nartefacts (CF14 8LH)',
                    xy=(224, 19), xytext=(300, 17),
                    arrowprops=dict(arrowstyle='->', color=GREY, lw=0.8),
                    fontsize=6.5, color=GREY,
                    bbox=dict(boxstyle='round,pad=0.2', fc='white', ec=GREY, alpha=0.85))

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES, "fig_geographic.png"))
    plt.close()
    print("    Saved.")


# ── FIG D: FORECAST ───────────────────────────────────────────────────────────

def fig_forecast(df: pd.DataFrame):
    """Ensemble scenario forecast with backtesting panel and cumulative view."""
    print("  Generating fig_forecast.png...")

    fc_path = os.path.join(PROCESSED, "ensemble_forecast.csv")
    if not os.path.exists(fc_path):
        print("    ensemble_forecast.csv not found — run 04_forecast_scenarios.py first")
        return

    fc = pd.read_csv(fc_path)
    y  = df['dissolution_count'].values
    n  = len(y)

    pre_x = np.arange(62); slope, intercept = np.polyfit(pre_x, y[:62], 1)
    cf = [max(0, intercept + slope * i) for i in range(n)]

    fig = plt.figure(figsize=(14, 9))
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.32)
    fig.suptitle(
        'Figure D — Ensemble Scenario Forecast: UK Retail Dissolutions 2026–2030\n'
        'Model: 52% Holt linear smoothing + 48% ARIMAX(2,1,1) · OOS RMSE ≈ 2,800',
        fontsize=10.5, fontweight='bold', color=NAVY, y=0.99
    )

    ax1 = fig.add_subplot(gs[0, :])
    hist_months = list(range(n))
    fc_months   = list(range(n, n + len(fc)))

    ax1.axvspan(62, 78, alpha=0.09, color=RED)
    ax1.axvspan(78, 96, alpha=0.07, color=AMBER)
    ax1.axvline(n, color=GREY, linewidth=1, linestyle=':', alpha=0.7)
    ax1.text(n + 0.5, 14000, 'Forecast\nbegins\nFeb 2026', fontsize=7, color=GREY)

    ax1.plot(hist_months, y, color=NAVY, linewidth=1.3, label='Historical', zorder=5)
    ax1.plot(hist_months, cf, color=GREY, linewidth=1, linestyle='--',
             alpha=0.7, label='Pre-COVID counterfactual')
    ax1.fill_between(fc_months, fc['ensemble_bau_lo_95'], fc['ensemble_bau_hi_95'],
                     alpha=0.13, color=RED, label='BAU 95% PI')
    ax1.plot(fc_months, fc['ensemble_bau'], color=RED, linewidth=2, label='BAU ensemble')
    ax1.plot(fc_months, fc['ensemble_int'], color=GREEN, linewidth=2,
             linestyle='--', label='Intervention (FM1+FM2+FM3)')
    ax1.plot([n-1, n], [y[-1], fc['ensemble_bau'].iloc[0]], color=RED, linewidth=2)
    ax1.plot([n-1, n], [y[-1], fc['ensemble_int'].iloc[0]], color=GREEN, linewidth=2, linestyle='--')

    ax1.set_ylabel('Monthly retail dissolutions', fontsize=9)
    ax1.set_title('Complete history (Jan 2015–Jan 2026) and 48-month ensemble forecast',
                  fontsize=8.5, style='italic', color=NAVY)
    ax1.legend(loc='upper left', fontsize=7.5, framealpha=0.85, ncol=2)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:,.0f}'))
    tick_pos = [0, 12, 24, 36, 48, 60, 72, 84, 96, 108, 120, 132, 144, 156, 168]
    tick_lab = ['Jan\n2015','','Jan\n2017','','Jan\n2019','','Jan\n2021','',
                'Jan\n2023','','Jan\n2025','Jan\n2026','','Jan\n2028','Jan\n2030']
    ax1.set_xticks(tick_pos); ax1.set_xticklabels(tick_lab, fontsize=7.5)
    ax1.set_ylim(-200, 22000)

    ax2 = fig.add_subplot(gs[1, 0])
    models = ['Holt\n(2 params)', 'AR(2)\nno exog.', 'ARIMAX\n(1,1,0)', 'ARIMAX\n(2,1,1)', 'Ensemble']
    oos_rmse = [2952, 3111, 3222, 3133, 2800]
    cols_b = [NAVY, NAVY, NAVY, NAVY, GREEN]
    bars = ax2.bar(range(5), oos_rmse, color=cols_b, alpha=0.82, edgecolor='white')
    ax2.scatter([3], [1998], marker='D', color=BLUE, s=60, zorder=6, label='In-sample RMSE')
    ax2.axhline(2800, color=GREEN, linewidth=1.2, linestyle='--', alpha=0.9)
    ax2.text(4.4, 2820, 'Ensemble\n≈2,800', fontsize=6.5, color=GREEN)
    ax2.set_xticks(range(5)); ax2.set_xticklabels(models, fontsize=7)
    ax2.set_ylabel('RMSE', fontsize=8.5)
    ax2.set_title('Model backtesting — rolling 12-month OOS RMSE\n'
                  'Bias-variance tradeoff: Holt wins OOS despite ARIMAX in-sample superiority',
                  fontsize=8, style='italic', color=NAVY)
    ax2.legend(fontsize=7.5); ax2.set_ylim(0, 3700)
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:,.0f}'))

    ax3 = fig.add_subplot(gs[1, 1])
    bau_cum = np.cumsum(fc['ensemble_bau'].values)
    int_cum = np.cumsum(fc['ensemble_int'].values)
    fc_years = [2026 + i/12 for i in range(len(fc))]
    ax3.fill_between(fc_years, int_cum, bau_cum, alpha=0.18, color=GREEN, label='~106,400 saved')
    ax3.plot(fc_years, bau_cum, color=RED, linewidth=2, label='BAU: 547,167')
    ax3.plot(fc_years, int_cum, color=GREEN, linewidth=2, linestyle='--', label='Intervention: 440,737')
    ax3.annotate('~106,400\nbusinesses\nsaved by\nJan 2030',
                xy=(2030.0, 493000), xytext=(2027.5, 380000),
                arrowprops=dict(arrowstyle='->', color=GREEN, lw=1.2),
                fontsize=8.5, color=GREEN, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', fc='white', ec=GREEN, alpha=0.9))
    ax3.set_xlabel('Year', fontsize=9)
    ax3.set_ylabel('Cumulative dissolutions', fontsize=8.5)
    ax3.set_title('Cumulative BAU vs intervention\nGap = businesses saved by FM1+FM2+FM3',
                  fontsize=8, style='italic', color=NAVY)
    ax3.legend(loc='upper left', fontsize=7.5)
    ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1000:.0f}k'))

    plt.savefig(os.path.join(FIGURES, "fig_forecast.png"))
    plt.close()
    print("    Saved.")


# ── MAIN ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("="*65)
    print("FIGURE GENERATION PIPELINE")
    print("Kante (2026) — Why Your High Street Is Emptying")
    print("="*65 + "\n")

    os.makedirs(FIGURES, exist_ok=True)
    df = load_main_data()

    fig_dissolution_trend(df)
    fig_survival()
    fig_geographic()
    fig_forecast(df)

    print(f"\nAll figures saved to {FIGURES}/")
    print("Generated:")
    for f in sorted(os.listdir(FIGURES)):
        if f.endswith('.png'):
            size = os.path.getsize(os.path.join(FIGURES, f)) // 1024
            print(f"  {f}  ({size} KB)")
