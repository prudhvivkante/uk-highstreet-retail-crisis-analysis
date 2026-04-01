# Why Your High Street Is Emptying
### UK Retail Structural Collapse — Kante (2026)

**Paper:** [paper/LocalCommerce_Crisis_Kante2026.pdf](paper/LocalCommerce_Crisis_Kante2026.pdf)  
**Author:** Venkata Prudhvi Kante · prudhvi.ncsnlr@gmail.com  
**Repo:** https://github.com/prudhvivkante/uk-highstreet-retail-crisis-analysis

## Key Findings
| Metric | Value |
|--------|-------|
| Total dissolutions Jan 2015–Jan 2026 | **666,462** |
| Jan 2026 rate vs pre-COVID avg | **5.9×** (10,781 vs 1,954/month) |
| Excess dissolutions above counterfactual | **~215,800** |
| 5-yr survival 2010–12 cohort | **75.7%** |
| 5-yr survival 2016–18 cohort | **3.6%** (20-fold collapse) |
| Granger causal lag | **1–2 months** |
| OLS β̂ (monthly, HAC) | **+599/pp** (t=9.29, p<0.001, R²=0.711) |
| BAU forecast 2026–2030 | **547,167** dissolutions |
| Saved by intervention | **~106,400** businesses |

## Structure
```
data/raw/        — Companies House CSV downloads (8 monthly patches + main)
data/processed/  — 133-month panel, ONS J4MC, forecast, SIC breakdown
scripts/         — 6 analysis scripts (01→05 pipeline + figure_generation_v2)
figures/         — All 13 paper figures (PNG, 150–300 DPI)
paper/           — PDF + LaTeX source + refs.bib + figs/
dashboard/       — 3 standalone HTML dashboards (no server needed)
```

## Quick Start
```bash
pip install -r requirements.txt
python scripts/03_statistical_analysis.py   # All stats
python scripts/04_forecast_scenarios.py     # Forecasts
python scripts/05_generate_figures.py       # Figures
open dashboard/forecast_dashboard.html      # Interactive
```

## Data Notes
- `__4_.csv` excluded (exact duplicate of `__3_.csv`, same 2016-10 data)
- `ch_dissolution_raw.csv` (104MB) not included — use `ch_dissolution_final.csv`
- Cardiff CF14 8LH artefact: 1,892 default-address records excluded from geo analysis

## Citation
```bibtex
@misc{Kante2026,
  author = {Kante, Venkata Prudhvi},
  title  = {Why Your High Street Is Emptying},
  year   = {2026}, month = {March},
  url    = {https://github.com/prudhvivkante/uk-highstreet-retail-crisis-analysis}
}
```
