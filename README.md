# Why Your High Street Is Emptying
### A Rigorous Data Science Account of UK Retail Structural Collapse

**Venkata Prudhvi Kante**  
Data Scientist  
MSc Business Analytics

📧 prudhvi.ncsnlr@gmail.com

---

## Read the Paper

PDF: paper/LocalCommerce_Crisis_Kante2026.pdf (this repo)
SSRN: add link after uploading
arXiv: add link after uploading

---

## What This Paper Is About

Rigorous data science analysis of why UK high streets are emptying. Combines three verified government datasets and ten statistical methods to establish a causal link between online retail penetration and physical retail enterprise collapse.

Central finding: UK retail went from +6,618 net new businesses/year (2016-2019) to -1,648/year (2022-2024). Granger causality confirms online retail penetration CAUSES enterprise dissolution with a two-year lag.

---

## Key Statistics

Structural break (Chow): F(2,129)=133.08, p<0.001
Break date: April 2020
Pre-COVID trend: +0.140%/month, R2=0.958
Cohen's d: 4.84 (6x the large-effect threshold)
Regression: b=-752 enterprises/pp (p=0.003, R2=0.686)
Granger causality lag 2: F=123.81, p=0.008
12-month forecast: 29.72% [26.44%, 33.00%]
Active retail businesses: 430,649

---

## Methods (all from scratch, no black-box libraries)

ADF unit root | STL decomposition | HP filter | Chow structural break
Welch t-test + Cohen's d | Holt-Winters additive | Engle-Granger cointegration
OLS + Newey-West HAC | Granger causality | VAR(1) + IRF

---

## Data Sources (all free, official UK government)

ONS RSI J4MC: https://www.ons.gov.uk/businessindustryandtrade/retailindustry/datasets/retailsales/current
ONS Business Demography: https://www.ons.gov.uk/businessindustryandtrade/business/activitysizeandlocation/datasets/businessdemographyreferencetable
Companies House: https://download.companieshouse.gov.uk/en_output.html
DBT SME Taskforce: https://www.gov.uk/government/publications/sme-digital-adoption-taskforce-final-report

---

## How to Reproduce

pip install numpy scipy pandas matplotlib
python3 analysis/full_analysis_real_data.py

---

## Repository Structure

paper/ - PDF, LaTeX source, bibliography
analysis/ - full_analysis_real_data.py
figures/ - all 9 publication figures
data/ - download instructions

---

## Citation

Kante, V. P. (2026). Why Your High Street Is Emptying: A Rigorous Data Science Account of UK Retail Structural Collapse. Independent Research Preprint.

---

## License

Code: MIT. Paper: Copyright Venkata Prudhvi Kante (2026). Data: Crown Copyright.
