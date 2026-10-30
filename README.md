# Customer Segmentation and CLV: a leakage case study

A customer lifetime value (CLV) model scored R² 0.9996. A re-check found why, and this repository now shows the honest version.

**Live demo:** https://danieldemoz.github.io/customer-segmentation-clv/
**Status:** re-analysed in October 2026. The original notebooks are kept in `notebooks/` for reference.

## What the re-check found

The notebooks defined CLV as **purchase amount x previous purchases**, then trained XGBoost to predict it with both of those columns as inputs.

| Model (5-fold cross-validation) | R² |
|---|---|
| Gradient boosting, all columns (as in the notebook) | 0.9996 |
| Gradient boosting, without the two formula columns | -0.0632 |
| Linear regression, without the two formula columns | -0.0121 |

1. **Target leakage.** The model re-did a multiplication. Without the formula inputs, nothing else in the data predicts CLV.
2. **The data looks synthetic.** Numeric fields are almost unrelated (largest correlation 0.04) and evenly spread.
3. **The segments are weak.** Silhouette for 3 groups is 0.18, and 4 to 6 groups score about the same.
4. **The 29.36x ROI is an assumption.** It assumes every targeted customer becomes a repeat buyer and all revenue is profit. At a 30% margin and a 20% repeat rate, the same numbers give about 0.8x.

## The demo

- A CLV calculator that shows the formula openly, with the nearest segment.
- Segment profiles, labelled as weak.
- An ROI scenario where you set the margin and repeat rate.

## Responsible AI, following the Reliable AI Adoption Framework

1. **Business problem:** decide which customers deserve marketing spend. That needs honest value estimates, not a model that repeats its inputs.
2. **Data readiness:** 3,900 records from a public shopping-behaviour dataset. It looks synthetic, so findings may not hold for a real store.
3. **Governance and risk:** leakage and optimistic assumptions can drive real budgets. Every number on the demo shows how it was produced.
4. **Technology and tools:** a formula replaces the model, because the target was a formula. Clustering is kept and labelled as weak.
5. **People and adoption:** the calculators put the assumptions in the open, so a marketing lead can test them before committing a budget.
6. **Outcomes and monitoring:** with real data, measure CLV on future purchases and compare segments on that.

## How to run

```bash
pip install -r requirements.txt
python analysis/run_analysis.py     # rebuilds docs/analysis.json
pytest -q
```

## Tech stack

Python, pandas, scikit-learn, vanilla JavaScript, GitHub Pages. The original notebooks also use XGBoost.
