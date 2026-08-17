# Customer Segmentation & CLV Prediction

Machine learning workflow for segmenting shoppers and predicting customer lifetime value (CLV) with ROI sensitivity analysis.

## Problem

Retail and e-commerce teams need to group customers by behavior, forecast lifetime value, and justify marketing spend with data-driven ROI estimates.

## Approach

Used a 3,900-record shopping behavior dataset across five Jupyter notebooks: business framing, feature engineering (CLV = purchase amount × previous purchases), K-Means segmentation (K=3 via elbow and silhouette), XGBoost CLV regression with RandomizedSearchCV, and advertising budget ROI sensitivity analysis.

## Results

- XGBoost CLV model: R² = 0.9996 (MSE 546.01)
- Three segments: young moderate spenders (~28), older high-value (~53), older low-value (~53)
- Average CLV: $1,517.88; optimal ad budget scenario: $250,000 for 5,000 customers at 29.36× ROI

## Tech stack

Python, scikit-learn, XGBoost, pandas, NumPy, matplotlib, seaborn, Plotly, Jupyter

## How to run

```bash
git clone https://github.com/DanielDemoz/customer-segmentation-clv.git
cd customer-segmentation-clv
pip install -r requirements.txt
jupyter notebook
```

Run notebooks 01–05 in order, or open the static dashboard if deployed.

## Screenshot / demo

**Live dashboard:** https://danieldemoz.github.io/customer-segmentation-clv/

Visualizations in `Visualizations/`: elbow/silhouette plots, cluster characteristics, ROI sensitivity chart.
