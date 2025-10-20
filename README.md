# Customer Segmentation & Customer Lifetime Value (CLV) Prediction

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![XGBoost](https://img.shields.io/badge/XGBoost-1.6+-orange.svg)](https://xgboost.readthedocs.io/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.20+-red.svg)](https://streamlit.io)

## Project Overview

This project demonstrates **customer segmentation** and **customer lifetime value (CLV) prediction** using advanced **machine learning** and **marketing analytics** techniques. It applies both **unsupervised learning** (K-Means clustering) and **supervised learning** (XGBoost regression) to help businesses improve **customer retention**, **predictive modeling accuracy**, and **ROI analysis** for targeted marketing.

### Key Achievements
- **99.96% Model Accuracy** (R² = 0.9996) with XGBoost
- **3 Distinct Customer Segments** identified with actionable insights
- **29.36x ROI** potential on advertising investment
- **$1,517.88 Average CLV** per customer

---

## Dataset
**File:** `shopping_behavior_updated.csv`  
**Size:** 3,900 customer records  
**Features:** 18 attributes including:
- Demographics (age, gender, location)
- Purchase behavior metrics
- Engagement and activity data
- Transactional history for CLV modeling

---

## Project Workflow

### 1. Business Understanding  
**File:** `01_business_understanding.ipynb`  
- Defined objectives for customer segmentation and CLV forecasting
- Identified key **KPIs** for customer analytics
- Established success metrics and evaluation criteria

### 2. Data Preparation & Feature Engineering  
**File:** `02_Data_Preparation_&_Feature_Engineering.ipynb`  
- Data cleaning and missing value handling
- Categorical variable encoding for **machine learning**
- Feature scaling and normalization
- Target variable creation (CLV = Purchase Amount × Previous Purchases)

### 3. Customer Segmentation (Unsupervised Learning)  
**File:** `03_Clustering_elbow,_silhouette,_K=3_segmentation.ipynb`  
- Applied **K-Means clustering** to identify customer groups
- Used **Elbow Method** and **Silhouette Score** for optimal cluster selection
- Visualized **cluster characteristics** for marketing strategy
- **Result:** 3 distinct customer segments identified

### 4. CLV Prediction (Supervised Learning)  
**File:** `04_CLV_prediction_(XGBoost)_—_split,_search,_train,_evaluate.ipynb`  
- Built **XGBoost regression** model for CLV prediction
- Performed **hyperparameter tuning** with RandomizedSearchCV
- Model comparison (XGBoost vs Random Forest)
- **Result:** 99.96% accuracy with XGBoost

### 5. ROI Sensitivity Analysis  
**File:** `05_CLV_prediction_ROI_analysis.ipynb`  
- Conducted **ROI sensitivity analysis** on advertising budgets
- Calculated optimal customer acquisition strategies
- **Result:** 29.36x ROI with $250,000 budget for 5,000 customers

---

## Key Results & Insights

### Model Performance
- **XGBoost R² Score:** 0.9996 (99.96% accuracy)
- **Mean Squared Error:** 546.01
- **Cross-validation:** Consistent performance across folds

### Customer Segmentation
- **Segment 0:** Young moderate spenders (avg age ~28)
- **Segment 1:** Older high-value customers (avg age ~53, highest spend)
- **Segment 2:** Older low-value customers (avg age ~53, lowest spend)

### Business Impact
- **Average CLV:** $1,517.88 per customer
- **ROI Potential:** 29.36x return on advertising investment
- **Optimal Ad Budget:** $250,000 for 5,000 new customers
- **Revenue Optimization:** Targeted segmentation enables better budget allocation

---

## Tech Stack
- **Languages:** Python 3.8+
- **ML Libraries:** Scikit-learn, XGBoost, Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn, Plotly
- **Dashboard:** Streamlit, Dash
- **Tools:** Jupyter Notebook, Joblib
- **Methods:** K-Means clustering, RFM analysis, predictive modeling, ROI analysis

---

## Quick Start

### Installation
```bash
# Clone the repository
git clone https://github.com/DanielDemoz/customer-segmentation-clv.git
cd customer-segmentation-clv

# Install dependencies
pip install -r requirements.txt
```

### Running the Analysis
```bash
# Start Jupyter Notebook
jupyter notebook

# Or run individual notebooks
jupyter nbconvert --execute 01_business_understanding.ipynb
```

### Interactive Dashboard
```bash
# Install Streamlit
pip install streamlit

# Run the dashboard (when implemented)
streamlit run dashboard.py
```

---

## Interactive Features

### Live CLV Predictor
- Input customer characteristics
- Get instant CLV predictions
- View confidence intervals and segment assignments

### Customer Segment Explorer
- Interactive segment comparison
- Characteristic sliders for exploration
- Revenue impact calculator

### ROI Optimization Tool
- Budget allocation simulator
- What-if scenario testing
- Sensitivity analysis charts

### Model Performance Monitor
- Real-time accuracy tracking
- Feature importance visualization
- A/B testing framework

---

## Repository Structure
```
customer-segmentation-clv/
├── Analysis Notebooks/
│   ├── 01_business_understanding.ipynb
│   ├── 02_Data_Preparation_&_Feature_Engineering (1).ipynb
│   ├── 03_Clustering_elbow,_silhouette,_K=3_segmentation.ipynb
│   ├── 04_CLV_prediction_(XGBoost)_—_split,_search,_train,_evaluate.ipynb
│   └── 05_CLV_prediction_ROI_analysis.ipynb
├── Visualizations/
│   ├── Cluster Characteristics.png
│   ├── Elbow Method for Optimal K.png
│   ├── Silhouette Method for Optimal K.png
│   └── ROI Sensitivity to Advertising Budget.png
├── Documentation/
│   ├── README.md
│   ├── PRESENTATION_GUIDE.md
│   └── DASHBOARD_IMPLEMENTATION.md
├── Data/
│   └── shopping_behavior_updated.csv
├── Configuration/
│   ├── requirements.txt
│   └── LICENSE
└── Deployment/
    └── (Dashboard files - to be implemented)
```

---

## Key Features

### Customer Segmentation
- **K-Means Clustering** with optimal K=3 selection
- **Elbow Method** and **Silhouette Analysis** for validation
- **3 Distinct Segments** with clear characteristics
- **Interactive Visualizations** for segment exploration

### CLV Prediction
- **XGBoost Regression** with 99.96% accuracy
- **Hyperparameter Tuning** using RandomizedSearchCV
- **Feature Engineering** with categorical encoding
- **Model Comparison** (XGBoost vs Random Forest)

### ROI Analysis
- **Sensitivity Analysis** for advertising budgets
- **Optimal Budget Calculation** ($250K for 5K customers)
- **29.36x ROI** potential identification
- **What-if Scenario** testing capabilities

---

## Interactive Dashboard Features

### Executive Dashboard
- Customer value distribution
- CLV prediction accuracy metrics
- ROI calculator with real-time updates
- Revenue impact projections

### Marketing Dashboard
- Customer journey mapping
- Campaign performance tracking
- Personalization engine
- Churn prediction alerts

### Data Science Dashboard
- Model performance monitoring
- Feature importance analysis
- A/B testing framework
- Data quality metrics

---

## Getting Started

### Prerequisites
- Python 3.8 or higher
- Jupyter Notebook
- Git

### Installation Steps
1. **Clone the repository**
   ```bash
   git clone https://github.com/DanielDemoz/customer-segmentation-clv.git
   cd customer-segmentation-clv
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the analysis**
   ```bash
   jupyter notebook
   ```

4. **Launch interactive dashboard** (when implemented)
   ```bash
   streamlit run dashboard.py
   ```

---

## Results Summary

| Metric | Value | Description |
|--------|-------|-------------|
| **Model Accuracy** | 99.96% | XGBoost R² Score |
| **Average CLV** | $1,517.88 | Per customer |
| **ROI Potential** | 29.36x | Return on advertising |
| **Customer Segments** | 3 | Distinct groups identified |
| **Optimal Ad Budget** | $250,000 | For 5,000 new customers |

---

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Contact

**Daniel Demoz** - [@DanielDemoz](https://github.com/DanielDemoz)

Project Link: [https://github.com/DanielDemoz/customer-segmentation-clv](https://github.com/DanielDemoz/customer-segmentation-clv)

---

## Acknowledgments

- Dataset sourced from customer behavior analytics
- Machine learning techniques from scikit-learn and XGBoost
- Visualization libraries: Matplotlib, Seaborn, Plotly
- Dashboard frameworks: Streamlit, Dash

---

## Future Enhancements

- [ ] Real-time dashboard implementation
- [ ] API development for model serving
- [ ] Advanced feature engineering
- [ ] Model monitoring and retraining pipeline
- [ ] Integration with CRM systems
- [ ] Mobile-responsive dashboard
- [ ] Advanced A/B testing framework
- [ ] Multi-language support
