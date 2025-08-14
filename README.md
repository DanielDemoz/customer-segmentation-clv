# Customer Segmentation & Customer Lifetime Value (CLV) Prediction

## Project Overview
This project demonstrates **customer segmentation** and **customer lifetime value (CLV) prediction** using **RFM analysis**, **machine learning**, and **marketing analytics** techniques.  
It applies both **unsupervised learning** (K-Means clustering) and **supervised learning** (XGBoost regression) to help businesses improve **customer retention**, **predictive modeling accuracy**, and **ROI analysis** for targeted marketing.

The workflow also includes **business intelligence** reporting and **data visualization** using Python libraries like Pandas, NumPy, Matplotlib, and Seaborn.

---

## Dataset
**File:** `shopping_behavior_updated.csv`  
Contains:
- Demographics (age, gender, location, etc.)
- Purchase behavior metrics
- Engagement and activity data
- Transactional history for CLV modeling

---

## Project Workflow

### 1. Business Understanding  
File: `01_business_understanding.ipynb`  
- Defined objectives for customer segmentation and CLV forecasting.
- Identified key **KPIs** for customer analytics.

### 2. Data Preparation & Feature Engineering  
File: `02_Data_Preparation_&_Feature_Engineering.ipynb`  
- Data cleaning, handling missing values.
- Encoded categorical variables for **machine learning**.
- Created **Recency, Frequency, Monetary Value** features.

### 3. Customer Segmentation (Unsupervised Learning)  
File: `03_Clustering_elbow_silhouette_K=3_segmentation.ipynb`  
- Applied **K-Means clustering** to identify customer groups.
- Used **Elbow Method** and **Silhouette Score** to determine optimal cluster count.
- Visualized **cluster characteristics** for marketing strategy.

### 4. CLV Prediction (Supervised Learning)  
File: `04_CLV_prediction_(XGBoost)__split_search_evaluation.ipynb`  
- Built an **XGBoost regression** model for CLV prediction.
- Performed **hyperparameter tuning** to improve predictive accuracy.

### 5. ROI Sensitivity Analysis  
File: `05_CLV_prediction_ROI_analysis.ipynb`  
- Conducted **ROI sensitivity analysis** on different advertising budgets.
- Identified diminishing returns beyond optimal spend thresholds.

---

## Key Results & Insights
- Segmented customers into 3 groups with distinct purchase patterns.
- High-value customers showed high frequency, high monetary value, and low recency.
- CLV predictions enable better **marketing budget allocation** and **campaign targeting**.
- ROI analysis provides a data-backed approach to optimize ad spend.

---

## Tech Stack
- **Languages:** Python
- **Libraries:** Pandas, NumPy, Scikit-learn, XGBoost, Matplotlib, Seaborn
- **Tools:** Jupyter Notebook
- **Methods:** K-Means clustering, RFM analysis, predictive modeling, ROI analysis

---

## Repository Structure
