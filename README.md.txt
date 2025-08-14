# Customer Segmentation & CLV Prediction

## Project Overview
This project applies Customer Segmentation and Customer Lifetime Value (CLV) Prediction to identify distinct customer groups and forecast their potential value over time.  
It also includes ROI sensitivity analysis to guide marketing budget allocation for maximum return.

The workflow combines unsupervised learning (K-Means clustering) and supervised learning (XGBoost regression) with actionable business recommendations.

---

## Dataset
**File:** `shopping_behavior_updated.csv`  
Contains:
- Demographics (age, gender, location, etc.)
- Purchase behavior
- Engagement metrics
- Transactional history

---

## Project Workflow

### 1. Business Understanding  
File: `01_business_understanding.ipynb`  
- Defined project objectives.
- Identified KPIs for segmentation and CLV modeling.

### 2. Data Preparation & Feature Engineering  
File: `02_Data_Preparation_&_Feature_Engineering.ipynb`  
- Cleaned missing values.
- Encoded categorical variables.
- Created RFM features (Recency, Frequency, Monetary Value).

### 3. Customer Segmentation (Unsupervised Learning)  
File: `03_Clustering_elbow_silhouette_K=3_segmentation.ipynb`  
- Applied K-Means clustering to group customers.  
- Determined optimal number of clusters using:
  - Elbow Method (`Elbow Method for Optimal K.png`)
  - Silhouette Score (`Silhouette Method for Optimal K.png`)
- Visualized cluster profiles (`Cluster Characteristics.png`).

### 4. CLV Prediction (Supervised Learning)  
File: `04_CLV_prediction_(XGBoost)__split_search_evaluation.ipynb`  
- Built XGBoost regression model for CLV prediction.
- Tuned hyperparameters with grid search.
- Evaluated performance using RMSE and R².

### 5. ROI Sensitivity Analysis  
File: `05_CLV_prediction_ROI_analysis.ipynb`  
- Analyzed ROI sensitivity to changes in advertising budget (`ROI Sensitivity to Advertising Budget.png`).
- Provided data-driven recommendations for marketing spend allocation.

---

## Key Results & Insights
- Segmented customers into 3 groups with distinct purchase and engagement patterns.
- High-value customers showed:
  - High purchase frequency
  - High monetary value
  - Low recency (recent transactions)
- ROI analysis revealed:
  - Optimal advertising budget allocation for maximum return.
  - Diminishing returns beyond certain spend thresholds.

---

## Tech Stack
- Languages: Python
- Libraries: Pandas, NumPy, Scikit-learn, XGBoost, Matplotlib, Seaborn
- Tools: Jupyter Notebook
- Visualization: Matplotlib, Seaborn

---

## Repository Structure
