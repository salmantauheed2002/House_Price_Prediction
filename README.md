# 🏠 House Price Prediction

A machine learning project that predicts house sale prices using the Ames Housing dataset from Kaggle. This project covers the full ML pipeline — from data cleaning and EDA to feature engineering, model training, and submission.

---

## 📊 Project Overview

| Item | Detail |
|---|---|
| Dataset | Ames Housing Dataset (Kaggle) |
| Task | Regression — Predict Sale Price |
| Best Model | Linear Regression |
| Validation RMSE | 0.1265 |
| Kaggle Score | 0.19569 |
| Features Used | 83 (after feature engineering) |

---

## 🔧 Tech Stack

- Python
- Pandas, NumPy
- Matplotlib, Seaborn
- Scikit-learn
- XGBoost

---

## 📁 Project Structure

```
house-price-prediction/
│
├── house-price-prediction.ipynb   # Main notebook
├── submission.csv                 # Final Kaggle submission
└── README.md                      # Project documentation
```

---

## 🔍 Approach

### 1. Exploratory Data Analysis
- Analyzed 81 features across 1460 houses
- Identified and visualized missing values across 19 columns
- Explored SalePrice distribution — found heavy right skew (skewness = 1.88)
- Applied log transformation to normalize the target variable

### 2. Data Cleaning
- Filled categorical missing values with `"None"` (e.g. no pool, no garage)
- Filled numeric missing values with `0` or median
- Removed 2 outliers — large houses sold at unusually low prices

### 3. Feature Engineering
Created 4 new features:
- `TotalSF` — combined square footage (basement + floor 1 + floor 2)
- `TotalBathrooms` — all bathrooms combined (half baths = 0.5)
- `HouseAge` — age of house at time of sale
- `IsRemodeled` — whether house was ever remodeled (1/0)

### 4. Model Training & Comparison

| Model | RMSE |
|---|---|
| Linear Regression | **0.1265** 🥇 |
| XGBoost | 0.1286 🥈 |
| Random Forest | 0.1454 🥉 |

### 5. Key Findings
- `OverallQual` is the strongest predictor of house price (importance: 0.29)
- `TotalSF` — an engineered feature — ranked 2nd most important (0.21)
- Linear Regression outperformed tree-based models on this dataset

---

## 📈 Results

- Validation RMSE: **0.1265** (~12% average error on validation set)
- Kaggle Leaderboard Score: **0.19569**
- For a $200,000 house, predictions are off by ~$24,000 on average

---

## 🚀 How to Run

1. Clone this repository
```bash
git clone https://github.com/YOUR_USERNAME/house-price-prediction.git
```

2. Open the notebook on Kaggle or locally with Jupyter

3. Download the dataset from:
```
https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques
```

4. Run all cells in order

---

## 💡 What I Learned

- How to handle missing values strategically (None vs 0 vs median)
- Importance of log transforming skewed target variables
- Feature engineering can outperform raw features (TotalSF ranked #2)
- Simpler models like Linear Regression can beat complex ones on clean data
- How to build and compare multiple ML models end to end

---

## 🔮 Future Improvements

- Try Ridge/Lasso regression to reduce overfitting
- Add more feature engineering (neighborhood encodings, price per sqft)
- Use cross-validation instead of single train/val split
- Try stacking multiple models for better Kaggle score
- Replace `YOUR_USERNAME` with your actual GitHub username
- Replace the `[LinkedIn](#)` link with your real LinkedIn URL
- Click **"Commit changes"**

---
