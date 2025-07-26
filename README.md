# 🏡 Price My House – House Price Prediction with Regression

This project aims to predict house prices using **regression models** on a real-world dataset. By analyzing features such as lot area, number of rooms, and neighborhood, we estimate the selling price of a house.

> 📊 A perfect beginner-friendly project to understand regression, feature engineering, and model evaluation.

---

## 🔍 Problem Statement

Given housing data with various features (location, area, rooms, etc.), build a machine learning model to **predict the sale price** of homes.

---

## 📁 Dataset

- Source: [Kaggle – House Prices: Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques)
- Files:
  - `train.csv` – training data (with target)
  - `test.csv` – test data (without target)

---

## ⚙️ Technologies Used

- Python 🐍
- Pandas, NumPy
- Matplotlib, Seaborn
- Scikit-learn
- Joblib (for model saving)

---

## 🧼 Data Preprocessing

✅ Steps followed:

- Handled missing values:
  - Dropped columns with more than **300+ missing values**
  - Imputed others using **mean/median/mode**
- Feature Engineering:
  - Converted categorical variables using **One-Hot Encoding**
- Scaled data using `StandardScaler`

---

## 📊 Exploratory Data Analysis (EDA)

- Analyzed feature correlations with `SalePrice`
- Visualized:
  - Distribution of prices
  - Relationship between features and price
  - Missing data heatmaps

*(Add screenshots here if available)*

---

## 🤖 Model Training

### ✔️ Algorithms Tried:
- Linear Regression
- Ridge Regression
- Lasso Regression
- Random Forest Regressor

### ✅ Final Accuracy:
- **Best model achieved R² score of: `95.07`** on test set

---

## 📌 Key Visualizations

| Plot | Description |
|------|-------------|
| 📈 Distribution Plot | Shows the spread of house prices |
| 🔥 Correlation Heatmap | Features highly related to price |
| 📊 Feature Importance | Top contributors to price prediction |

---

## 🧠 How to Use

```python
# Load model and scaler
import joblib
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')

# Predict on new data
scaled_input = scaler.transform(new_data)
predicted_price = model.predict(scaled_input)
