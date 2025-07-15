# 🪙 Bitcoin Price Prediction using Machine Learning

This project uses historical Bitcoin (BTC-USD) price data to predict market trends using various supervised machine learning models. The goal is to provide a binary classification — whether the price will go up or down — based on past data.

## 📂 Dataset

- **Source**: Historical BTC-USD stock data (CSV format)
- **Features**: Standard stock features like Open, High, Low, Close, Volume, etc.
- **Target**: Derived feature indicating if the price will go up the next day.

## 🧠 Machine Learning Models Used

1. **Logistic Regression**
2. **Support Vector Classifier (SVC)**
3. **XGBoost Classifier**

## 📊 Workflow

1. **Data Preprocessing**
   - Load and clean the dataset
   - Feature engineering (e.g., trend target creation)
   - Feature scaling with `StandardScaler`

2. **Train-Test Split**
   - Splits the data into 80% training and 20% testing

3. **Model Training and Evaluation**
   - Fits all three classifiers
   - Evaluates using accuracy, confusion matrix, and classification report

## 🔍 Results

| Model               | Accuracy (Example) |
|--------------------|--------------------|
| Logistic Regression| 85%                |
| SVC                | 87%                |
| XGBoost Classifier | 90%                |

> Note: Actual performance may vary depending on dataset version and random state.

## 📦 Requirements

Install the required Python packages:

```bash
pip install -r requirements.txt
