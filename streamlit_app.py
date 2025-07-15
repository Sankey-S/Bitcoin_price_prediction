import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

st.title("Bitcoin Movement Prediction App")

st.markdown("""
Upload a Bitcoin historical price CSV (like `bitcoin.csv`) and choose a model to predict whether the price will go up the next day.
""")

# Upload CSV
uploaded_file = st.file_uploader("Upload your input CSV file", type=["csv"])

# Model selection
model_name = st.selectbox("Select model", ["Logistic Regression", "SVM", "XGBoost"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # Preprocess
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.drop(['Adj Close'], axis=1)

    df['year'] = df['Date'].dt.year
    df['month'] = df['Date'].dt.month
    df['day'] = df['Date'].dt.day
    df['is_quarter_end'] = np.where(df['month'] % 3 == 0, 1, 0)
    df['open-close'] = df['Open'] - df['Close']
    df['low-high'] = df['Low'] - df['High']
    df['target'] = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)

    # Drop last row due to shift
    df = df[:-1]

    X = df[['open-close', 'low-high', 'is_quarter_end']]
    y = df['target']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Train and evaluate all models
    models = {
        "Logistic Regression": LogisticRegression(),
        "SVM": SVC(),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    }

    accuracies = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        accuracies[name] = acc

    best_model_name = max(accuracies, key=accuracies.get)
    best_accuracy = accuracies[best_model_name]

    st.markdown(f"### Best Model: {best_model_name} with Accuracy: {best_accuracy:.2f}")

    # Use selected model for prediction
    selected_model = models[model_name]
    selected_model.fit(X_scaled, y)
    predictions = selected_model.predict(X_scaled)

    # Show predictions
    df_result = df[['Date', 'Close']].copy()
    df_result['Prediction'] = predictions
    df_result['Movement'] = df_result['Prediction'].map({1: 'Up', 0: 'Down'})

    st.subheader("Prediction Results")
    st.dataframe(df_result.tail(20))

    # Download option
    csv = df_result.to_csv(index=False).encode('utf-8')
    st.download_button("Download Predictions", csv, "predictions.csv", "text/csv")
