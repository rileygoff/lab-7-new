import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load Dataset
def load_data():
    file_path = "/mnt/data/Ameshousing.csv"
    try:
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return None

def preprocess_data(df):
    if df is None:
        return None, None, None
    df = df.select_dtypes(include=[np.number])  # Keep only numeric columns
    df = df.fillna(df.median())  # Handle missing values
    if 'SalePrice' not in df.columns:
        st.error("Error: 'SalePrice' column missing from dataset.")
        return None, None, None
    X = df.drop(columns=['SalePrice'])
    y = df['SalePrice']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y, scaler

# Train Model
def train_model(X, y):
    if X is None or y is None:
        return None, None, None
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model, X_test, y_test

# Load & Process Data
df = load_data()
X, y, scaler = preprocess_data(df)
model, X_test, y_test = train_model(X, y)
if model:
    joblib.dump(model, "model.pkl")
    joblib.dump(scaler, "scaler.pkl")

# Streamlit Web App
st.title("Ames Housing Price Prediction")
st.write("Enter property details below to predict the house price.")

if df is not None and model is not None:
    # Create input fields for user
    inputs = {}
    for feature in df.drop(columns=['SalePrice'], errors='ignore').select_dtypes(include=[np.number]).columns:
        inputs[feature] = st.number_input(feature, float(df[feature].min()), float(df[feature].max()), float(df[feature].mean()))

    # Predict Price
    if st.button("Predict Price"):
        input_data = np.array([inputs[feature] for feature in inputs]).reshape(1, -1)
        input_data = scaler.transform(input_data)
        prediction = model.predict(input_data)[0]
        st.success(f"Estimated House Price: ${prediction:,.2f}")

    # Model Evaluation
    st.subheader("Model Performance")
    y_pred = model.predict(X_test)
    st.write(f"Mean Absolute Error: {mean_absolute_error(y_test, y_pred):,.2f}")
    st.write(f"Mean Squared Error: {mean_squared_error(y_test, y_pred):,.2f}")
    st.write(f"R² Score: {r2_score(y_test, y_pred):.2f}")
else:
    st.error("Model training failed. Please check dataset and try again.")
