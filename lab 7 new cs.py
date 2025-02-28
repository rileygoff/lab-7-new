import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load Dataset
def load_data(uploaded_file):
    try:
        # Check file extension and read accordingly
        if uploaded_file is not None:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith(('.xls', '.xlsx')):
                df = pd.read_excel(uploaded_file, engine='openpyxl')  # openpyxl for xlsx
            else:
                st.error("Unsupported file format. Please upload a CSV or Excel file.")
                return None
            return df
        return None
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

# Streamlit Web App
st.title("Ames Housing Price Prediction")
st.write("Upload a CSV or Excel file below to predict the house price.")

uploaded_file = st.file_uploader("Choose a CSV or Excel file", type=["csv", "xls", "xlsx"])

# Load & Process Data
df = load_data(uploaded_file)
X, y, scaler = preprocess_data(df)
model, X_test, y_test = train_model(X, y)

if model:
    joblib.dump(model, "model.pkl")
    joblib.dump(scaler, "scaler.pkl")

    # User Input for Prediction
    st.write("Enter property details below to predict the house price.")
    inputs = {}
    if df is not None:
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
