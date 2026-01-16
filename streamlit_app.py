import streamlit as st

st.title("Hello World!")
st.write("My AI Portfolio Advisor is coming soon...")
import pandas as pd

uploaded_file = st.file_uploader("Upload dataset", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write(df.head())
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

uploaded_file = st.file_uploader("📤 Upload your CSV file", type=["csv"])
if uploaded_file:
    # Load the CSV
    df = pd.read_csv(uploaded_file)

    st.subheader("🔍 Data Preview")
    st.write(df.head())

    # Handle datasets where Date column may have spaces or different names
    df = df.rename(columns=lambda x: x.strip())

    # List columns except Date
    stock_cols = df.columns[1:]   # Assuming first column is Date

    # User picks stock
    target_stock = st.selectbox("🎯 Select Stock to Predict", stock_cols)
    df['index'] = np.arange(len(df))
    train_size = int(len(df) * 0.9)
    train = df.iloc[:train_size]
    test = df.iloc[train_size:]

    X_train = train[['index']]
    y_train = train[target_stock]
    X_test = test[['index']]
    y_test = test[target_stock]
    model = LinearRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    # Build DF: actual + predicted
    predicted_series = np.concatenate([np.repeat(np.nan, len(train)), predictions])

    output_df = pd.DataFrame({
        "Actual": df[target_stock],
        "Predicted": predicted_series
    })

    st.subheader(f"📊 Actual vs Predicted — {target_stock}")
    st.line_chart(output_df)
    next_index = len(df)
    next_pred = model.predict([[next_index]])

    st.subheader("🔮 Forecast Next Value")
    st.success(f"**Expected next price for {target_stock} → {next_pred[0]:.2f}**")

else:
    st.info("👆 Upload a CSV file to start.")


