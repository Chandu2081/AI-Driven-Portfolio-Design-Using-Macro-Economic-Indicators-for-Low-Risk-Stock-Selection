import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error, mean_squared_error
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.api import add_constant


st.title("📊 AI Portfolio Analytics Dashboard")
st.write("Upload dataset → Select stock → View correlation, VIF, regression fit, scatter plots & accuracy.")

# =====================================
# 1. UPLOAD FILE
# =====================================
uploaded_file = st.file_uploader("📤 Upload your CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df = df.rename(columns=lambda x: x.strip())

    # Fill missing values
    df = df.fillna(method='ffill')
    df = df.fillna(0)

    st.subheader("🔍 Data Preview")
    st.write(df.head())

    columns = df.columns[1:]  # skip Date column
    target_stock = st.selectbox("🎯 Select stock to analyze", columns)

    # =====================================
    # 2. CORRELATION HEATMAP
    # =====================================
    st.subheader("📈 Correlation Heatmap")

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(df[columns].corr(), cmap="coolwarm", ax=ax)
    st.pyplot(fig)

    # =====================================
    # 3. VIF CALCULATION
    # =====================================
    st.subheader("🧮 Variance Inflation Factor (VIF)")
    numeric_df = df[columns].select_dtypes(include=[np.number])

    numeric_sample = numeric_df.iloc[:, :10]  # take first 10 to control runtime
    X_vif = add_constant(numeric_sample)

    vif_data = pd.DataFrame()
    vif_data["Feature"] = X_vif.columns
    vif_data["VIF"] = [variance_inflation_factor(X_vif.values, i) for i in range(X_vif.shape[1])]
    st.write(vif_data)

    # =====================================
    # 4. BUILD TIME INDEX & TRAIN
    # =====================================
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

    # =====================================
    # 5. METRICS
    # =====================================
    mae = mean_absolute_error(y_test, predictions)
    mape = mean_absolute_percentage_error(y_test, predictions) * 100
    mse = mean_squared_error(y_test, predictions)

    st.subheader("📊 Model Accuracy Metrics")
    st.write(f"**MAE:** {mae:.3f}")
    st.write(f"**MAPE:** {mape:.2f}%")
    st.write(f"**MSE:** {mse:.3f}")

    # =====================================
    # 6. REGRESSION EQUATION
    # =====================================
    slope = model.coef_[0]
    intercept = model.intercept_
    st.subheader("🧾 Regression Equation")
    st.write(f"**y = {slope:.4f} × index + {intercept:.4f}**")

    # =====================================
    # 7. LINE CHART (Actual vs Predicted)
    # =====================================
    st.subheader("📊 Actual vs Predicted Curve")
    pred_series = np.concatenate([np.repeat(np.nan, len(train)), predictions])
    chart_df = pd.DataFrame({"Actual": df[target_stock], "Predicted": pred_series})
    st.line_chart(chart_df)

    # =====================================
    # 8. SCATTER PLOT WITH REGRESSION LINE
    # =====================================
    st.subheader("📉 Scatter Plot With Regression Line (Train Data)")

    fig2, ax2 = plt.subplots(figsize=(8,4))
    ax2.scatter(train['index'], y_train, color='blue', label="Actual")
    ax2.plot(train['index'], model.predict(X_train), color='red', label=f"Fit Line")
    ax2.set_xlabel("Time Index")
    ax2.set_ylabel(target_stock)
    ax2.legend()
    st.pyplot(fig2)

    # =====================================
    # 9. SCATTER PLOT TEST SECTION
    # =====================================
    st.subheader("📍 Scatter Plot on Test Data")

    fig3, ax3 = plt.subplots(figsize=(8,4))
    ax3.scatter(test['index'], y_test, color='green', label="Actual Test")
    ax3.plot(test['index'], predictions, color='orange', label="Predicted Line")
    ax3.set_xlabel("Time Index (Test)")
    ax3.set_ylabel(target_stock)
    ax3.legend()
    st.pyplot(fig3)

else:
    st.info("👆 Upload a CSV file to start analysis.")

