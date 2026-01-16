import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import BayesianRidge, LassoCV, RidgeCV
from sklearn.metrics import mean_absolute_percentage_error
from statsmodels.api import OLS, add_constant
from statsmodels.stats.outliers_influence import variance_inflation_factor

st.title("✨ Best Model Selection → Auto P-value Filter → Future Forecast")

uploaded_file = st.file_uploader("📤 Upload CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df = df.rename(columns=lambda x: x.strip())
    df = df.fillna(method='ffill').fillna(0)

    date_col = df.columns[0]
    columns = df.columns[1:]
    st.subheader("🔍 Data Preview")
    st.write(df.head())

    target = st.selectbox("🎯 Select target stock", columns)

    # =======================
    # STEP 1 — FEATURE IMPORTANCE
    # =======================
    X = df[columns].select_dtypes(include=[np.number]).drop(columns=[target], errors='ignore')
    y = df[target]

    rf = RandomForestRegressor(n_estimators=300, random_state=42)
    rf.fit(X, y)

    importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': rf.feature_importances_
    }).sort_values(by='Importance', ascending=False)

    st.subheader("🌲 Random Forest Feature Importance (Top 30)")
    st.dataframe(importance.head(30))

    top_features = importance.head(30)['Feature'].tolist()

    # =======================
    # STEP 2 — VIF FILTERING
    # =======================
    st.subheader("🧮 VIF Filtering")
    vif_df = pd.DataFrame()
    X_vif = add_constant(df[top_features])

    vif_df["Feature"] = X_vif.columns
    vif_df["VIF"] = [variance_inflation_factor(X_vif.values, i)
                     for i in range(X_vif.shape[1])]
    st.dataframe(vif_df)

    vif_cutoff = st.slider("VIF cutoff", 2.0, 20.0, 10.0)
    selected = vif_df[vif_df["VIF"] < vif_cutoff]["Feature"].tolist()
    if 'const' in selected:
        selected.remove('const')

    st.success(f"Selected {len(selected)} features after VIF")

    X_features = df[selected]

    # =======================
    # STEP 3 — TRAIN-TEST SPLIT
    # =======================
    split = int(len(df) * 0.9)
    X_train = X_features.iloc[:split]
    X_test = X_features.iloc[split:]
    y_train = y.iloc[:split]
    y_test = y.iloc[split:]

    # =======================
    # STEP 4 — COMPARE MODELS
    # =======================
    st.subheader("🤖 Model Comparison (Train vs Test MAPE)")

    models = {
        "Bayesian Ridge": BayesianRidge(),
        "LassoCV": LassoCV(cv=5, random_state=42),
        "RidgeCV": RidgeCV(cv=5),
        "RandomForestRegressor": RandomForestRegressor(n_estimators=300, random_state=42)
    }

    results = []
    best_name = None
    best_test = 1e9
    best_model = None

    for name, model in models.items():
        model.fit(X_train, y_train)
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)

        train_mape = mean_absolute_percentage_error(y_train, train_pred)*100
        test_mape = mean_absolute_percentage_error(y_test, test_pred)*100

        results.append([name, round(train_mape,2), round(test_mape,2)])

        if test_mape < best_test:
            best_test = test_mape
            best_name = name
            best_model = model

    perf_df = pd.DataFrame(results, columns=["Model", "Train MAPE %", "Test MAPE %"])
    st.dataframe(perf_df)
    st.success(f"🏆 Best model selected: {best_name}")

    # =======================
    # STEP 5 — FINAL OLS WITH AUTO P-VALUE FILTER
    # =======================
    st.subheader("🧾 Final OLS Regression (Auto Drop p > 0.05)")
    X_reg = add_constant(X_features)
    ols_model = OLS(y, X_reg).fit()

    pvals = ols_model.pvalues.drop("const", errors='ignore')
    sig_features = pvals[pvals < 0.05].index.tolist()

    st.write("📌 Significant variables (p < 0.05):")
    st.write(sig_features)

    X_final = add_constant(df[sig_features])
    final_ols = OLS(y, X_final).fit()
    st.write(final_ols.summary())

    # =======================
    # STEP 6 — In-sample fitted
    # =======================
    fitted = final_ols.predict(X_final)

    # =======================
    # STEP 7 — FUTURE FORECAST
    # =======================
    st.subheader("🔮 Forecast Future Points")

    future_steps = st.slider("Forecast periods ahead:", 1, 24, 6)
    future_df = df[sig_features].iloc[-1:].copy()

    forecasts = []
    for _ in range(future_steps):
        pred = final_ols.predict(add_constant(future_df))[0]
        forecasts.append(pred)
        future_df = future_df.copy()

    last_dates = pd.date_range(start=pd.to_datetime(df[date_col].iloc[-1]), periods=future_steps+1, closed='right')

    forecast_df = pd.DataFrame({"Date": last_dates, "Predicted": forecasts})

    # =======================
    # STEP 8 — CHART
    # =======================
    st.subheader("📈 Actual + Fitted + Forecast vs Date")

    chart_df = pd.DataFrame({
        "Date": df[date_col],
        "Actual": y,
        "Fitted": fitted
    })

    chart_df.set_index("Date", inplace=True)

    st.line_chart(chart_df)

    st.write("📌 Future Forecast")
    st.line_chart(forecast_df.set_index("Date"))

else:
    st.info("👆 Upload a CSV to begin")
