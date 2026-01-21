import streamlit as st
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import BayesianRidge, LassoCV, RidgeCV
from sklearn.metrics import mean_absolute_percentage_error

from statsmodels.api import OLS, add_constant
from statsmodels.stats.outliers_influence import variance_inflation_factor

st.title("📈 AI-Driven Stock Forecast & Feature Selection Dashboard")

uploaded_file = st.file_uploader("📤 Upload CSV file", type=["csv"])

if uploaded_file:
    # =====================
    # LOAD & CLEAN DATA
    # =====================
    df = pd.read_csv(uploaded_file)
    df = df.rename(columns=lambda x: x.strip())
    df = df.fillna(method="ffill").fillna(0)

    date_col = df.columns[0]
    columns = df.columns[1:]

    st.subheader("🔍 Data Preview")
    st.write(df.head())

    target = st.selectbox("🎯 Select target stock", columns)

    # =====================
    # STEP 1 — RF FEATURE IMPORTANCE
    # =====================
    X = df[columns].select_dtypes(include=[np.number]).drop(columns=[target], errors="ignore")
    y = df[target]

    rf = RandomForestRegressor(n_estimators=300, random_state=42)
    rf.fit(X, y)

    importance = pd.DataFrame({
        "Feature": X.columns,
        "Importance": rf.feature_importances_
    }).sort_values(by="Importance", ascending=False)

    st.subheader("🌲 Random Forest Feature Importance (Top 30)")
    st.dataframe(importance.head(30))

    top_features = importance.head(30)["Feature"].tolist()

    # =====================
    # STEP 2 — VIF FILTERING
    # =====================
    X_vif = add_constant(df[top_features])
    vif_df = pd.DataFrame({
        "Feature": X_vif.columns,
        "VIF": [variance_inflation_factor(X_vif.values, i)
                for i in range(X_vif.shape[1])]
    })

    st.subheader("🧮 VIF Table")
    st.write(vif_df)

    vif_cutoff = st.slider("VIF cutoff", 2.0, 20.0, 10.0)
    selected = vif_df[vif_df["VIF"] < vif_cutoff]["Feature"].tolist()
    if "const" in selected:
        selected.remove("const")

    st.success(f"Selected {len(selected)} variables after VIF")

    X_features = df[selected]

    # =====================
    # STEP 3 — TRAIN / TEST SPLIT
    # =====================
    split = int(len(df) * 0.9)
    X_train, X_test = X_features.iloc[:split], X_features.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    # =====================
    # STEP 4 — MODEL COMPARISON
    # =====================
    models = {
        "Bayesian Ridge": BayesianRidge(),
        "LassoCV": LassoCV(cv=5),
        "RidgeCV": RidgeCV(cv=5),
        "Random Forest": RandomForestRegressor(n_estimators=300, random_state=42)
    }

    results = []

    for name, model in models.items():
        model.fit(X_train, y_train)
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)

        results.append([
            name,
            round(mean_absolute_percentage_error(y_train, train_pred) * 100, 2),
            round(mean_absolute_percentage_error(y_test, test_pred) * 100, 2)
        ])

    perf_df = pd.DataFrame(results, columns=["Model", "Train MAPE %", "Test MAPE %"])
    st.subheader("🤖 Model Comparison")
    st.dataframe(perf_df)

    # =====================
    # STEP 5 — FINAL OLS + p-VALUE FILTER
    # =====================
    X_reg = add_constant(X_features)
    ols = OLS(y, X_reg).fit()

    pvals = ols.pvalues.drop("const", errors="ignore")
    sig_features = pvals[pvals < 0.05].index.tolist()

    if len(sig_features) == 0:
        st.error("❌ No significant variables (p < 0.05)")
    else:
        st.success("Significant variables:")
        st.write(sig_features)

        X_final = add_constant(df[sig_features])
        final_ols = OLS(y, X_final).fit()
        st.write(final_ols.summary())

        fitted = final_ols.predict(X_final)

        # =====================
        # STEP 6 — FUTURE FORECAST (FIXED)
        # =====================
        st.subheader("🔮 Future Forecast")

        future_steps = st.slider("Forecast periods ahead", 1, 24, 6)

        last_row = df[sig_features].iloc[-1:].copy()
        forecasts = []

        for _ in range(future_steps):
            future_in = add_constant(last_row, has_constant="add")
            pred = final_ols.predict(future_in).iloc[0]  # ✅ FIX
            forecasts.append(pred)

        last_date = pd.to_datetime(df[date_col].iloc[-1])
        future_dates = pd.date_range(start=last_date, periods=future_steps + 1, closed="right")

        forecast_df = pd.DataFrame({
            "Date": future_dates,
            "Forecast": forecasts
        }).set_index("Date")

        # =====================
        # STEP 7 — FINAL CHART
        # =====================
        st.subheader("📈 Actual vs Fitted vs Forecast")

        history_df = pd.DataFrame({
            "Date": df[date_col],
            "Actual": y,
            "Fitted": fitted
        }).set_index("Date")

        st.line_chart(history_df)
        st.line_chart(forecast_df)

else:
    st.info("👆 Upload a CSV to begin")
