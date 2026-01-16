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

st.title("✨ Auto Feature Selection → OLS → Forecast Dashboard")

uploaded_file = st.file_uploader("📤 Upload CSV file", type=["csv"])

if uploaded_file:
    # LOAD DATA
    df = pd.read_csv(uploaded_file)
    df = df.rename(columns=lambda x: x.strip())
    df = df.fillna(method='ffill').fillna(0)

    date_col = df.columns[0]
    columns = df.columns[1:]

    st.subheader("🔍 Data Preview")
    st.write(df.head())

    target = st.selectbox("🎯 Select target stock", columns)

    # STEP 1 — FEATURE IMPORTANCE
    st.subheader("🌲 Step 1 — Random Forest Feature Ranking")

    X = df[columns].select_dtypes(include=[np.number]).drop(columns=[target], errors='ignore')
    y = df[target]

    rf = RandomForestRegressor(n_estimators=300, random_state=42)
    rf.fit(X, y)

    importance = pd.DataFrame({
        "Feature": X.columns,
        "Importance": rf.feature_importances_
    }).sort_values(by="Importance", ascending=False)

    st.write("📌 Top 30 most important variables")
    st.dataframe(importance.head(30))

    top_features = importance.head(30)['Feature'].tolist()

    # STEP 2 — VIF FILTER
    st.subheader("🧮 Step 2 — VIF Filter")

    X_vif_raw = add_constant(df[top_features])
    vif_df = pd.DataFrame()
    vif_df["Feature"] = X_vif_raw.columns
    vif_df["VIF"] = [variance_inflation_factor(X_vif_raw.values, i) 
                      for i in range(X_vif_raw.shape[1])]
    st.write(vif_df)

    vif_cutoff = st.slider("VIF cutoff", 2.0, 20.0, 10.0)
    selected = vif_df[vif_df["VIF"] < vif_cutoff]["Feature"].tolist()

    if "const" in selected:
        selected.remove("const")

    st.success(f"✔ {len(selected)} features kept after VIF filtering")

    X_features = df[selected]

    # STEP 3 — TRAIN-TEST SPLIT
    split = int(len(df) * 0.9)
    X_train = X_features.iloc[:split]
    X_test = X_features.iloc[split:]
    y_train = y.iloc[:split]
    y_test = y.iloc[split:]

    # STEP 4 — MODEL COMPARISON
    st.subheader("🤖 Step 3 — Compare Models (Train vs Test MAPE)")

    models = {
        "Bayesian Ridge": BayesianRidge(),
        "LassoCV": LassoCV(cv=5, random_state=42),
        "RidgeCV": RidgeCV(cv=5),
        "RandomForestRegressor": RandomForestRegressor(n_estimators=300, random_state=42)
    }

    results = []
    best_model_name = None
    best_test_mape = 1e9

    for name, model in models.items():
        model.fit(X_train, y_train)
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)

        train_mape = mean_absolute_percentage_error(y_train, train_pred)*100
        test_mape = mean_absolute_percentage_error(y_test, test_pred)*100

        results.append([name, round(train_mape,2), round(test_mape,2)])

        if test_mape < best_test_mape:
            best_test_mape = test_mape
            best_model_name = name

    perf_df = pd.DataFrame(results, columns=["Model","Train MAPE %","Test MAPE %"])
    st.dataframe(perf_df)
    st.success(f"🏆 Best model selected: {best_model_name}")

    # STEP 5 — FINAL OLS & AUTO DROP p>0.05
    st.subheader("🧾 Step 4 — OLS Regression (p < 0.05 only)")

    X_reg = add_constant(X_features)
    ols = OLS(y, X_reg).fit()

    pvals = ols.pvalues.drop("const", errors='ignore')
    sig_features = pvals[pvals < 0.05].index.tolist()

    if len(sig_features) == 0:
        st.error("❌ No variables significant at p < 0.05")
    else:
        st.write("✔ Significant variables:", sig_features)

        X_final = add_constant(df[sig_features])
        final_ols = OLS(y, X_final).fit()
        st.write(final_ols.summary())

        fitted = final_ols.predict(X_final)

        # STEP 6 — FUTURE FORECAST
        st.subheader("🔮 Step 5 — Forecast")

        future_steps = st.slider("Periods ahead:", 1, 24, 6)
        last_row = df[sig_features].iloc[-1:].copy()

        forecasts = []
        for i in range(future_steps):
            future_in = add_constant(last_row, has_constant='add')
            pred = final_ols.predict(future_in)[0]
            forecasts.append(pred)

        last_date = pd.to_datetime(df[date_col].iloc[-1])
        future_dates = pd.date_range(start=last_date, periods=future_steps+1, closed='right')

        forecast_df = pd.DataFrame({"Date": future_dates, "Predicted": forecasts})

        # STEP 7 — CHART
        st.subheader("📈 Final: Actual + Fitted + Forecast")

        chart_df = pd.DataFrame({
            "Date": df[date_col],
            "Actual": y,
            "Fitted": fitted
        }).set_index("Date")

        st.line_chart(chart_df)
        st.line_chart(forecast_df.set_index("Date"))

else:
    st.info("👆 Upload CSV to begin")
