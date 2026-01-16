import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import BayesianRidge, LassoCV, RidgeCV
from sklearn.metrics import mean_absolute_percentage_error

from statsmodels.api import add_constant
from statsmodels.stats.outliers_influence import variance_inflation_factor

st.title("📌 Feature Selection → VIF → Multiple Models Evaluation Dashboard")

uploaded_file = st.file_uploader("📤 Upload your CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df = df.rename(columns=lambda x: x.strip())
    df = df.fillna(method='ffill').fillna(0)

    st.subheader("🔍 Data Preview")
    st.write(df.head())

    columns = df.columns[1:]  # skip date or index col
    target = st.selectbox("🎯 Select target stock", columns)

    # =============================
    # RANDOM FOREST FEATURE IMPORTANCE
    # =============================
    st.subheader("🌲 Step 1 — Random Forest Feature Importance")

    X = df[columns].select_dtypes(include=[np.number]).drop(columns=[target], errors='ignore')
    y = df[target]

    rf = RandomForestRegressor(n_estimators=300, random_state=42)
    rf.fit(X, y)

    importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': rf.feature_importances_
    }).sort_values(by='Importance', ascending=False)

    st.write("📌 Top 30 most important variables")
    st.dataframe(importance.head(30))

    top_features = importance.head(30)['Feature'].tolist()

    # =============================
    # VIF ON TOP FEATURES
    # =============================
    st.subheader("🧮 Step 2 — VIF Filtering")

    X_top = df[top_features]
    X_vif_const = add_constant(X_top)

    vif_df = pd.DataFrame()
    vif_df["Feature"] = X_vif_const.columns
    vif_df["VIF"] = [variance_inflation_factor(X_vif_const.values, i) 
                      for i in range(X_vif_const.shape[1])]
    st.write(vif_df)

    vif_cutoff = st.slider("VIF cutoff", 2.0, 20.0, 10.0)
    selected = vif_df[vif_df["VIF"] < vif_cutoff]["Feature"].tolist()

    if 'const' in selected:
        selected.remove('const')

    st.success(f"Using {len(selected)} features after VIF filtering")

    X_features = df[selected]

    # =============================
    # HEATMAP AFTER FEATURE SELECTION
    # =============================
    st.subheader("🌡 Step 3 — Correlation Heatmap (Selected Variables)")
    fig, ax = plt.subplots(figsize=(10, 6))

    # Fix NaN/Inf issue
    corr = X_features.corr().replace([np.inf, -np.inf], np.nan).fillna(0)

    sns.heatmap(corr, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

    # =============================
    # TRAIN-TEST SPLIT
    # =============================
    split = int(len(df) * 0.9)
    X_train = X_features.iloc[:split]
    X_test = X_features.iloc[split:]
    y_train = y.iloc[:split]
    y_test = y.iloc[split:]

    # =============================
    # SCATTER PLOTS
    # =============================
    st.subheader("📍 Step 4 — Scatter Plots vs Target")
    for col in selected[:5]:   # show only first 5 to keep UI clean
        fig2, ax2 = plt.subplots()
        ax2.scatter(df[col], y, alpha=0.5)
        ax2.set_xlabel(col)
        ax2.set_ylabel(target)
        st.pyplot(fig2)

    # =============================
    # MODELS TO TEST
    # =============================
    st.subheader("🤖 Step 5 — Model Training + MAPE Comparison")

    models = {
        "Bayesian Ridge": BayesianRidge(),
        "LassoCV": LassoCV(cv=5, random_state=42),
        "RidgeCV": RidgeCV(cv=5),
        "RandomForestRegressor": RandomForestRegressor(n_estimators=300, random_state=42)
    }

    results = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)

        train_mape = mean_absolute_percentage_error(y_train, train_pred) * 100
        test_mape = mean_absolute_percentage_error(y_test, test_pred) * 100

        results.append([name, round(train_mape,2), round(test_mape,2)])

    perf_df = pd.DataFrame(results, columns=["Model", "Train MAPE %", "Test MAPE %"])
    st.dataframe(perf_df)

    # Show best model
    best = perf_df.loc[perf_df["Test MAPE %"].idxmin()]
    st.success(f"🏆 Best model: {best['Model']} with Test MAPE = {best['Test MAPE %']}%")

else:
    st.info("👆 Upload a CSV to begin.")

