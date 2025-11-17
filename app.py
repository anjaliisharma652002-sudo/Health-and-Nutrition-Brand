# app.py
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error,
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
)
from mlxtend.frequent_patterns import apriori, association_rules

import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns


# --------------------------------------------------------
# Page Setup
# --------------------------------------------------------
st.set_page_config(
    page_title="DNA-Based Wellness Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🌿 DNA Wellness — Advanced Analytics Dashboard")
st.markdown(
    """
A fully interactive dashboard for:
- **EDA**
- **Association Rule Mining**
- **Clustering & Segmentation**
- **Regression (4 Models)**
- **Classification (Decision Tree & Gradient Boosting)**
"""
)

# --------------------------------------------------------
# Load Data
# --------------------------------------------------------
@st.cache_data
def load_data():
    p1 = Path("data/dna_survey_synthetic_600.csv")
    p2 = Path("data/DNA_Wellness_Survey_Synthetic_Data.csv")

    if p1.exists():
        df = pd.read_csv(p1)
    elif p2.exists():
        df = pd.read_csv(p2)
    else:
        st.error("❌ No dataset found in `data/` folder.")
        st.stop()

    return df


df = load_data()

# Fix mixed-type numeric columns
for col in df.columns:
    try:
        df[col] = pd.to_numeric(df[col], errors="ignore")
    except:
        pass

uploaded = st.sidebar.file_uploader("📤 Upload Your CSV (Optional)", type=["csv"])
if uploaded:
    df = pd.read_csv(uploaded)
    st.sidebar.success("Custom CSV loaded!")

st.sidebar.write(f"📊 **Rows:** {df.shape[0]} | **Columns:** {df.shape[1]}")

# --------------------------------------------------------
# KPI ROW
# --------------------------------------------------------
colA, colB, colC, colD = st.columns(4)
colA.metric("Rows", df.shape[0])
colB.metric("Columns", df.shape[1])
colC.metric("Numeric Features", df.select_dtypes(include="number").shape[1])
colD.metric("Categorical Features", df.select_dtypes(include="object").shape[1])

st.markdown("---")

# --------------------------------------------------------
# TABS UI
# --------------------------------------------------------
tab_eda, tab_assoc, tab_cluster, tab_reg, tab_clf = st.tabs(
    ["📊 EDA", "🛒 Association Rules", "🧭 Clustering", "📉 Regression", "🎯 Classification"]
)

# --------------------------------------------------------
# 1. EDA
# --------------------------------------------------------
with tab_eda:
    st.subheader("✨ Exploratory Data Analysis")

    numeric_cols = df.select_dtypes(include=["number"]).columns
    cat_cols = df.select_dtypes(include=["object"]).columns

    st.markdown("### 🔍 Numeric Distributions")
    for col in numeric_cols[:4]:
        fig = px.histogram(df, x=col, nbins=30, color_discrete_sequence=["#636EFA"])
        fig.update_layout(title=f"Distribution of {col}")
        st.plotly_chart(fig, use_container_width=True)

    if len(numeric_cols) >= 2:
        st.markdown("### 🔥 Correlation Heatmap")
        fig = px.imshow(
            df[numeric_cols].corr(),
            text_auto=True,
            color_continuous_scale="Blues",
        )
        st.plotly_chart(fig, use_container_width=True)

    if len(cat_cols) > 0 and len(numeric_cols) > 0:
        st.markdown("### 🏷️ Category-wise Trends")
        ccol = st.selectbox("Choose a Categorical Column", cat_cols)
        ncol = st.selectbox("Choose a Numeric Column", numeric_cols)
        fig = px.box(
            df, x=ccol, y=ncol, color=ccol, color_discrete_sequence=px.colors.qualitative.Set3
        )
        st.plotly_chart(fig, use_container_width=True)

# --------------------------------------------------------
# 2. Association Rules
# --------------------------------------------------------
with tab_assoc:
    st.subheader("🛒 Association Rule Mining (Apriori)")

    binary_cols = [
        c for c in df.columns if set(df[c].dropna().unique()) <= {0, 1}
    ]

    st.write("Detected binary (0/1) columns:", binary_cols)

    min_support = st.slider("Min Support", 0.01, 0.5, 0.02)
    min_conf = st.slider("Min Confidence", 0.1, 1.0, 0.3)
    min_lift = st.slider("Min Lift", 0.5, 5.0, 1.0)

    if st.button("Run Apriori"):
        if not binary_cols:
            st.error("❌ No binary columns found.")
        else:
            df_bin = df[binary_cols].astype(int)
            freq = apriori(df_bin, min_support=min_support, use_colnames=True)
            rules = association_rules(freq, metric="confidence", min_threshold=min_conf)
            rules = rules[rules["lift"] >= min_lift]

            if rules.empty:
                st.warning("No rules found.")
            else:
                rules["antecedents"] = rules["antecedents"].apply(
                    lambda x: ", ".join(list(x))
                )
                rules["consequents"] = rules["consequents"].apply(
                    lambda x: ", ".join(list(x))
                )

                st.dataframe(rules.head(20))

                fig = px.scatter(
                    rules,
                    x="confidence",
                    y="lift",
                    size="support",
                    hover_data=["antecedents", "consequents"],
                    color="lift",
                    color_continuous_scale="Turbo",
                    title="💡 Association Rules Strength Visualization",
                )
                st.plotly_chart(fig, use_container_width=True)

# --------------------------------------------------------
# 3. Clustering
# --------------------------------------------------------
with tab_cluster:
    st.subheader("🧭 Customer Segmentation (KMeans)")

    num_cols = list(df.select_dtypes(include="number").columns)

    if len(num_cols) < 2:
        st.error("Need at least 2 numeric columns for clustering.")
    else:
        features = st.multiselect("Choose features", num_cols, default=num_cols[:3])
        k = st.slider("K Clusters", 2, 10, 3)

        if st.button("Run Clustering"):
            X = df[features].fillna(0)
            scaler = StandardScaler()
            Xs = scaler.fit_transform(X)

            # Elbow Plot
            sse = []
            for i in range(2, 10):
                km = KMeans(n_clusters=i, random_state=42, n_init="auto").fit(Xs)
                sse.append(km.inertia_)

            fig_elbow = px.line(
                x=list(range(2, 10)),
                y=sse,
                markers=True,
                title="Elbow Method (SSE vs K)",
            )
            st.plotly_chart(fig_elbow, use_container_width=True)

            # Final Clustering
            km = KMeans(n_clusters=k, random_state=42, n_init="auto").fit(Xs)
            df["cluster"] = km.labels_

            st.write("Cluster Sizes:")
            st.write(df["cluster"].value_counts())

            # PCA 2D scatter
            pca = PCA(n_components=2)
            coords = pca.fit_transform(Xs)

            fig = px.scatter(
                x=coords[:, 0],
                y=coords[:, 1],
                color=df["cluster"].astype(str),
                hover_data=df[features],
                title="🌐 PCA Cluster Visualization",
                color_discrete_sequence=px.colors.qualitative.Bold,
            )
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("### 📌 Cluster Profiles")
            st.dataframe(df.groupby("cluster")[features].mean())

# --------------------------------------------------------
# 4. Regression
# --------------------------------------------------------
with tab_reg:
    st.subheader("📉 Regression — Predicting Spending / Willingness to Pay")

    num_cols = df.select_dtypes(include="number").columns
    target = st.selectbox("Choose target column", num_cols)

    if st.button("Train Regression Models"):
        X = df[num_cols].drop(columns=[target]).fillna(0)
        y = df[target].fillna(df[target].mean())

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2
        )

        models = {
            "Linear": LinearRegression(),
            "Ridge": Ridge(),
            "Lasso": Lasso(max_iter=5000),
            "Random Forest": RandomForestRegressor(n_estimators=200),
        }

        results = []
        for name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            results.append(
                {
                    "Model": name,
                    "R2": r2_score(y_test, y_pred),
                    "RMSE": mean_squared_error(y_test, y_pred, squared=False),
                    "MAE": mean_absolute_error(y_test, y_pred),
                }
            )

        st.dataframe(pd.DataFrame(results).sort_values("R2", ascending=False))

        best = pd.DataFrame(results).sort_values("R2", ascending=False).iloc[0][
            "Model"
        ]

        st.subheader(f"🌟 Best Model — {best}")

        model = models[best]
        y_pred = model.predict(X_test)

        fig = px.scatter(
            x=y_test,
            y=y_pred,
            title="Actual vs Predicted",
            color_discrete_sequence=["#FF1493"],
        )
        fig.add_shape(
            type="line",
            x0=min(y_test),
            y0=min(y_test),
            x1=max(y_test),
            y1=max(y_test),
            line=dict(color="green", dash="dash"),
        )
        st.plotly_chart(fig, use_container_width=True)

# --------------------------------------------------------
# 5. Classification
# --------------------------------------------------------
with tab_clf:
    st.subheader("🎯 Classification — Predicting Interest")

    label_col = st.selectbox("Label Column", df.columns)

    if st.button("Train Classifiers"):
        y = df[label_col].astype("category").cat.codes
        X = df.select_dtypes(include="number").fillna(0)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25
        )

        dt = DecisionTreeClassifier(max_depth=5)
        gbc = GradientBoostingClassifier()

        dt.fit(X_train, y_train)
        gbc.fit(X_train, y_train)

        y_pred_dt = dt.predict(X_test)
        y_pred_gbc = gbc.predict(X_test)

        st.markdown("### 🧪 Decision Tree Report")
        st.text(classification_report(y_test, y_pred_dt))

        st.markdown("### 🚀 Gradient Boosting Report")
        st.text(classification_report(y_test, y_pred_gbc))

        cm = confusion_matrix(y_test, y_pred_gbc)
        fig = px.imshow(
            cm,
            text_auto=True,
            color_continuous_scale="Purples",
            title="Confusion Matrix — Gradient Boosting",
        )
        st.plotly_chart(fig, use_container_width=True)

        if len(np.unique(y)) == 2:
            probs = gbc.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, probs)
            auc_score = auc(fpr, tpr)

            fig = px.area(
                x=fpr,
                y=tpr,
                title=f"ROC Curve (AUC = {auc_score:.3f})",
                labels={
                    "x": "False Positive Rate",
                    "y": "True Positive Rate",
                },
                color_discrete_sequence=["#FF5733"],
            )
            st.plotly_chart(fig, use_container_width=True)
