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
    auc
)
from mlxtend.frequent_patterns import apriori, association_rules

import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
import joblib


# --------------------------------------------------------
# Page Setup
# --------------------------------------------------------
st.set_page_config(
    page_title="DNA-Based Wellness Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🌿 DNA Wellness — Advanced Analytics Dashboard")
st.markdown("""
A fully interactive dashboard for:
- **EDA**
- **Association Rule Mining**
- **Clustering & Segmentation**
- **Regression (4 Models)**
- **Classification (Decision Tree & Gradient Boosting)**
""")

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
colC.metric("Numeric Features", df.select_dtypes(include='number').shape[1])
colD.metric("Categorical Features", df.select_dtypes(include='object').shape[1])

st.markdown("---")

# --------------------------------------------------------
# TABS UI
# --------------------------------------------------------
tab_eda, tab_assoc, tab_cluster, tab_reg, tab_clf = st.tabs([
    "📊 EDA", "🛒 Association Rules", "🧭 Clustering", "📉 Regression", "🎯 Classification"
])

# --------------------------------------------------------
# 1. EDA
# -
