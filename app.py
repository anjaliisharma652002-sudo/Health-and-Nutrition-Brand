# app.py
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from mlxtend.frequent_patterns import apriori, association_rules
import plotly.express as px
import matplotlib.pyplot as plt
import joblib
import io

st.set_page_config(layout="wide", page_title="DNA Wellness Analytics Dashboard")

# ---------------------------
# Utility functions
# ---------------------------
@st.cache_data
def load_data():
    # Attempt to load the two CSVs (prefer first)
    p1 = Path("data/dna_survey_synthetic_600.csv")
    p2 = Path("data/DNA_Wellness_Survey_Synthetic_Data.csv")
    if p1.exists():
        df = pd.read_csv(p1)
    elif p2.exists():
        df = pd.read_csv(p2)
    else:
        st.error("No data file found. Upload data/dna_survey_synthetic_600.csv or data/DNA_Wellness_Survey_Synthetic_Data.csv")
        st.stop()
    return df

def safe_columns(df):
    return df.columns.tolist()

def convert_bucket_to_numeric(x):
    # Helper to convert price buckets like "200-400" or "<200" to numeric midpoints
    if pd.isna(x): 
        return np.nan
    s = str(x)
    if s.startswith("<"):
        try:
            return float(s.replace("<","").strip()) * 0.5
        except:
            return np.nan
    if "+" in s:
        try:
            return float(s.replace("+","").replace("AED","").strip()) 
        except:
            return np.nan
    if "-" in s:
        parts = s.split("-")
        try:
            return (float(parts[0]) + float(parts[1])) / 2.0
        except:
            return np.nan
    try:
        return float(s)
    except:
        return np.nan

# ---------------------------
# Load dataset
# ---------------------------
st.title("DNA-based Wellness — Analytics Dashboard")
st.markdown("Interactive dashboard: EDA | Association Rules | Clustering | Regression | Classification")

df = load_data()
st.sidebar.header("Controls")

# allow user to upload alternative CSV
uploaded = st.sidebar.file_uploader("Upload CSV (optional) - will override sample", type=["csv"])
if uploaded:
    df = pd.read_csv(uploaded)
    st.sidebar.success("Uploaded file loaded")

st.sidebar.markdown(f"Rows: **{df.shape[0]}** | Columns: **{df.shape[1]}**")

# Basic EDA
if st.sidebar.checkbox("Show raw data", value=False):
    st.subheader("Raw data (top 100 rows)")
    st.dataframe(df.head(100))

st.markdown("## Quick EDA")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Rows", df.shape[0])
col2.metric("Columns", df.shape[1])
col3.metric("Numeric cols", df.select_dtypes(include='number').shape[1])
col4.metric("String cols", df.select_dtypes(include='object').shape[1])

# Basic column list (helpful for selecting targets)
st.markdown("**Columns detected:**")
st.write(safe_columns(df))

st.markdown("---")

# ---------------------------
# Association Rule Mining
# ---------------------------
st.header("Association Rule Mining (Apriori)")
st.write("This section expects multi-select / binary item columns. If your CSV has columns like 'Supplements_Multivitamins' with 0/1, they will be used.")

# Auto-detect binary item columns (0/1):
binary_cols = [c for c in df.columns if set(df[c].dropna().unique()) <= {0,1}]
st.write("Auto-detected binary item columns (0/1):", binary_cols[:50])

# Option to enter columns manually (comma separated)
manual_cols = st.text_input("If items not detected automatically, enter column names (comma separated):", value="")
if manual_cols:
    user_cols = [c.strip() for c in manual_cols.split(",") if c.strip() in df.columns]
else:
    user_cols = binary_cols

min_support = st.slider("Min support", 0.01, 0.5, 0.02, 0.01)
min_confidence = st.slider("Min confidence", 0.1, 1.0, 0.3, 0.05)
min_lift = st.slider("Min lift", 0.5, 5.0, 1.0, 0.1)

if st.button("Run Apriori"):
    if not user_cols:
        st.warning("No item columns selected for association mining. Provide item columns or ensure binary columns exist.")
    else:
        trans = df[user_cols].fillna(0).astype(int)
        with st.spinner("Running apriori..."):
            frequent_itemsets = apriori(trans, min_support=min_support, use_colnames=True)
            if frequent_itemsets.empty:
                st.warning("No frequent itemsets for the chosen min_support.")
            else:
                rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
                rules = rules[ rules['lift'] >= min_lift ]
                if rules.empty:
                    st.warning("No association rules after applying thresholds.")
                else:
                    rules_sorted = rules.sort_values(['lift','confidence'], ascending=False)
                    rules_sorted['antecedents'] = rules_sorted['antecedents'].apply(lambda x: ', '.join(list(x)))
                    rules_sorted['consequents'] = rules_sorted['consequents'].apply(lambda x: ', '.join(list(x)))
                    st.subheader("Top association rules")
                    st.dataframe(rules_sorted[['antecedents','consequents','support','confidence','lift']].head(30))

st.markdown("---")

# ---------------------------
# Clustering
# ---------------------------
st.header("Clustering & Segmentation (KMeans)")

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
st.write("Numeric columns available for clustering (choose at least 2):", numeric_cols[:50])
chosen_features = st.multiselect("Select numeric features for clustering (defaults chosen)", numeric_cols, default=numeric_cols[:4])

k = st.slider("Choose K (n_clusters)", 2, 8, 3)
if st.button("Run Clustering"):
    if len(chosen_features) < 2:
        st.warning("Please select at least 2 numeric features for clustering.")
    else:
        X = df[chosen_features].fillna(0).values
        scaler = StandardScaler().fit(X)
        Xs = scaler.transform(X)
        # elbow plot
        sse = []
        for i in range(2,9):
            km = KMeans(n_clusters=i, random_state=42, n_init=10).fit(Xs)
            sse.append(km.inertia_)
        fig_elbow = px.line(x=list(range(2,9)), y=sse, markers=True, title="Elbow plot (SSE vs K)")
        st.plotly_chart(fig_elbow)
        # run KM
        km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(Xs)
        labels = km.labels_
        df['cluster'] = labels
        st.write("Cluster sizes:")
        st.write(df['cluster'].value_counts().sort_index())
        # PCA 2D
        pca = PCA(n_components=2, random_state=42)
        coords = pca.fit_transform(Xs)
        fig = px.scatter(x=coords[:,0], y=coords[:,1], color=df['cluster'].astype(str),
                         title="Cluster scatter (PCA 2D)", labels={'x':'PC1','y':'PC2'})
        st.plotly_chart(fig)
        # cluster profiles
        st.subheader("Cluster profiles (means of chosen features)")
        st.dataframe(df.groupby('cluster')[chosen_features].mean().round(3))

st.markdown("---")

# ---------------------------
# Regression: 4 models
# ---------------------------
st.header("Regression — Predict Willingness to Pay (Regression)")
st.write("Select the column that represents spending/willingness to pay. The app will attempt to convert buckets to numeric midpoints if needed.")

# Default attempts to find likely price column
candidate_price_cols = [c for c in df.columns if 'price' in c.lower() or 'pay' in c.lower() or 'willingness' in c.lower() or 'spend' in c.lower()]
if candidate_price_cols:
    default_price = candidate_price_cols[0]
else:
    default_price = df.columns[0]

target_col = st.selectbox("Choose target column for regression", options=df.columns.tolist(), index=df.columns.tolist().index(default_price))
test_size = st.slider("Test set fraction", 0.1, 0.4, 0.2, 0.05)

if st.button("Run regression models"):
    # prepare X,y
    df_reg = df.copy()
    y_raw = df_reg[target_col].apply(convert_bucket_to_numeric)
    # if too many NaNs after conversion, fallback to numeric values if present
    if y_raw.isna().mean() > 0.5:
        st.warning("Conversion to numeric produced many NaNs. Ensure target is numeric or bucket format. Showing fallback: use numeric columns only.")
        y_raw = None
    if y_raw is None:
        st.warning("Cannot run regression: target not convertible to numeric. Please choose another column.")
    else:
        # simple X: use numeric columns excluding target
        X = df_reg.select_dtypes(include=[np.number]).copy()
        if X.shape[1] < 1:
            st.warning("No numeric predictors available. Consider encoding categorical variables or choose a different target.")
        else:
            X = X.fillna(0)
            y = y_raw.fillna(y_raw.mean())
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
            models = {
                "Linear": LinearRegression(),
                "Ridge": Ridge(),
                "Lasso": Lasso(max_iter=5000),
                "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42)
            }
            results = []
            preds = {}
            for name, m in models.items():
                m.fit(X_train, y_train)
                y_pred = m.predict(X_test)
                r2 = r2_score(y_test, y_pred)
                rmse = mean_squared_error(y_test, y_pred, squared=False)
                mae = mean_absolute_error(y_test, y_pred)
                results.append({"Model": name, "R2": round(r2,4), "RMSE": round(rmse,4), "MAE": round(mae,4)})
                preds[name] = (y_test, y_pred)
                joblib.dump(m, f"models/{name}_regressor.pkl")
            res_df = pd.DataFrame(results).sort_values("R2", ascending=False)
            st.dataframe(res_df)
            best = res_df.iloc[0]['Model']
            st.write(f"Showing Actual vs Predicted for best model: **{best}**")
            y_t, y_p = preds[best]
            fig = px.scatter(x=y_t, y=y_p, labels={'x':'Actual','y':'Predicted'}, trendline="ols")
            st.plotly_chart(fig)

st.markdown("---")

# ---------------------------
# Classification: Decision Tree & Gradient Boosting
# ---------------------------
st.header("Classification — Predict Interest (Decision Tree, Gradient Boosting)")
# default label detection:
label_candidates = [c for c in df.columns if 'interest' in c.lower() or 'interested' in c.lower() or 'interest in' in c.lower()]
label_default = label_candidates[0] if label_candidates else df.columns[0]
label_col = st.selectbox("Select label column for classification (prefer Yes/No/Maybe or categorical):", options=df.columns.tolist(), index=df.columns.tolist().index(label_default))

if st.button("Run classifiers"):
    # prepare X,y: drop label, use numeric + simple dummies for small number of categories
    df_clf = df.copy()
    if label_col not in df_clf.columns:
        st.error("Label not found.")
    else:
        y = df_clf[label_col].astype(str)
        # simple mapping to binary if multi-class: map 'Yes'->1, 'No'->0, others->maybe mapped to 2 (multiclass)
        unique_vals = y.unique()
        mapping = {}
        if set(['Yes','No']).issubset(set(unique_vals)):
            mapping = {'Yes':1,'No':0}
            # map any other to 2
            for v in unique_vals:
                if v not in mapping:
                    mapping[v] = 2
        else:
            # fallback: factorize
            mapping = {v:i for i,v in enumerate(sorted(unique_vals))}
        y_num = y.map(mapping).fillna(0).astype(int)
        X = df_clf.drop(columns=[label_col])
        # convert categorical to dummies but limit cardinality
        cat_cols = X.select_dtypes(include='object').columns.tolist()
        for c in cat_cols:
            if X[c].nunique() <= 10:
                d = pd.get_dummies(X[c], prefix=c, drop_first=True)
                X = pd.concat([X.drop(columns=[c]), d], axis=1)
            else:
                # drop high-card columns
                X = X.drop(columns=[c])
        X = X.select_dtypes(include=[np.number]).fillna(0)
        if X.shape[1] < 1:
            st.warning("No usable numeric features for classification after encoding.")
        else:
            X_train, X_test, y_train, y_test = train_test_split(X, y_num, test_size=0.25, random_state=42)
            dt = DecisionTreeClassifier(max_depth=5, random_state=42)
            gbc = GradientBoostingClassifier(random_state=42)
            dt.fit(X_train, y_train)
            gbc.fit(X_train, y_train)
            ydt = dt.predict(X_test)
            ygbc = gbc.predict(X_test)
            st.subheader("Decision Tree classification report")
            st.text(classification_report(y_test, ydt))
            st.subheader("Gradient Boosting classification report")
            st.text(classification_report(y_test, ygbc))
            # confusion matrix (DT)
            cm = confusion_matrix(y_test, ydt)
            fig, ax = plt.subplots()
            ax.imshow(cm, cmap='Blues')
            ax.set_title("Decision Tree Confusion Matrix")
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            for (i, j), val in np.ndenumerate(cm):
                ax.text(j, i, int(val), ha='center', va='center', color='black')
            st.pyplot(fig)
            # ROC AUC if binary
            if len(np.unique(y_num)) == 2:
                yprob = gbc.predict_proba(X_test)[:,1]
                auc = roc_auc_score(y_test, yprob)
                st.write("Gradient Boosting ROC AUC:", round(auc,3))
                fpr, tpr, _ = roc_curve(y_test, yprob)
                fig2 = px.area(x=fpr, y=tpr, title=f'ROC curve (AUC={round(auc,3)})', labels={'x':'FPR','y':'TPR'})
                st.plotly_chart(fig2)
            joblib.dump(dt, "models/decision_tree_clf.pkl")
            joblib.dump(gbc, "models/gradient_boosting_clf.pkl")

st.markdown("---")

# ---------------------------
# Model download & processed CSV
# ---------------------------
st.header("Download processed CSV or models")
# small processed preview (fillna and convert buckets)
preview_df = df.copy().fillna("")
csv = preview_df.to_csv(index=False).encode('utf-8')
st.download_button("Download processed CSV", data=csv, file_name="processed_survey_preview.csv", mime="text/csv")

# download models if exist
models_folder = Path("models")
if not models_folder.exists():
    models_folder.mkdir()
# List models
model_files = list(models_folder.glob("*"))
if model_files:
    for m in model_files:
        with open(m, "rb") as f:
            st.download_button(label=f"Download {m.name}", data=f, file_name=m.name)
else:
    st.info("No trained model files in /models yet. Train regression/classification first to save models here.")

st.caption("App generated by assistant — adapt column names if your CSV uses different headers.")
