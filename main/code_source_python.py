import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from catboost import CatBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
import plotly.express as px
import plotly.graph_objects as go
import os

# Page config
st.set_page_config(page_title="Intrusion Detection Dashboard", layout="wide")
st.title("🛡️ Intrusion Detection Dashboard")

st.markdown("""
Bienvenue dans le tableau de bord d’analyse des intrusions réseau. 
Vous pouvez explorer les données, entraîner différents modèles de machine learning, et visualiser les performances.
""")

# === CHARGEMENT DES DONNÉES ===
@st.cache_data
def load_data():
    file_path = os.path.join(os.path.dirname(__file__), '../datasets/dataset.csv')
    return pd.read_csv(file_path)

dataset = load_data()

# === PRÉTRAITEMENT ===
def preprocess_data(dataset, apply_smote=False):
    dataset = dataset.copy()
    dataset['Attack Type'] = dataset['Attack Type'].astype('category').cat.codes
    for col in dataset.select_dtypes(include=['object']).columns:
        dataset[col] = dataset[col].astype('category').cat.codes

    dataset['total_bytes'] = dataset['src_bytes'] + dataset['dst_bytes']
    dataset['bytes_ratio'] = dataset['src_bytes'] / (dataset['dst_bytes'] + 1)
    dataset['log_src_bytes'] = np.log(dataset['src_bytes'] + 1)
    dataset['log_dst_bytes'] = np.log(dataset['dst_bytes'] + 1)

    X = dataset.drop(columns=['target', 'Attack Type'], errors='ignore')
    y = dataset['Attack Type']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    if apply_smote:
        smote = SMOTE(random_state=42)
        X_train, y_train = smote.fit_resample(X_train, y_train)

    return X_train, X_test, y_train, y_test

# === ENTRAÎNEMENT DES MODÈLES ===
@st.cache_resource
def train_model(model_type, X_train, y_train, X_test, y_test):
    if model_type == "Random Forest":
        model = RandomForestClassifier(random_state=42, max_depth=20, min_samples_leaf=1, min_samples_split=5, n_estimators=100)
    elif model_type == "CatBoost":
        model = CatBoostClassifier(verbose=0, random_state=42, depth=6, iterations=100, l2_leaf_reg=3, learning_rate=0.1)
    elif model_type == "XGBoost":
        model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
    elif model_type == "Decision Tree":
        model = DecisionTreeClassifier(random_state=42)
    elif model_type == "Extra Trees":
        model = ExtraTreesClassifier(random_state=42)
    elif model_type == "LightGBM":
        model = LGBMClassifier(random_state=42, verbose=-1)
    elif model_type == "MLP":
        model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=100, random_state=42)
    else:
        raise ValueError("Modèle non reconnu.")

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return model, confusion_matrix(y_test, y_pred), classification_report(y_test, y_pred, output_dict=True), y_pred

# === EXTRACTION SCORE ===
def extract_f1(report, class_idx=4):
    return float(report.get(str(class_idx), {}).get('f1-score', 0))

# === INTERFACE ===
tab1, tab2, tab3, tab4 = st.tabs(["📊 Exploration", "⚙️ Modélisation", "📈 Évaluation", "📌 Importance"])

with tab1:
    st.subheader("Exploration des données")
    st.write(f"Nombre de lignes : {dataset.shape[0]}")
    st.write(f"Nombre de colonnes : {dataset.shape[1]}")
    st.dataframe(dataset.head())

    st.subheader("Statistiques descriptives")
    stats = dataset.describe().T[['mean', '50%', 'std', 'min', 'max']].rename(columns={'50%': 'median'})
    st.dataframe(stats)

    st.subheader("Distribution des types d'attaques")
    attack_counts = dataset['Attack Type'].value_counts().rename_axis('Attack_Type').reset_index(name='count')
    st.dataframe(attack_counts)

    fig = px.bar(attack_counts, x='Attack_Type', y='count', title="Distribution des types d'attaque")
    st.plotly_chart(fig)

    st.subheader("Heatmap de corrélation")
    corr = dataset.corr(numeric_only=True)
    fig2 = go.Figure(data=go.Heatmap(z=corr.values, x=corr.columns, y=corr.columns, colorscale='RdBu', zmin=-1, zmax=1))
    fig2.update_layout(title="Matrice de corrélation")
    st.plotly_chart(fig2)

with tab2:
    st.subheader("Entraînement du modèle")
    model_choice = st.selectbox("Choisir un modèle", ["Random Forest", "CatBoost", "XGBoost", "Decision Tree", "Extra Trees", "LightGBM", "MLP"])
    apply_smote = st.checkbox("Utiliser SMOTE pour rééquilibrer les classes")

    if st.button("🎯 Lancer l'entraînement"):
        X_train, X_test, y_train, y_test = preprocess_data(dataset, apply_smote)
        model, conf_matrix, report, y_pred = train_model(model_choice, X_train, y_train, X_test, y_test)
        st.session_state.update({
            "model": model,
            "conf_matrix": conf_matrix,
            "report": report,
            "model_type": model_choice,
        })
        st.success(f"Modèle {model_choice} entraîné avec succès !")

with tab3:
    st.subheader("Évaluation du modèle")

    if "model" in st.session_state:
        st.markdown(f"### Matrice de confusion : {st.session_state['model_type']}")
        fig = go.Figure(data=go.Heatmap(z=st.session_state['conf_matrix'], colorscale='Blues'))
        fig.update_layout(width=600, height=500)
        st.plotly_chart(fig)

        st.markdown("### Rapport de classification")
        rep_df = pd.DataFrame(st.session_state['report']).T
        st.dataframe(rep_df[['precision', 'recall', 'f1-score', 'support']])

        f1_class4 = extract_f1(st.session_state['report'], class_idx=4)
        fig_bar = go.Figure(data=[go.Bar(x=[st.session_state['model_type']], y=[f1_class4])])
        fig_bar.update_layout(title="F1-score - Classe 4 (minoritaire)", yaxis_range=[0, 1])
        st.plotly_chart(fig_bar)

with tab4:
    st.subheader("Importance des variables")
    if "model" in st.session_state and hasattr(st.session_state['model'], 'feature_importances_'):
        X_cols = dataset.drop(columns=['target', 'Attack Type'], errors='ignore').columns
        importances = st.session_state['model'].feature_importances_
        df_imp = pd.DataFrame({'Feature': X_cols, 'Importance': importances}).sort_values(by='Importance', ascending=False)
        st.dataframe(df_imp.head(10))

        fig_imp = px.bar(df_imp.head(10), x='Importance', y='Feature', orientation='h', title="Top 10 features")
        st.plotly_chart(fig_imp)
    else:
        st.info("Le modèle sélectionné ne fournit pas d'importance des variables.")

# Footer
st.markdown("---")
st.caption("Développé avec ❤️ par Yohann Yendi | Intrusion Detection 2025")
