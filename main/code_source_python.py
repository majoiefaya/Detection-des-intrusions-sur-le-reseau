# Projet de Détection d'Intrusions Réseau avec Machine Learning
# =============================================================
# Ce projet a pour but de détecter automatiquement les attaques réseau à partir de données simulées.
# Il utilise plusieurs modèles de classification pour identifier les comportements suspects.
# Le dashboard Streamlit permet de :
# - Visualiser les données
# - Appliquer un prétraitement (encodage, SMOTE)
# - Sélectionner et entraîner différents modèles
# - Évaluer la performance (matrice de confusion, rapport, courbes)

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from imblearn.over_sampling import SMOTE
import io


st.set_page_config(page_title="Détection d'Intrusions Réseau", layout="wide")

@st.cache_data
# Chargement du dataset sélectionné ou par défaut
# ==============================================
def load_data(uploaded_file):
    if uploaded_file is not None:
        return pd.read_csv(uploaded_file)
    else:
        import os 
        file_path = os.path.join(os.path.dirname(__file__), '../datasets/dataset.csv')
        return pd.read_csv(file_path)

st.sidebar.header("📁 Chargement des données")
uploaded_file = st.sidebar.file_uploader("Choisissez un fichier CSV", type="csv")

dataset = load_data(uploaded_file)

# Interface utilisateur
st.title("🔐 Dashboard de Détection d'Intrusions Réseau")
st.markdown("Explorez le dataset, visualisez les tendances et entraînez plusieurs modèles de classification pour détecter les attaques réseau.")

# Exploration initiale
if st.checkbox("📄 Afficher les premières lignes du dataset"):
    st.dataframe(dataset.head())

if st.checkbox("📊 Répartition des types d'attaques"):
    fig, ax = plt.subplots()
    dataset['Attack Type'].value_counts().plot(kind='bar', ax=ax)
    ax.set_title("Répartition des types d'attaques")
    st.pyplot(fig)

if st.checkbox("📈 Statistiques descriptives"):
    st.dataframe(dataset.describe())

if st.checkbox("📉 Heatmap de corrélation"):
    corr_matrix = dataset.corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(15, 10))
    sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', ax=ax)
    st.pyplot(fig)

# Prétraitement
st.header("🛠️ Prétraitement et Modélisation")
# Encodage des variables catégorielles pour les rendre utilisables par les modèles ML
# Transformation de la cible pour le modèle de classification

dataset['Attack Type'] = dataset['Attack Type'].astype('category').cat.codes
for col in dataset.select_dtypes(include=['object']).columns:
    dataset[col] = dataset[col].astype('category').cat.codes

X = dataset.drop(columns=['target', 'Attack Type'], errors='ignore')
y = dataset['Attack Type']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Application de SMOTE si nécessaire pour équilibrer les classes minoritaires
if st.checkbox("🔁 Appliquer SMOTE"):
    smote = SMOTE(random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)

# Sélection du modèle par l'utilisateur
model_choice = st.selectbox("🧠 Choisissez un modèle", [
    'Random Forest', 'Extra Trees', 'Decision Tree', 'MLP', 'Logistic Regression',
    'SVC', 'XGBoost', 'LightGBM', 'CatBoost'])

if st.button("🚀 Entraîner le modèle"):
    if model_choice == 'Random Forest':
        model = RandomForestClassifier(random_state=42)
    elif model_choice == 'Extra Trees':
        model = ExtraTreesClassifier(random_state=42)
    elif model_choice == 'Decision Tree':
        model = DecisionTreeClassifier(random_state=42)
    elif model_choice == 'MLP':
        model = MLPClassifier(random_state=42, max_iter=300)
    elif model_choice == 'Logistic Regression':
        model = LogisticRegression(max_iter=300)
    elif model_choice == 'SVC':
        model = SVC()
    elif model_choice == 'XGBoost':
        model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
    elif model_choice == 'LightGBM':
        model = LGBMClassifier(random_state=42)
    elif model_choice == 'CatBoost':
        model = CatBoostClassifier(verbose=0, random_state=42)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Matrice de confusion pour observer les bonnes et mauvaises classifications
    st.subheader("📋 Matrice de confusion")
    conf_matrix = confusion_matrix(y_test, y_pred)
    fig_cm, ax_cm = plt.subplots()
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', ax=ax_cm)
    st.pyplot(fig_cm)

    # Rapport de classification : précision, rappel, f1-score
    st.subheader("📑 Rapport de classification")
    report = classification_report(y_test, y_pred, output_dict=True)
    st.dataframe(pd.DataFrame(report).transpose())

    # Importance des variables pour les modèles qui le permettent
    if hasattr(model, 'feature_importances_'):
        st.subheader("📌 Importance des variables")
        importances = model.feature_importances_
        features = X.columns
        df_imp = pd.DataFrame({'Feature': features, 'Importance': importances})
        st.dataframe(df_imp.sort_values(by="Importance", ascending=False).head(10))
        indices = np.argsort(importances)[::-1][:10]
        fig_imp, ax_imp = plt.subplots()
        ax_imp.barh(range(10), importances[indices][::-1])
        ax_imp.set_yticks(range(10))
        ax_imp.set_yticklabels(features[indices][::-1])
        ax_imp.set_title("Top 10 variables influentes")
        st.pyplot(fig_imp)

# Courbe d'apprentissage pour détecter un éventuel overfitting
if st.checkbox("📚 Afficher la courbe d'apprentissage (Random Forest seulement)"):
    model = RandomForestClassifier(random_state=42)
    train_sizes, train_scores, test_scores = learning_curve(
        model, X_train, y_train, cv=5, scoring='f1_weighted',
        train_sizes=np.linspace(0.1, 1.0, 5), n_jobs=-1)

    train_scores_mean = np.mean(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)

    fig_lc, ax_lc = plt.subplots()
    ax_lc.plot(train_sizes, train_scores_mean, 'o-', label='Entraînement')
    ax_lc.plot(train_sizes, test_scores_mean, 'o-', label='Validation')
    ax_lc.set_title("Courbe d'apprentissage - F1 pondéré")
    ax_lc.set_xlabel("Taille de l'ensemble d'entraînement")
    ax_lc.set_ylabel("Score F1")
    ax_lc.legend()
    st.pyplot(fig_lc)

    st.markdown("""
    **Interprétation :**
    Une courbe d'entraînement plate à 1.00 et un écart significatif avec la courbe de validation indique un surapprentissage.
    SMOTE et le réglage des hyperparamètres peuvent aider à le corriger.
    """)

# Conclusion
st.markdown("""
---
### 🧾 Conclusion

- Le modèle Random Forest, surtout après l'application de **SMOTE**, s'est révélé particulièrement performant.
- Le **f1-score de la classe minoritaire** est un bon indicateur de la capacité du modèle à détecter les attaques rares.
- CatBoost et XGBoost offrent aussi de très bons résultats avec une généralisation solide.
- La combinaison **analyse exploratoire + rééquilibrage + sélection de modèle** permet d'optimiser les performances en cybersécurité.

🔍 **En production**, ces modèles peuvent être intégrés à un système IDS (Intrusion Detection System) pour une surveillance active.
""")
