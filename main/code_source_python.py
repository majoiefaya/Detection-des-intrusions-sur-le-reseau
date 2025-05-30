import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
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
import re
import uuid
import os
# Set page configuration
st.set_page_config(page_title="Intrusion Detection Dashboard", layout="wide")

# Title and description
st.title("Intrusion Detection Dashboard")
st.markdown("""
This application allows you to explore the intrusion detection dataset, train machine learning models, and visualize their performance.
Use the tabs below to navigate through different sections of the analysis.
""")

# Load dataset
@st.cache_data
def load_data():
    # Replace with the actual path to your dataset
    file_path = os.path.join(os.path.dirname(__file__), '../datasets/dataset.csv')
    dataset = pd.read_csv(file_path)  # Update this path as needed
    return dataset

dataset = load_data()

# Preprocessing function
def preprocess_data(dataset, apply_smote=False):
    dataset = dataset.copy()
    
    # Encode categorical columns
    categorical_columns = dataset.select_dtypes(include=['object']).columns
    for col in categorical_columns:
        dataset[col] = dataset[col].astype('category').cat.codes
    
    # Create derived features
    dataset['total_bytes'] = dataset['src_bytes'] + dataset['dst_bytes']
    dataset['bytes_ratio'] = dataset['src_bytes'] / (dataset['dst_bytes'] + 1)
    dataset['log_src_bytes'] = (dataset['src_bytes'] + 1).apply(np.log)
    dataset['log_dst_bytes'] = (dataset['dst_bytes'] + 1).apply(np.log)
    
    # Encode target
    dataset['Attack Type'] = dataset['Attack Type'].astype('category').cat.codes
    
    # Separate features and target
    X = dataset.drop(columns=['target', 'Attack Type'])
    y = dataset['Attack Type']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    if apply_smote:
        smote = SMOTE(random_state=42)
        X_train, y_train = smote.fit_resample(X_train, y_train)
    
    return X_train, X_test, y_train, y_test

# Function to train and evaluate models
@st.cache_resource
def train_model(model_type, X_train, y_train, X_test, y_test, use_smote=False):
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
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred, output_dict=True)
    
    return model, conf_matrix, class_report, y_pred

# Function to extract f1-scores
def extract_f1_scores(report, class_idx=None):
    if class_idx is None:
        return float(report['macro avg']['f1-score'])
    return float(report[str(class_idx)]['f1-score'])

# Create tabs for different sections
tab1, tab2, tab3, tab4 = st.tabs(["Data Exploration", "Model Training", "Model Performance", "Feature Importance"])

with tab1:
    st.header("Data Exploration")
    
    # Dataset overview
    st.subheader("Dataset Overview")
    st.write(f"Number of rows: {dataset.shape[0]}")
    st.write(f"Number of columns: {dataset.shape[1]}")
    st.dataframe(dataset.head())
    
    # Descriptive statistics
    st.subheader("Descriptive Statistics")
    desc_stats = dataset.describe().T[['mean', '50%', 'std', 'min', 'max']].rename(columns={'50%': 'median'})
    st.dataframe(desc_stats)
    
    # Attack type distribution
    st.subheader("Attack Type Distribution")
    attack_counts = dataset['Attack Type'].value_counts(normalize=True).rename('proportion').to_frame().join(dataset['Attack Type'].value_counts().rename('count'))
    st.dataframe(attack_counts)
    
    # Plot attack type distribution
    fig = px.bar(attack_counts.reset_index(), x='index', y='count', title="Attack Type Distribution", labels={'index': 'Attack Type', 'count': 'Count'})
    st.plotly_chart(fig)
    
    # Correlation heatmap
    st.subheader("Correlation Heatmap")
    corr_matrix = dataset.corr(numeric_only=True)
    fig = go.Figure(data=go.Heatmap(z=corr_matrix.values, x=corr_matrix.columns, y=corr_matrix.columns, colorscale='RdBu', zmin=-1, zmax=1))
    fig.update_layout(title="Correlation Heatmap", width=800, height=600)
    st.plotly_chart(fig)

with tab2:
    st.header("Model Training")
    
    # Model selection
    model_options = ["Random Forest", "CatBoost", "XGBoost", "Decision Tree", "Extra Trees", "LightGBM", "MLP"]
    selected_model = st.selectbox("Select Model", model_options)
    
    # SMOTE option
    use_smote = st.checkbox("Apply SMOTE for class imbalance", value=False)
    
    # Train button
    if st.button("Train Model"):
        with st.spinner("Training model..."):
            X_train, X_test, y_train, y_test = preprocess_data(dataset, apply_smote=use_smote)
            model, conf_matrix, class_report, y_pred = train_model(selected_model, X_train, y_train, X_test, y_test, use_smote)
            
            st.session_state['model'] = model
            st.session_state['conf_matrix'] = conf_matrix
            st.session_state['class_report'] = class_report
            st.session_state['y_pred'] = y_pred
            st.session_state['model_type'] = selected_model
            st.session_state['use_smote'] = use_smote
            st.success(f"{selected_model} trained successfully!")

with tab3:
    st.header("Model Performance")
    
    if 'model' in st.session_state:
        st.subheader(f"Performance of {st.session_state['model_type']}")
        
        # Confusion matrix
        st.write("Confusion Matrix")
        fig = go.Figure(data=go.Heatmap(z=st.session_state['conf_matrix'], x=[f'Pred {i}' for i in range(st.session_state['conf_matrix'].shape[1])], 
                                        y=[f'True {i}' for i in range(st.session_state['conf_matrix'].shape[0])], 
                                        colorscale='Blues', text=st.session_state['conf_matrix'], texttemplate="%{text}"))
        fig.update_layout(title="Confusion Matrix", width=600, height=600)
        st.plotly_chart(fig)
        
        # Classification report
        st.write("Classification Report")
        class_report_df = pd.DataFrame(st.session_state['class_report']).T
        st.dataframe(class_report_df[['precision', 'recall', 'f1-score', 'support']])
        
        # F1-score comparison for class 4
        st.subheader("F1-Score for Class 4 (Minority Class)")
        f1_class4 = extract_f1_scores(st.session_state['class_report'], class_idx=4)
        fig = go.Figure(data=[go.Bar(x=[st.session_state['model_type']], y=[f1_class4], text=[f"{f1_class4:.2f}"], textposition='auto')])
        fig.update_layout(title="F1-Score for Class 4", yaxis_title="F1-Score", yaxis_range=[0, 1.1])
        st.plotly_chart(fig)

with tab4:
    st.header("Feature Importance")
    
    if 'model' in st.session_state and hasattr(st.session_state['model'], 'feature_importances_'):
        importances = st.session_state['model'].feature_importances_
        features = dataset.drop(columns=['target', 'Attack Type']).columns
        importance_df = pd.DataFrame({'Feature': features, 'Importance': importances}).sort_values(by='Importance', ascending=False)
        
        st.write("Top 10 Most Important Features")
        st.dataframe(importance_df.head(10))
        
        # Plot feature importance
        fig = px.bar(importance_df.head(10), x='Importance', y='Feature', orientation='h', title="Top 10 Feature Importance")
        st.plotly_chart(fig)
    else:
        st.write("Feature importance is only available for models that support it (e.g., Random Forest, CatBoost, XGBoost, Extra Trees).")

# Footer
st.markdown("---")
st.markdown("Developed with Streamlit | Dataset: Intrusion Detection | © 2025")