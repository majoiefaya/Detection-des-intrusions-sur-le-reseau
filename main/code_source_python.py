# -*- coding: utf-8 -*-
"""
Streamlit application for network attack type prediction.
This app allows users to explore the dataset, visualize key insights, and predict network attack types using a pre-trained Random Forest model.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import os
import joblib

# Set page configuration
st.set_page_config(page_title="Network Attack Detection", layout="wide")

# Cache the data loading and preprocessing to optimize performance
@st.cache_data
def load_and_preprocess_data(file_path):
    try:
        # Load dataset
        dataset = pd.read_csv(file_path)
        
        # Check for missing values
        if dataset.isnull().sum().sum() > 0:
            st.warning("Missing values detected. Filling with zeros for simplicity.")
            dataset.fillna(0, inplace=True)
        
        # Encode categorical variables
        le = LabelEncoder()
        dataset['protocol_type_encoded'] = le.fit_transform(dataset['protocol_type'])
        dataset['service_encoded'] = le.fit_transform(dataset['service'])
        dataset['flag_encoded'] = le.fit_transform(dataset['flag'])
        dataset['Attack Type_encoded'] = le.fit_transform(dataset['Attack Type'])
        
        # Log transformation for numerical variables
        numerical_columns = [
            'src_bytes', 'dst_bytes', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate', 
            'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate',
            'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate', 
            'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate',
            'dst_host_serror_rate', 'dst_host_srv_serror_rate', 'dst_host_rerror_rate', 
            'dst_host_srv_rerror_rate'
        ]
        for col in numerical_columns:
            dataset[f"{col}_log"] = np.log1p(dataset[col].clip(lower=0))
        
        # Normalize numerical features
        scaler = StandardScaler()
        dataset[[f"{col}_log" for col in numerical_columns]] = scaler.fit_transform(
            dataset[[f"{col}_log" for col in numerical_columns]]
        )
        
        return dataset, scaler, le
    except FileNotFoundError:
        st.error("Dataset file not found. Please ensure 'dataset.csv' is in the correct directory.")
        return None, None, None
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None, None, None

# Train and save model (or load if already trained)
@st.cache_resource
def train_model(X, y):
    model = RandomForestClassifier(random_state=42)
    model.fit(X, y)
    joblib.dump(model, "random_forest_model.pkl")
    return model

# Load dataset
file_path = os.path.join(os.path.dirname(__file__), 'dataset.csv')
dataset, scaler, le = load_and_preprocess_data(file_path)

if dataset is not None:
    # Sidebar for user options
    st.sidebar.title("Network Attack Detection App")
    task = st.sidebar.selectbox(
        "Choose a Task",
        ["Overview", "Data Exploration", "Visualizations", "Predict Attack Type"]
    )

    # Main content based on user selection
    st.title("Network Attack Detection Dashboard")

    if task == "Overview":
        st.header("Dataset Overview")
        st.write("This application analyzes network logs to predict types of network attacks (e.g., normal, DoS, probe, R2L, U2R).")
        st.subheader("Dataset Snapshot")
        st.dataframe(dataset.head())
        st.subheader("Dataset Info")
        st.write(dataset.info())
        st.subheader("Summary Statistics")
        st.dataframe(dataset.describe())
        st.subheader("Missing Values")
        st.write(dataset.isnull().sum())
        st.subheader("Class Distribution (Attack Type)")
        st.write(dataset['Attack Type'].value_counts())

    elif task == "Data Exploration":
        st.header("Data Exploration")
        st.subheader("Select a Feature to Explore")
        feature = st.selectbox("Feature", dataset.columns)
        st.write(f"**Description of {feature}**")
        st.write(dataset[feature].describe())
        st.write(f"Unique Values: {dataset[feature].nunique()}")
        if dataset[feature].dtype in ['int64', 'float64']:
            st.write("Distribution Plot")
            fig, ax = plt.subplots()
            sns.histplot(dataset[feature], bins=50, kde=True, ax=ax)
            ax.set_title(f"Distribution of {feature}")
            st.pyplot(fig)
        else:
            st.write("Value Counts")
            st.bar_chart(dataset[feature].value_counts())

    elif task == "Visualizations":
        st.header("Visualizations")
        viz_type = st.selectbox(
            "Select Visualization",
            ["Correlation Heatmap", "Feature Importance", "Attack Type Distribution", "Boxplot of Numeric Feature"]
        )
        
        if viz_type == "Correlation Heatmap":
            st.subheader("Correlation Heatmap")
            numeric_columns = dataset.select_dtypes(include=['number']).columns
            fig, ax = plt.subplots(figsize=(12, 10))
            sns.heatmap(dataset[numeric_columns].corr(), annot=False, cmap='coolwarm', linewidths=0.5, ax=ax)
            ax.set_title("Correlation Matrix")
            st.pyplot(fig)
        
        elif viz_type == "Feature Importance":
            st.subheader("Feature Importance (Random Forest)")
            X = dataset[[f"{col}_log" for col in [
                'src_bytes', 'dst_bytes', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate', 
                'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate',
                'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate', 
                'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate',
                'dst_host_serror_rate', 'dst_host_srv_serror_rate', 'dst_host_rerror_rate', 
                'dst_host_srv_rerror_rate'
            ]]]
            y = dataset['Attack Type_encoded']
            model = train_model(X, y)
            feat_importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
            fig, ax = plt.subplots(figsize=(10, 6))
            feat_importances.plot(kind='bar', ax=ax)
            ax.set_title("Feature Importance")
            st.pyplot(fig)
        
        elif viz_type == "Attack Type Distribution":
            st.subheader("Attack Type Distribution")
            fig, ax = plt.subplots(figsize=(8, 4))
            sns.countplot(x='Attack Type', data=dataset, ax=ax)
            ax.set_title("Distribution of Attack Types")
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
            st.pyplot(fig)
        
        elif viz_type == "Boxplot of Numeric Feature":
            st.subheader("Boxplot")
            numeric_feature = st.selectbox("Select Numeric Feature", dataset.select_dtypes(include=['number']).columns)
            fig, ax = plt.subplots(figsize=(8, 4))
            sns.boxplot(x=dataset[numeric_feature], ax=ax)
            ax.set_title(f"Boxplot of {numeric_feature}")
            st.pyplot(fig)

    elif task == "Predict Attack Type":
        st.header("Predict Attack Type")
        st.write("Enter network log features to predict the type of attack.")
        
        # Input fields for features
        input_features = {}
        for col in [
            'src_bytes', 'dst_bytes', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate', 
            'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate',
            'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate', 
            'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate',
            'dst_host_serror_rate', 'dst_host_srv_serror_rate', 'dst_host_rerror_rate', 
            'dst_host_srv_rerror_rate'
        ]:
            input_features[col] = st.number_input(f"{col}", min_value=0.0, value=0.0, step=0.1)
        
        # Additional categorical inputs
        protocol_type = st.selectbox("Protocol Type", dataset['protocol_type'].unique())
        service = st.selectbox("Service", dataset['service'].unique())
        flag = st.selectbox("Flag", dataset['flag'].unique())
        
        if st.button("Predict"):
            # Preprocess input
            input_data = pd.DataFrame([input_features])
            input_data['protocol_type_encoded'] = le.transform([protocol_type])[0]
            input_data['service_encoded'] = le.transform([service])[0]
            input_data['flag_encoded'] = le.transform([flag])[0]
            
            # Apply log transformation
            for col in input_features.keys():
                input_data[f"{col}_log"] = np.log1p(input_data[col].clip(lower=0))
            
            # Normalize input
            input_data[[f"{col}_log" for col in input_features.keys()]] = scaler.transform(
                input_data[[f"{col}_log" for col in input_features.keys()]]
            )
            
            # Select features for prediction
            features_for_model = [f"{col}_log" for col in input_features.keys()] + [
                'protocol_type_encoded', 'service_encoded', 'flag_encoded'
            ]
            X_input = input_data[features_for_model]
            
            # Load or train model
            X_train = dataset[features_for_model]
            y_train = dataset['Attack Type_encoded']
            model = train_model(X_train, y_train)
            
            # Predict
            prediction = model.predict(X_input)[0]
            attack_type = le.inverse_transform([prediction])[0]
            
            st.subheader("Prediction Result")
            st.write(f"Predicted Attack Type: **{attack_type}**")
            st.write("**Explanation**:")
            st.write("- **Normal**: No attack, regular network traffic.")
            st.write("- **DoS**: Denial of Service attack, overwhelming the network.")
            st.write("- **Probe**: Scanning or probing attempts for reconnaissance.")
            st.write("- **R2L**: Remote to Local, unauthorized access attempts.")
            st.write("- **U2R**: User to Root, privilege escalation attempts.")

# Footer
st.sidebar.markdown("---")
st.sidebar.write("Built with Streamlit and scikit-learn")
st.sidebar.write("Dataset: Network Intrusion Detection")