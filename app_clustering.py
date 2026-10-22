import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score
import streamlit as st

st.set_page_config(
    page_title='Student Profile Clustering App', page_icon='🎓', layout='wide'
)

st.title('🎓 Students Social Network Profile Clustering')
st.markdown('Segmentation dashboard using pre-trained K-Means model.')


@st.cache_resource
def load_saved_model():
  model = joblib.load('kmeans_model.pkl')
  scaler = joblib.load('scaler.pkl')
  df_sns = pd.read_csv('03_Clustering_Marketing.csv')
  return model, scaler, df_sns


try:
  model, scaler, df_sns = load_saved_model()

  # Clean dataset for cluster assignment
  df_clean = df_sns.copy()
  df_clean['age'] = pd.to_numeric(df_clean['age'], errors='coerce')
  df_clean['age'].fillna(df_clean['age'].median(), inplace=True)
  df_clean['gender'].fillna(df_clean['gender'].mode()[0], inplace=True)

  X = df_clean.drop(columns=['gradyear', 'gender'], errors='ignore')
  X = X.apply(pd.to_numeric, errors='coerce')
  X.fillna(X.median(), inplace=True)

  # Scale features & predict clusters
  X_scaled = scaler.transform(X)
  df_clean['Cluster'] = model.predict(X_scaled)

  # Calculate Silhouette Score
  sil_score = silhouette_score(X_scaled, df_clean['Cluster'])
  st.metric(label='Model Silhouette Score', value=f'{sil_score:.4f}')

  # Display segmented dataset
  st.subheader('Segmented Student Dataset Sample')
  st.dataframe(df_clean.head(10))

  # Cluster Statistics
  st.subheader('📊 Cluster Profile Averages')
  cluster_avg = df_clean.drop(columns=['gradyear', 'gender']).groupby('Cluster').mean()
  st.dataframe(cluster_avg)

except Exception as e:
  st.error(
      f'Error loading model files: {e}. Please run the model saving code first.'
  )