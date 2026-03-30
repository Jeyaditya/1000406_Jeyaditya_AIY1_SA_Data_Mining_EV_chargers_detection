import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from mlxtend.frequent_patterns import apriori, association_rules

# --- 1. CONFIGURATION & STYLING ---
st.set_page_config(page_title="Pro-EV Analytics | JD", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
    <style>
    [data-testid="stMetric"] { 
        background-color: #000000; 
        padding: 15px; 
        border-radius: 10px; 
        border: 1px solid #333333; 
    }
    .main-title {
        text-align: center;
        color: #00FFAA;
        font-size: 48px;
        font-weight: bold;
        margin-bottom: 0px;
    }
    .sub-title {
        text-align: center;
        font-size: 20px;
        color: #888888;
        margin-bottom: 20px;
    }
    </style>
    """, unsafe_allow_html=True) 

# --- 2. HEADER & TITLE SECTION ---
st.markdown('<p class="main-title">SmartCharging Analytics</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Uncovering EV Behavior Patterns & Infrastructure Intelligence</p>', unsafe_allow_html=True)

with st.expander("View Project Scope & Objectives:"):
    st.write("""
    **Goal:** Analyze global EV charging patterns to improve station utilization and customer experience.
    **Key Objectives:**
    * **Clustering:** Group stations by behavior (Daily, Occasional, Heavy).
    * **Anomaly Detection:** Identify geographic errors and statistical usage spikes.
    * **Association Mining:** Discover links between Renewable Energy and High Demand.
    """)

st.markdown("---")

# --- 3. DATA PIPELINE ---
@st.cache_data
def run_analytics_pipeline():
    try:
        df = pd.read_csv('detailed_ev_charging_stations.csv')
    except FileNotFoundError:
        st.error("Error: 'detailed_ev_charging_stations.csv' not found. Please ensure the dataset is in the repository.")
        return None, None, None

    df['Reviews (Rating)'] = df['Reviews (Rating)'].fillna(df['Reviews (Rating)'].median())
    df['Renewable Energy Source'] = df['Renewable Energy Source'].fillna('No')
    
    df['Is_Geographic_Outlier'] = df.apply(
        lambda row: True if (abs(row['Latitude']) < 0.5 and abs(row['Longitude']) < 0.5) 
        or (row['Latitude'] > -10 and row['Latitude'] < 10 and row['Longitude'] > -40 and row['Longitude'] < -10)
        else False, axis=1
    )
    
    df_clean = df[df['Is_Geographic_Outlier'] == False].copy()

    features = ['Usage Stats (avg users/day)', 'Cost (USD/kWh)', 'Charging Capacity (kW)']
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df_clean[features])

    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    df_clean['Cluster'] = kmeans.fit_predict(scaled_features) 
    
    cluster_map = {0: "Daily Commuters", 1: "Occasional Users", 2: "Heavy Users"}
    df_clean['Segment'] = df_clean['Cluster'].map(cluster_map)
    
    color_palette = {"Daily Commuters": "#1f77b4", "Occasional Users": "#2ca02c", "Heavy Users": "#d62728"}
    df_clean['Visual_Color'] = df_clean['Segment'].map(color_palette)
    mu = df_clean['Usage Stats (avg users/day)'].mean()
    sigma = df_clean['Usage Stats (avg users/day)'].std()
    df_clean['Usage_ZScore'] = (df_clean['Usage Stats (avg users/day)'] - mu) / sigma
    df_clean['Status'] = df_clean['Usage_ZScore'].apply(lambda x: "Anomaly" if abs(x) > 2.5 else "Normal")

    df_clean['High_Demand'] = (df_clean['Usage Stats (avg users/day)'] > df_clean['Usage Stats (avg users/day)'].median()).astype(int)
    df_clean['Eco_Friendly'] = (df_clean['Renewable Energy Source'] == 'Yes').astype(int)
    
    basket = df_clean[['High_Demand', 'Eco_Friendly']]
    freq_items = apriori(basket, min_support=0.05, use_colnames=True)
    rules = association_rules(freq_items, metric="lift", min_threshold=1)
    
    ocean_outliers = df[df['Is_Geographic_Outlier'] == True].copy()
    
    return df_clean, rules, ocean_outliers

df_clean, rules, ocean_outliers = run_analytics_pipeline()

# --- 4. DEPLOYMENT INTERFACE ---
if df_clean is not None:
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Select View", ["Executive Summary", "Geographic Intelligence", "Behavioral Clusters", "Anomalies & Integrity"])

    if page == "Executive Summary":
        st.header("Strategic Overview")
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Active Stations", len(df_clean))
        c2.metric("Mean Daily Usage", f"{df_clean['Usage Stats (avg users/day)'].mean():.1f} Users")
        c3.metric("Association Rules Found", len(rules))
        
        st.subheader("Frequent Pattern Discovery (Apriori)")
        st.dataframe(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head(), use_container_width=True)
        
        st.subheader("Actionable Insight")
        st.info("The data indicates 'Heavy Users' are the most critical segment for infrastructure expansion. Stations utilizing Renewable Energy correlate with higher demand.")

    elif page == "Geographic Intelligence":
        st.header("Global Infrastructure Map")
        st.map(df_clean, latitude='Latitude', longitude='Longitude', color='Visual_Color')
        st.caption("🔵 Blue: Daily Commuters | 🟢 Green: Occasional | 🔴 Red: Heavy Users (Priority Expansion)")

    elif page == "Behavioral Clusters":
        st.header("Machine Learning Segmentation")
        fig = px.scatter(df_clean, x='Charging Capacity (kW)', y='Usage Stats (avg users/day)', 
                         color='Segment', size='Cost (USD/kWh)', hover_name='Station ID',
                         color_discrete_map={"Daily Commuters": "#1f77b4", "Occasional Users": "#2ca02c", "Heavy Users": "#d62728"},
                         title="Usage vs Capacity Cluster Map")
        st.plotly_chart(fig, use_container_width=True)

    elif page == "Anomalies & Integrity":
        st.header("Data Integrity & Anomaly Report")
        
        col_a, col_b = st.columns(2)
        with col_a:
            st.subheader("🚩 Geographic Errors")
            st.error(f"Detected {len(ocean_outliers)} stations in oceanic coordinates.")
            st.write(ocean_outliers[['Station ID', 'Latitude', 'Longitude']])
            
        with col_b:
            st.subheader("⚠️ Statistical Anomalies")
            anomalies = df_clean[df_clean['Status'] == "Anomaly"]
            st.warning(f"Detected {len(anomalies)} abnormal usage patterns ($|Z| > 2.5$).")
            st.write(anomalies[['Station ID', 'Usage Stats (avg users/day)', 'Usage_ZScore']])

# --- FOOTER ---
st.markdown("---")
st.caption("Candidate: Jeyaditya (JD) | ID: 1000406 | School: Jain Vidyalaya IB World School")
