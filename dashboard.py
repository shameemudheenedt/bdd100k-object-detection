"""
Interactive Dashboard for BDD100K Dataset Analysis
Run with: streamlit run dashboard.py
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import json
from collections import Counter
import numpy as np


st.set_page_config(page_title="BDD100K Analysis Dashboard", layout="wide")

st.title("🚗 BDD100K Object Detection Dataset Analysis Dashboard")

# Sidebar configuration
st.sidebar.header("Configuration")
data_dir = st.sidebar.text_input("Data Directory", "/home/hp/Documents/Impliment/data")
analysis_dir = st.sidebar.text_input("Analysis Output Directory", "analysis_output")

@st.cache_data
def load_data(data_path):
    """Load and cache dataset labels."""
    with open(data_path, 'r') as f:
        return json.load(f)

@st.cache_data
def compute_class_distribution(data):
    """Compute class distribution from loaded data."""
    class_counts = Counter()
    for item in data:
        if 'labels' in item:
            for label in item['labels']:
                if 'box2d' in label:
                    class_counts[label['category']] += 1
    return dict(class_counts)

# Main content
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    " Overview", 
    "📈 Class Distribution", 
    " Bounding Box Stats",
    "🌤️ Attributes",
    " Anomalies"
])

with tab1:
    st.header("Dataset Overview")
    
    col1, col2 = st.columns(2)
    
    try:
        # Load from pre-computed class distribution CSV
        class_csv = Path(analysis_dir) / "class_distribution.csv"
        if class_csv.exists():
            df = pd.read_csv(class_csv, index_col=0)
            df = df.reset_index()
            df.columns = ['Class', 'Train', 'Val']
            
            train_images = 69863  # BDD100K train set size
            val_images = 10000    # BDD100K val set size
            train_objects = df['Train'].sum()
            val_objects = df['Val'].sum()
            
            with col1:
                st.metric("Training Images", f"{train_images:,}")
                st.metric("Training Objects", f"{train_objects:,}")
                
            with col2:
                st.metric("Validation Images", f"{val_images:,}")
                st.metric("Validation Objects", f"{val_objects:,}")
            
            st.subheader("Dataset Split Ratio")
            split_data = pd.DataFrame({
                'Split': ['Train', 'Val'],
                'Images': [train_images, val_images],
                'Objects': [train_objects, val_objects]
            })
            
            fig = px.bar(split_data, x='Split', y=['Images', 'Objects'], 
                        barmode='group', title='Train/Val Split Comparison')
            st.plotly_chart(fig, use_container_width=True)
            
        else:
            st.error("Analysis files not found. Please run 'python data_analysis.py' first.")
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")

with tab2:
    st.header("Class Distribution Analysis")
    
    try:
        class_csv = Path(analysis_dir) / "class_distribution.csv"
        if class_csv.exists():
            df = pd.read_csv(class_csv, index_col=0)
            df = df.reset_index()
            df.columns = ['Class', 'Train', 'Val']
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.bar(df, x='Class', y=['Train', 'Val'], 
                           title='Class Distribution: Train vs Val',
                           barmode='group')
                fig.update_xaxes(tickangle=45)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                df['Train_Pct'] = df['Train'] / df['Train'].sum() * 100
                df['Val_Pct'] = df['Val'] / df['Val'].sum() * 100
                
                fig = px.bar(df, x='Class', y=['Train_Pct', 'Val_Pct'],
                           title='Class Distribution: Percentage',
                           barmode='group',
                           labels={'value': 'Percentage (%)'})
                fig.update_xaxes(tickangle=45)
                st.plotly_chart(fig, use_container_width=True)
            
            st.subheader("Class Distribution Table")
            st.dataframe(df.style.format({
                'Train': '{:,.0f}',
                'Val': '{:,.0f}',
                'Train_Pct': '{:.2f}%',
                'Val_Pct': '{:.2f}%'
            }), use_container_width=True)
            
            # Class imbalance analysis
            st.subheader("Class Imbalance Analysis")
            max_class = df.loc[df['Train'].idxmax()]
            min_class = df.loc[df['Train'].idxmin()]
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Most Common Class", max_class['Class'], f"{max_class['Train']:,.0f} instances")
            with col2:
                st.metric("Least Common Class", min_class['Class'], f"{min_class['Train']:,.0f} instances")
            with col3:
                imbalance_ratio = max_class['Train'] / min_class['Train']
                st.metric("Imbalance Ratio", f"{imbalance_ratio:.2f}:1")
                
    except Exception as e:
        st.error(f"Error analyzing class distribution: {str(e)}")

with tab3:
    st.header("Bounding Box Statistics")
    
    try:
        csv_path = Path(analysis_dir) / "bbox_statistics.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.bar(df, x='Class', y='Avg_Width', 
                           title='Average Width by Class',
                           labels={'Avg_Width': 'Width (pixels)'})
                fig.update_xaxes(tickangle=45)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = px.bar(df, x='Class', y='Avg_Height',
                           title='Average Height by Class',
                           labels={'Avg_Height': 'Height (pixels)'})
                fig.update_xaxes(tickangle=45)
                st.plotly_chart(fig, use_container_width=True)
            
            st.subheader("Area Statistics")
            fig = px.scatter(df, x='Avg_Width', y='Avg_Height', 
                           size='Avg_Area', color='Class',
                           title='Bounding Box Dimensions by Class',
                           labels={'Avg_Width': 'Width (pixels)', 
                                  'Avg_Height': 'Height (pixels)'})
            st.plotly_chart(fig, use_container_width=True)
            
            st.subheader("Detailed Statistics")
            st.dataframe(df.style.format({
                'Avg_Width': '{:.2f}',
                'Avg_Height': '{:.2f}',
                'Avg_Area': '{:.2f}',
                'Std_Width': '{:.2f}',
                'Std_Height': '{:.2f}',
                'Min_Area': '{:.2f}',
                'Max_Area': '{:.2f}'
            }), use_container_width=True)
        else:
            st.warning("Run data_analysis.py first to generate statistics.")
    except Exception as e:
        st.error(f"Error loading bbox statistics: {str(e)}")

with tab4:
    st.header("Dataset Attributes Analysis")
    
    try:
        for attr in ['weather', 'scene', 'timeofday']:
            csv_path = Path(analysis_dir) / f"{attr}_distribution.csv"
            if csv_path.exists():
                df = pd.read_csv(csv_path, index_col=0)
                
                st.subheader(f"{attr.capitalize()} Distribution")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    fig = px.bar(df, x=df.index, y=['Train', 'Val'],
                               title=f'{attr.capitalize()}: Train vs Val',
                               barmode='group')
                    fig.update_xaxes(tickangle=45)
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    fig = px.pie(df, values='Train', names=df.index,
                               title=f'{attr.capitalize()} Distribution (Train)')
                    st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Error loading attribute statistics: {str(e)}")

with tab5:
    st.header("Anomaly Detection")
    
    try:
        csv_path = Path(analysis_dir) / "anomalies.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            
            st.subheader("Outlier Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.bar(df, x='Class', y='Outlier_Ratio',
                           title='Outlier Ratio by Class',
                           labels={'Outlier_Ratio': 'Outlier Ratio'})
                fig.update_xaxes(tickangle=45)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = px.bar(df, x='Class', 
                           y=['Very_Small_(<100px)', 'Very_Large_(>100000px)'],
                           title='Extreme Size Objects',
                           barmode='group')
                fig.update_xaxes(tickangle=45)
                st.plotly_chart(fig, use_container_width=True)
            
            st.subheader("Anomaly Details")
            st.dataframe(df.style.format({
                'Total_Instances': '{:,.0f}',
                'Outliers': '{:,.0f}',
                'Outlier_Ratio': '{:.4f}',
                'Very_Small_(<100px)': '{:,.0f}',
                'Very_Large_(>100000px)': '{:,.0f}'
            }), use_container_width=True)
            
            # Highlight classes with high anomaly rates
            st.subheader(" Classes with High Anomaly Rates")
            high_anomaly = df[df['Outlier_Ratio'] > 0.05].sort_values('Outlier_Ratio', ascending=False)
            if not high_anomaly.empty:
                st.dataframe(high_anomaly, use_container_width=True)
            else:
                st.success("No classes with anomaly rate > 5%")
                
    except Exception as e:
        st.error(f"Error loading anomaly data: {str(e)}")

# Footer
st.sidebar.markdown("---")
st.sidebar.info("""
**BDD100K Analysis Dashboard**

This dashboard provides comprehensive analysis of the BDD100K dataset for object detection tasks.

**Usage:**
1. Run `python data_analysis.py` first
2. Launch dashboard: `streamlit run dashboard.py`
3. Explore different tabs for insights
""")
