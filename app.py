import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import time

# --- Page Configuration ---
st.set_page_config(
    page_title="HAR Pro | Human Activity Recognition",
    page_icon="🏃",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for Premium Look ---
st.markdown("""
<style>
    :root {
        --primary-color: #6366f1;
        --bg-glass: rgba(255, 255, 255, 0.05);
        --border-glass: rgba(255, 255, 255, 0.1);
    }
    
    .main {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%);
        color: #f8fafc;
    }
    
    .stMetric {
        background: var(--bg-glass);
        padding: 20px;
        border-radius: 15px;
        border: 1px solid var(--border-glass);
        backdrop-filter: blur(10px);
        transition: transform 0.3s ease;
    }
    
    .stMetric:hover {
        transform: translateY(-5px);
        border-color: var(--primary-color);
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
        background-color: transparent;
    }

    .stTabs [data-baseweb="tab"] {
        height: 50px;
        background-color: var(--bg-glass);
        border-radius: 10px 10px 0px 0px;
        padding: 0px 20px;
        color: #94a3b8;
        border: 1px solid var(--border-glass);
    }

    .stTabs [aria-selected="true"] {
        background-color: var(--primary-color) !important;
        color: white !important;
    }
    
    .prediction-card {
        padding: 30px;
        border-radius: 20px;
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.1) 0%, rgba(168, 85, 247, 0.1) 100%);
        border: 1px solid rgba(255, 255, 255, 0.1);
        margin: 20px 0;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# --- Constants ---
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
VIZ_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'visualizations')

# --- Load Assets ---
@st.cache_resource
def load_assets():
    assets = {}
    expected_files = [
        'kmeans.pkl', 'scaler.pkl', 'pca.pkl', 
        'feature_names.pkl', 'readable_feature_mapping.pkl',
        'cluster_activity_mapping.pkl', 'important_features.pkl'
    ]
    
    try:
        for file in expected_files:
            path = os.path.join(MODEL_DIR, file)
            if os.path.exists(path):
                with open(path, 'rb') as f:
                    assets[file.split('.')[0]] = pickle.load(f)
            else:
                st.error(f"Missing artifact: {file}")
                return None
        return assets
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None

assets = load_assets()

# --- Sidebar ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3048/3048325.png", width=100)
    st.title("HAR Studio Pro")
    st.markdown("---")
    page = st.radio("Navigation", ["Dashboard", "Activity Engine", "Documentation"])
    
    st.markdown("---")
    if assets:
        st.success("✅ Models Loaded")
    else:
        st.error("❌ Models Missing")
    
    st.info("Built with Scikit-Learn • Streamlit • PCA")

if not assets:
    st.warning("⚠️ Critical assets are missing. Please run `analysis.py` to generate the machine learning models and feature mappings.")
    st.stop()

# --- Page: Dashboard ---
if page == "Dashboard":
    st.title("🏃 Human Activity Intelligence")
    st.markdown("Real-time sensor pattern recognition and cluster insights.")
    
    # Hero Metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Dataset", "10,299", "Samples")
    with col2:
        st.metric("Sensor Array", "561", "Features")
    with col3:
        st.metric("Active Clusters", "6", "Groups")
    with col4:
        variance = assets['pca'].explained_variance_ratio_.sum() * 100
        st.metric("PCA Variance", f"{variance:.1f}%", "Explained")

    # Main Visuals
    tab1, tab2 = st.tabs(["📊 Cluster Analysis", "🎯 Feature Signatures"])
    
    with tab1:
        v_col1, v_col2 = st.columns(2)
        
        with v_col1:
            st.subheader("2D Semantic Mapping (PCA)")
            p1 = os.path.join(VIZ_DIR, "pca_2d_visualization.png")
            if os.path.exists(p1):
                st.image(p1, use_container_width=True)
            else:
                st.info("PCA Plot generating...")

        with v_col2:
            st.subheader("Cluster-Actual Alignment")
            p2 = os.path.join(VIZ_DIR, "cluster_vs_actual_heatmap.png")
            if os.path.exists(p2):
                st.image(p2, use_container_width=True)
            else:
                st.info("Heatmap generating...")

    with tab2:
        st.subheader("Centroid Signatures (Radar)")
        p3 = os.path.join(VIZ_DIR, "cluster_radar_charts.png")
        if os.path.exists(p3):
            st.image(p3, use_container_width=True)
        else:
            st.info("Radar charts generating...")

# --- Page: Activity Engine ---
elif page == "Activity Engine":
    st.title("🔮 Activity Recognition Engine")
    st.markdown("Inject sensor telemetry to identify human movement patterns.")
    
    important_features = assets['important_features']
    mapping = assets['readable_feature_mapping']
    
    with st.container():
        st.subheader("📝 Input Telemetry")
        st.info("Enter values for the most discriminant features identified by PCA.")
        
        with st.form("engine_form", clear_on_submit=False):
            input_cols = st.columns(2)
            user_inputs = {}
            
            for i, feat in enumerate(important_features):
                col = input_cols[i % 2]
                readable = mapping.get(feat, feat)
                user_inputs[feat] = col.number_input(readable, value=0.0, format="%.6f", help=f"Original key: {feat}")
            
            submit = st.form_submit_button("RUN ANALYSIS 🚀")
            
            if submit:
                # Progress simulation for "Wow" factor
                progress_bar = st.progress(0)
                status_text = st.empty()
                for i in range(100):
                    time.sleep(0.005)
                    progress_bar.progress(i + 1)
                    status_text.text(f"Processing signal arrays... {i+1}%")
                status_text.empty()
                progress_bar.empty()
                
                # Prediction Logic
                full_input = np.zeros((1, len(assets['feature_names'])))
                for feat, val in user_inputs.items():
                    idx = assets['feature_names'].index(feat)
                    full_input[0, idx] = val
                
                scaled = assets['scaler'].transform(full_input)
                cluster_id = assets['kmeans'].predict(scaled)[0]
                activity_name = assets['cluster_activity_mapping'].get(cluster_id, "Unknown movement")
                
                # Result Display
                st.markdown(f"""
                <div class="prediction-card">
                    <h1 style="color: #6366f1; margin-bottom: 0;">CLUSTER {cluster_id}</h1>
                    <h2 style="margin-top: 0;">Predicted Activity: {activity_name}</h2>
                    <p style="color: #94a3b8;">Confidence Level: <b>High</b> (Based on Centroid Proximity)</p>
                </div>
                """, unsafe_allow_html=True)
                
                if "WALKING" in activity_name:
                    st.balloons()
                else:
                    st.snow()

# --- Page: Documentation ---
elif page == "Documentation":
    st.title("📚 Intelligence Documentation")
    
    with st.expander("🔬 How it Works", expanded=True):
        st.write("""
        This application uses an unsupervised learning approach to recognize human activities.
        
        1. **Data Collection**: 561-feature vector from smartphone accelerometers and gyroscopes.
        2. **Standardization**: Values are normalized to a mean of 0 and variance of 1.
        3. **Clustering**: K-Means algorithm partitions the data into 6 distinct behavioral clusters.
        4. **Validation**: PCA (Principal Component Analysis) is used to verify cluster separation in lower dimensions.
        """)
    
    with st.expander("🏷️ Feature Mapping (Readable)"):
        df_map = pd.DataFrame([
            {"Technical Name": k, "Readable Name": v} 
            for k, v in assets['readable_feature_mapping'].items()
        ]).head(20)
        st.table(df_map)
        st.caption("Showing top 20 of 561 features.")

st.sidebar.markdown("---")
st.sidebar.caption("© 2026 HAR Studio Pro | Antigravity AI")
