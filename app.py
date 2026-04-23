"""
🏃 Human Activity Recognition — Premium Streamlit Dashboard
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import os, warnings
warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════
#  PAGE CONFIG
# ═══════════════════════════════════════════════
st.set_page_config(
    page_title="HAR · Activity Intelligence",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═══════════════════════════════════════════════
#  PREMIUM CSS — dark glassmorphism theme
# ═══════════════════════════════════════════════
st.markdown("""
<style>
/* ── Import premium font ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

/* ── Root variables ── */
:root {
    --bg-primary: #0a0e1a;
    --bg-secondary: #111827;
    --bg-card: rgba(17, 24, 39, 0.65);
    --glass: rgba(255,255,255,0.04);
    --glass-border: rgba(255,255,255,0.08);
    --accent-1: #6366f1;
    --accent-2: #8b5cf6;
    --accent-3: #a78bfa;
    --accent-4: #c084fc;
    --success: #10b981;
    --warn: #f59e0b;
    --danger: #ef4444;
    --text-primary: #f1f5f9;
    --text-secondary: #94a3b8;
    --text-muted: #64748b;
    --gradient-1: linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #a78bfa 100%);
    --gradient-2: linear-gradient(135deg, #0ea5e9 0%, #6366f1 100%);
    --gradient-3: linear-gradient(135deg, #10b981 0%, #6366f1 100%);
}

/* ── Global ── */
html, body, [data-testid="stAppViewContainer"],
[data-testid="stApp"] {
    background: var(--bg-primary) !important;
    color: var(--text-primary) !important;
    font-family: 'Inter', sans-serif !important;
}
[data-testid="stHeader"] { background: transparent !important; }
[data-testid="stToolbar"] { display: none !important; }

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f172a 0%, #1e1b4b 100%) !important;
    border-right: 1px solid var(--glass-border) !important;
}
section[data-testid="stSidebar"] * {
    color: var(--text-primary) !important;
}

/* ── Hero header ── */
.hero-wrap {
    text-align: center;
    padding: 2.5rem 1rem 1rem;
}
.hero-badge {
    display: inline-flex;
    align-items: center;
    gap: .5rem;
    background: var(--glass);
    border: 1px solid var(--glass-border);
    border-radius: 50px;
    padding: .35rem 1.1rem;
    font-size: .78rem;
    font-weight: 500;
    color: var(--accent-3);
    letter-spacing: .5px;
    text-transform: uppercase;
    margin-bottom: 1rem;
}
.hero-title {
    font-size: 2.8rem;
    font-weight: 800;
    background: var(--gradient-1);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    line-height: 1.15;
    margin-bottom: .5rem;
}
.hero-sub {
    font-size: 1.05rem;
    color: var(--text-secondary);
    max-width: 640px;
    margin: 0 auto 1.5rem;
    line-height: 1.6;
}

/* ── Glass cards ── */
.glass-card {
    background: var(--bg-card);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border: 1px solid var(--glass-border);
    border-radius: 16px;
    padding: 1.6rem;
    transition: transform .22s ease, box-shadow .22s ease;
}
.glass-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 12px 40px rgba(99,102,241,.12);
}

/* ── Metric cards ── */
.metric-grid { display: grid; grid-template-columns: repeat(4,1fr); gap: 1rem; margin-bottom: 1.5rem; }
.metric-item {
    background: var(--bg-card);
    border: 1px solid var(--glass-border);
    border-radius: 14px;
    padding: 1.3rem 1.4rem;
    position: relative;
    overflow: hidden;
}
.metric-item::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    border-radius: 14px 14px 0 0;
}
.metric-item:nth-child(1)::before { background: var(--gradient-1); }
.metric-item:nth-child(2)::before { background: var(--gradient-2); }
.metric-item:nth-child(3)::before { background: var(--gradient-3); }
.metric-item:nth-child(4)::before { background: linear-gradient(135deg,#f59e0b,#ef4444); }
.metric-label {
    font-size: .78rem;
    font-weight: 500;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: .8px;
    margin-bottom: .35rem;
}
.metric-value {
    font-size: 2rem;
    font-weight: 800;
    color: var(--text-primary);
}
.metric-icon {
    position: absolute;
    top: 1rem; right: 1.2rem;
    font-size: 1.6rem;
    opacity: .4;
}

/* ── Section titles ── */
.section-title {
    font-size: 1.35rem;
    font-weight: 700;
    color: var(--text-primary);
    margin: 2rem 0 1rem;
    display: flex;
    align-items: center;
    gap: .5rem;
}
.section-title::after {
    content: '';
    flex: 1;
    height: 1px;
    background: var(--glass-border);
    margin-left: .8rem;
}

/* ── Prediction result card ── */
.predict-result {
    background: var(--gradient-1);
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
    color: #fff;
    position: relative;
    overflow: hidden;
}
.predict-result::after {
    content: '';
    position: absolute;
    top: -40%; right: -20%;
    width: 260px; height: 260px;
    border-radius: 50%;
    background: rgba(255,255,255,.06);
}
.predict-result h2 { font-size: 2.6rem; font-weight: 900; margin: 0; }
.predict-result h4 { font-weight: 500; opacity: .9; margin: .3rem 0 0; font-size: 1.15rem; }

/* ── PCA coords ── */
.pca-coords {
    background: var(--bg-card);
    border: 1px solid var(--glass-border);
    border-radius: 14px;
    padding: 1.4rem 1.6rem;
}
.pca-row {
    display: flex;
    justify-content: space-between;
    padding: .55rem 0;
    border-bottom: 1px solid var(--glass-border);
    font-size: .92rem;
}
.pca-row:last-child { border-bottom: none; }
.pca-label { color: var(--text-secondary); }
.pca-val { font-weight: 700; font-family: 'SF Mono', 'Fira Code', monospace; color: var(--accent-3); }

/* ── Tabs ── */
button[data-baseweb="tab"] {
    font-family: 'Inter', sans-serif !important;
    font-weight: 600 !important;
    color: var(--text-secondary) !important;
    border-radius: 10px !important;
    padding: .6rem 1.2rem !important;
    transition: all .2s !important;
}
button[data-baseweb="tab"][aria-selected="true"] {
    color: #fff !important;
    background: var(--accent-1) !important;
}
[data-baseweb="tab-highlight"] { display: none !important; }
[data-baseweb="tab-border"] { display: none !important; }

/* ── Input fields ── */
[data-testid="stNumberInput"] label,
[data-testid="stSelectbox"] label {
    color: var(--text-secondary) !important;
    font-weight: 500 !important;
    font-size: .85rem !important;
}
input[type="number"] {
    background: var(--bg-secondary) !important;
    border: 1px solid var(--glass-border) !important;
    border-radius: 10px !important;
    color: var(--text-primary) !important;
}

/* ── Buttons ── */
button[kind="primary"], .stButton > button[kind="primary"] {
    background: var(--gradient-1) !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    padding: .6rem 1.6rem !important;
    transition: transform .18s, box-shadow .18s !important;
}
button[kind="primary"]:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 24px rgba(99,102,241,.3) !important;
}
button[kind="secondary"], .stButton > button:not([kind="primary"]) {
    background: var(--glass) !important;
    border: 1px solid var(--glass-border) !important;
    color: var(--text-primary) !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
}

/* ── Plotly dark background ── */
.js-plotly-plot .plotly .main-svg { border-radius: 14px; }

/* ── Info / alert boxes ── */
[data-testid="stAlert"] {
    background: var(--glass) !important;
    border: 1px solid var(--glass-border) !important;
    border-radius: 12px !important;
    color: var(--text-primary) !important;
}

/* ── Divider ── */
hr { border-color: var(--glass-border) !important; }

/* ── Selectbox ── */
[data-baseweb="select"] > div {
    background: var(--bg-secondary) !important;
    border: 1px solid var(--glass-border) !important;
    border-radius: 10px !important;
    color: var(--text-primary) !important;
}

/* ── Sidebar widgets ── */
.sidebar-stat {
    display: flex;
    align-items: center;
    gap: .6rem;
    padding: .6rem .8rem;
    background: rgba(255,255,255,.04);
    border-radius: 10px;
    margin-bottom: .5rem;
    font-size: .88rem;
}
.sidebar-dot {
    width: 8px; height: 8px;
    border-radius: 50%;
    flex-shrink: 0;
}

/* ── Responsive ── */
@media (max-width: 768px) {
    .metric-grid { grid-template-columns: repeat(2,1fr); }
    .hero-title { font-size: 2rem; }
}

/* ── Animations ── */
@keyframes fadeUp {
    from { opacity: 0; transform: translateY(18px); }
    to   { opacity: 1; transform: translateY(0); }
}
.glass-card, .metric-item { animation: fadeUp .5s ease both; }
.metric-item:nth-child(2) { animation-delay: .06s; }
.metric-item:nth-child(3) { animation-delay: .12s; }
.metric-item:nth-child(4) { animation-delay: .18s; }

/* ── Metric default hide ── */
[data-testid="stMetric"] { display: none !important; }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

SIMPLE_NAMES = {
    "tBodyAcc-mean()-X": "Body Accel · Avg X",
    "tBodyAcc-mean()-Y": "Body Accel · Avg Y",
    "tBodyAcc-mean()-Z": "Body Accel · Avg Z",
    "tBodyAcc-std()-X": "Body Accel · StdDev X",
    "tBodyAcc-std()-Y": "Body Accel · StdDev Y",
    "tBodyAcc-std()-Z": "Body Accel · StdDev Z",
    "tGravityAcc-mean()-X": "Gravity · Avg X",
    "tGravityAcc-mean()-Y": "Gravity · Avg Y",
    "tGravityAcc-mean()-Z": "Gravity · Avg Z",
    "tBodyGyro-mean()-X": "Gyroscope · Avg X",
    "tBodyGyro-mean()-Y": "Gyroscope · Avg Y",
    "tBodyGyro-mean()-Z": "Gyroscope · Avg Z",
}

def simple(name):
    if name in SIMPLE_NAMES:
        return SIMPLE_NAMES[name]
    n = name.replace("tBody", "Body ").replace("tGravity", "Gravity ")
    n = n.replace("fBody", "FFT ").replace("fBodyBody", "FFT ")
    n = n.replace("Acc", "Accel").replace("Mag", " Mag").replace("Jerk", " Jerk")
    n = n.replace("-mean()", " Avg").replace("-std()", " StdDev")
    n = n.replace("-mad()", " MAD").replace("-max()", " Max").replace("-min()", " Min")
    n = n.replace("-energy()", " Energy").replace("-entropy()", " Entropy")
    n = n.replace("-skewness()", " Skew").replace("-kurtosis()", " Kurt")
    n = n.replace("angle(", "∠ ").replace(")", "").replace("-", " ").replace("()", "")
    return n.strip()

CLUSTER_ACTIVITY = {
    0: ("Walking", "🚶", "#6366f1"),
    1: ("Upstairs", "⬆️", "#0ea5e9"),
    2: ("Downstairs", "⬇️", "#14b8a6"),
    3: ("Sitting", "🪑", "#f59e0b"),
    4: ("Standing", "🧍", "#8b5cf6"),
    5: ("Laying", "🛌", "#ec4899"),
}

PLOTLY_DARK = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter", color="#e2e8f0"),
    margin=dict(l=40, r=20, t=50, b=40),
)
COLORS_SEQ = ["#6366f1","#0ea5e9","#14b8a6","#f59e0b","#8b5cf6","#ec4899"]

# ═══════════════════════════════════════════════
#  LOAD / TRAIN MODEL
# ═══════════════════════════════════════════════
@st.cache_resource
def load_or_train():
    data_dir = os.path.join(BASE_DIR, "data")
    model_dir = os.path.join(BASE_DIR, "models")
    paths = {k: os.path.join(model_dir, f) for k, f in
             [("km","kmeans_model.pkl"),("sc","scaler.pkl"),
              ("pca","pca_model.pkl"),("fn","feature_names.pkl")]}
    train = pd.read_csv(os.path.join(data_dir, "train.csv"))
    X = train.drop(columns=["Activity","subject"], errors="ignore")
    fnames = list(X.columns)

    if all(os.path.exists(p) for p in paths.values()):
        km = joblib.load(paths["km"])
        sc = joblib.load(paths["sc"])
        pca = joblib.load(paths["pca"])
        Xs = sc.transform(X)
        Xp = pca.transform(Xs)
        cl = km.predict(Xs)
        return km, sc, pca, fnames, train, Xp, cl

    X = X.fillna(X.median())
    sc = StandardScaler(); Xs = sc.fit_transform(X)
    km = KMeans(n_clusters=6, init="k-means++", n_init=10, max_iter=300, random_state=42)
    km.fit(Xs); cl = km.labels_
    pca = PCA(n_components=3, random_state=42); Xp = pca.fit_transform(Xs)
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(km, paths["km"]); joblib.dump(sc, paths["sc"])
    joblib.dump(pca, paths["pca"]); joblib.dump(fnames, paths["fn"])
    return km, sc, pca, fnames, train, Xp, cl

kmeans, scaler, pca, feature_names, train_df, X_pca, cluster_labels = load_or_train()

# ═══════════════════════════════════════════════
#  SIDEBAR
# ═══════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style="text-align:center;padding:1.5rem 0 .8rem">
        <div style="font-size:2.8rem;margin-bottom:.3rem">🧬</div>
        <div style="font-size:1.3rem;font-weight:800;background:linear-gradient(135deg,#6366f1,#a78bfa);
             -webkit-background-clip:text;-webkit-text-fill-color:transparent">HAR Intelligence</div>
        <div style="font-size:.75rem;color:#64748b;margin-top:.2rem;letter-spacing:1px;text-transform:uppercase">
            Sensor · Cluster · Predict</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("##### 🎯 Detected Activities")
    for cid, (name, icon, color) in CLUSTER_ACTIVITY.items():
        st.markdown(f"""
        <div class="sidebar-stat">
            <div class="sidebar-dot" style="background:{color}"></div>
            <span style="font-weight:600">{icon} {name}</span>
            <span style="margin-left:auto;color:#64748b;font-size:.78rem">C{cid}</span>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("##### 🌍 Real-World Uses")
    for icon, use in [("🏋️","Fitness Tracking"),("🏥","Healthcare"),("👴","Elder Care"),("🏭","Workplace Safety")]:
        st.markdown(f"""
        <div class="sidebar-stat">
            <span>{icon}</span><span>{use}</span>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.caption("Built with Scikit-learn · Streamlit · Plotly")

# ═══════════════════════════════════════════════
#  HERO
# ═══════════════════════════════════════════════
st.markdown("""
<div class="hero-wrap">
    <div class="hero-badge">🔬 Machine Learning Lab Assessment</div>
    <div class="hero-title">Human Activity<br>Recognition</div>
    <div class="hero-sub">Unsupervised K-Means clustering on smartphone accelerometer
        &amp; gyroscope data to discover hidden activity patterns.</div>
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════
#  TABS
# ═══════════════════════════════════════════════
tab1, tab2 = st.tabs([
    "📊  Dashboard", "🔮  Predict"
])

# ─────────────── TAB 1  ·  DASHBOARD ───────────────
with tab1:
    # ── Metrics ──
    st.markdown(f"""
    <div class="metric-grid">
        <div class="metric-item">
            <div class="metric-icon">📋</div>
            <div class="metric-label">Total Samples</div>
            <div class="metric-value">{len(train_df):,}</div>
        </div>
        <div class="metric-item">
            <div class="metric-icon">🔢</div>
            <div class="metric-label">Sensor Features</div>
            <div class="metric-value">{len(feature_names)}</div>
        </div>
        <div class="metric-item">
            <div class="metric-icon">🎯</div>
            <div class="metric-label">Clusters (K)</div>
            <div class="metric-value">6</div>
        </div>
        <div class="metric-item">
            <div class="metric-icon">📐</div>
            <div class="metric-label">PCA Dims</div>
            <div class="metric-value">3</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Charts row ──
    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown('<div class="section-title">📊 Cluster Distribution</div>', unsafe_allow_html=True)
        cc = pd.Series(cluster_labels).value_counts().sort_index()
        fig_bar = go.Figure()
        for idx in cc.index:
            name, icon, color = CLUSTER_ACTIVITY.get(idx, (f"C{idx}", "", "#888"))
            fig_bar.add_trace(go.Bar(
                x=[f"{icon} {name}"], y=[cc[idx]],
                marker=dict(color=color, cornerradius=6),
                text=[cc[idx]], textposition="outside",
                textfont=dict(size=13, color="#e2e8f0"),
                name=name, showlegend=False
            ))
        fig_bar.update_layout(**PLOTLY_DARK, height=380, yaxis_title="Samples",
                              xaxis_title="", bargap=.25)
        st.plotly_chart(fig_bar, width='stretch')

    with col_r:
        st.markdown('<div class="section-title">🥧 Actual Activity Split</div>', unsafe_allow_html=True)
        if "Activity" in train_df.columns:
            ac = train_df["Activity"].value_counts()
            fig_pie = go.Figure(go.Pie(
                labels=ac.index, values=ac.values,
                hole=.45,
                marker=dict(colors=COLORS_SEQ, line=dict(color="#0a0e1a", width=2)),
                textinfo="percent+label", textfont=dict(size=12),
            ))
            fig_pie.update_layout(**PLOTLY_DARK, height=380, showlegend=False)
            st.plotly_chart(fig_pie, width='stretch')

    # ── 2-D scatter ──
    st.markdown('<div class="section-title">🗺️ Cluster Map · PCA 2-D</div>', unsafe_allow_html=True)
    viz = pd.DataFrame({"PC1": X_pca[:,0], "PC2": X_pca[:,1],
                         "Cluster": [f'{CLUSTER_ACTIVITY[c][1]} {CLUSTER_ACTIVITY[c][0]}' for c in cluster_labels]})
    if "Activity" in train_df.columns:
        viz["Actual"] = train_df["Activity"].values
    fig2d = px.scatter(viz, x="PC1", y="PC2", color="Cluster",
                       color_discrete_sequence=COLORS_SEQ, opacity=.55,
                       hover_data=["Actual"] if "Activity" in train_df.columns else None)
    fig2d.update_traces(marker=dict(size=4))
    fig2d.update_layout(**PLOTLY_DARK, height=520, legend=dict(
        orientation="h", y=-0.15, x=.5, xanchor="center",
        font=dict(size=12)))
    st.plotly_chart(fig2d, width='stretch')

# ─────────────── TAB 2  ·  PREDICT ───────────────
with tab2:
    st.markdown('<div class="section-title">🔮 Real-Time Activity Prediction</div>', unsafe_allow_html=True)
    st.markdown("""
    <p style="color:var(--text-secondary);margin-bottom:1.5rem;line-height:1.7">
        Feed sensor readings into the trained K-Means pipeline. Hit <b>Random Sample</b> to
        auto-fill from the dataset, or enter your own accelerometer & gyroscope values.
    </p>
    """, unsafe_allow_html=True)

    key_features = [
        "tBodyAcc-mean()-X","tBodyAcc-mean()-Y","tBodyAcc-mean()-Z",
        "tBodyAcc-std()-X","tBodyAcc-std()-Y","tBodyAcc-std()-Z",
        "tGravityAcc-mean()-X","tGravityAcc-mean()-Y","tGravityAcc-mean()-Z",
        "tBodyGyro-mean()-X"
    ]

    b1, b2 = st.columns(2)
    with b1:
        use_random = st.button("🎲  Fill Random Sample", type="primary", use_container_width=True)
    with b2:
        use_custom = st.button("✏️  Clear Fields", use_container_width=True)

    if use_random:
        ri = np.random.randint(0, len(train_df))
        s = train_df.iloc[ri]
        st.session_state["sv"] = {f: float(s.get(f,0)) for f in key_features}
        if "Activity" in train_df.columns:
            st.info(f"📌  Loaded sample **#{ri}** — Actual: **{s['Activity']}**")
    if use_custom:
        st.session_state["sv"] = {f: 0.0 for f in key_features}

    iv = {}
    cols = st.columns(3)
    for i, feat in enumerate(key_features):
        d = st.session_state.get("sv", {}).get(feat, 0.0)
        with cols[i % 3]:
            iv[feat] = st.number_input(simple(feat), value=d, format="%.6f", key=f"in_{feat}")

    st.markdown("")
    if st.button("🚀  Predict Activity Cluster", type="primary", use_container_width=True):
        full = np.zeros(len(feature_names))
        for j, fn in enumerate(feature_names):
            if fn in iv: full[j] = iv[fn]

        if "sv" in st.session_state:
            ri2 = np.random.randint(0, len(train_df))
            row = train_df.drop(columns=["Activity","subject"], errors="ignore").iloc[ri2]
            for fn2 in key_features:
                if fn2 in iv: row[fn2] = iv[fn2]
            full = row.values

        fs = scaler.transform(full.reshape(1,-1))
        pred = kmeans.predict(fs)[0]
        ppca = pca.transform(fs)[0]
        pname, picon, pcolor = CLUSTER_ACTIVITY.get(pred, ("?","❓","#888"))

        st.markdown("---")
        rc1, rc2 = st.columns([1.2, 1])
        with rc1:
            st.markdown(f"""
            <div class="predict-result" style="background:linear-gradient(135deg,{pcolor},#6366f1)">
                <h2>{picon} Cluster {pred}</h2>
                <h4>{pname}</h4>
            </div>""", unsafe_allow_html=True)
        with rc2:
            st.markdown(f"""
            <div class="pca-coords">
                <div style="font-weight:700;margin-bottom:.8rem;color:var(--text-primary)">📍 PCA Coordinates</div>
                <div class="pca-row"><span class="pca-label">Principal Component 1</span><span class="pca-val">{ppca[0]:+.4f}</span></div>
                <div class="pca-row"><span class="pca-label">Principal Component 2</span><span class="pca-val">{ppca[1]:+.4f}</span></div>
                <div class="pca-row"><span class="pca-label">Principal Component 3</span><span class="pca-val">{ppca[2]:+.4f}</span></div>
            </div>""", unsafe_allow_html=True)

        # scatter with prediction point
        fig_p = px.scatter(viz, x="PC1", y="PC2", color="Cluster",
                           color_discrete_sequence=COLORS_SEQ, opacity=.35)
        fig_p.update_traces(marker=dict(size=3))
        fig_p.add_trace(go.Scatter(
            x=[ppca[0]], y=[ppca[1]], mode="markers+text",
            marker=dict(size=16, color=pcolor, symbol="star",
                        line=dict(width=2, color="#fff")),
            text=["⭐ YOU"], textposition="top center",
            textfont=dict(size=13, color="#fff"), name="Your Input"
        ))
        fig_p.update_layout(**PLOTLY_DARK, height=480,
                            legend=dict(orientation="h", y=-0.15, x=.5, xanchor="center"))
        st.plotly_chart(fig_p, width='stretch')


