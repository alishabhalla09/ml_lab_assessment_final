# 🏃 Human Activity Recognition Using Smartphones

Unsupervised K-Means clustering on smartphone sensor data to discover human activity patterns.

---

## 🔗 Live Demo

> **Streamlit App:** [Paste your deployed link here]

---

## 📖 About the Project

This project applies **unsupervised machine learning** on the **UCI HAR Dataset** to group smartphone accelerometer and gyroscope readings into activity clusters — without using any labels during training.

**Key highlights:**
- Dataset: **7,352 samples** × **561 sensor features** from Samsung Galaxy S II
- **K-Means Clustering** (K=6) with **PCA** dimensionality reduction
- Interactive **Streamlit dashboard** for real-time activity prediction
- **Adjusted Rand Index = 0.42** — moderate cluster–label agreement

---

## 📁 Project Structure

```
lab_assessment/
│
├── data/
│   ├── train.csv              # Training data (7,352 samples)
│   └── test.csv               # Testing data (2,947 samples)
│
├── models/
│   ├── kmeans_model.pkl       # Trained K-Means model
│   ├── scaler.pkl             # Fitted StandardScaler
│   ├── pca_model.pkl          # Fitted PCA transformer
│   └── feature_names.pkl      # Feature names list
│
├── visualizations/
│   ├── elbow_method.png       # Elbow + Silhouette plot
│   ├── pca_2d_visualization.png
│   ├── pca_3d_visualization.png
│   ├── cluster_vs_actual_heatmap.png
│   └── cluster_radar_charts.png
│
├── code.ipynb                 # Jupyter Notebook (complete analysis)
├── analysis.py                # Python script (ML pipeline)
├── app.py                     # Streamlit web application
└── README.md                  # This file
```

---

## 🚀 How to Run

```bash
# Install dependencies
pip install pandas numpy scikit-learn matplotlib seaborn streamlit plotly joblib

# Run the analysis
python analysis.py

# Launch the dashboard
streamlit run app.py
```

---

## Plotly

Plotly is used in 
app.py
 to create interactive charts inside the Streamlit dashboard — things like the 2D PCA scatter plot, cluster bar chart, and donut chart. Unlike Matplotlib (which makes static images), Plotly charts let users hover, zoom, and pan directly in the browser.

## Joblib

Joblib is used to save and load the trained ML models (
.pkl
 files). Instead of re-training K-Means every time the app starts, we:

Train once → joblib.dump(model, 'file.pkl') (saves to disk)
Load later → joblib.load('file.pkl') (instant, no re-training)
It saves these 4 files in models/:

kmeans_model.pkl
 — the trained K-Means clusterer
scaler.pkl
 — the StandardScaler (for normalizing new inputs)
pca_model.pkl
 — the PCA transformer (for dimensionality reduction)
feature_names.pkl
 — list of 561 feature names
In short: Plotly = interactive graphs, Joblib = save/load models.

## 🛠️ Tech Stack

Python · Pandas · NumPy · Scikit-learn · Matplotlib · Seaborn · Plotly · Streamlit

---

*ML & Pattern Recognition Lab Assessment · 2026*
