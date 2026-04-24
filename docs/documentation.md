# 📘 Human Activity Recognition Studio - Documentation

## 🌍 Live Deployment
**App URL**: [https://alisha-bhalla-09-first-deployment.streamlit.app](https://alisha-bhalla-09-first-deployment.streamlit.app)

## 🌟 Project Overview
This project is an end-to-end Machine Learning system designed to recognize human physical activities (such as **Walking**, **Standing**, **Laying**, **Walking Upstairs**, and **Walking Downstairs**) using smartphone sensor data. 

The system uses unsupervised learning (**K-Means Clustering**) to group similar movement patterns from a 561-feature dataset collected from accelerometers and gyroscopes.


---

## 🛠️ Technology Stack
- **Dashboard**: Streamlit (for a premium web interface)
- **Machine Learning**: Scikit-Learn (K-Means, PCA, StandardScaler)
- **Data Processing**: Pandas, NumPy
- **Visualizations**: Matplotlib, Seaborn
- **Development**: Python 3.8+

---

## ✨ Key Features
1. **Readable Sensor Data**: Automatically converts technical names like `tBodyAcc-mean()-X` into `Body Acceleration Mean X`.
2. **Interactive Dashboard**: Explore behavior clusters through 2D PCA mapping and activity heatmaps.
3. **Behavioral Signatures**: View radar charts showing the "fingerprint" of different physical movements.
4. **Real-time Prediction Engine**: Input sensor telemetry values to instantly predict the associated activity.

---

## 🚀 How to Run the Project

### 1. Install Dependencies
Ensure you have the required libraries installed:
```bash
pip install -r requirements.txt
```

### 2. Run Data Analysis (Training)
Run this script to process the data, generate visualizations, and save the AI models:
```bash
python analysis.py
```

### 3. Start the Dashboard
Launch the web interface locally:
```bash
streamlit run app.py
```

### 4. Cloud Deployment
If deploying to **Streamlit Cloud**, simply push the repository to GitHub. The app will use the provided `requirements.txt` and pre-trained models in the `models/` folder.

---

## 🔬 Technical Workflow
1. **Preprocessing**: The raw sensor data is cleaned and scaled to ensure the AI detects patterns based on movement, not signal magnitude.
2. **Clustering**: K-Means splits the data into 6 distinct groups based on behavioral similarity.
3. **PCA**: Reduces 561 dimensions down to 3 components for intuitive 2D/3D visualization.
4. **Interpretation**: The system matches clusters to actual activities by identifying the most common movement in each group.
