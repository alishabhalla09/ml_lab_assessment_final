# 🧠 How It Works (Simplified)

### Step 1: Learning Patterns (Unsupervised)
Since we don't always tell the AI exactly what each movement is at first, it uses a method called **K-Means Clustering**. It looks at the sensor data and groups together movements that look similar.

### Step 2: Making Data Simpler (PCA)
Sensor data has 561 variables. That's too much for humans to see! We use **PCA (Principal Component Analysis)** to squash those 561 numbers down to just 2 or 3, so we can draw them on a map.

### Step 3: Naming the Groups
Once the AI has made 6 groups (clusters), we look at the samples inside each group. We might see that Cluster 5 is mostly people standing still, so we label it "STANDING".

### Step 4: Predicting New Movements
When you enter new sensor data in the app, the AI calculates which of the 6 groups it matches most closely and gives you its best guess!
