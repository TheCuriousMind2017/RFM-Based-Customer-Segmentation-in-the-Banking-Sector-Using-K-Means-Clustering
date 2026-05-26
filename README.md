# Bank Customer Segmentation Pipeline using K-Means Clustering

An end-to-end, high-throughput machine learning pipeline designed to aggregate, normalize, optimize, and segment bank transactional data into highly distinct customer behavioral archetypes.

## 📌 Project Architecture

The repository is built using a strict modular approach to maintain an isolated environment and fully reproducible execution sequence:

```text
ARM-Demo/
├── 01_rfm_aggregation.py       # Date parsing, feature engineering, Log1p transform, and Z-score scaling
├── 02_kmeans_clustering.py  # Dual-axis elbow evaluation (WCSS vs. Silhouette Coefficient)
├── 03_assign_clusters.py       # Final model training and cluster assignment extraction
├── 04_spatial_visualization.py # Interactive 3D Plotly visualization generation
├── 05_pairplot_matrix.py       # Publication-grade 2D Seaborn pairplots
└── requirements.txt            # Project dependencies

🛠️ Installation & Environment Setup
Clone the repository:
## 🛠️ Installation & Environment Setup

Follow these steps exactly to configure your local development environment cleanly:

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/TheCuriousMind2017/RFM-Based-Customer-Segmentation-in-the-Banking-Sector-Using-K-Means-Clustering.git](https://github.com/TheCuriousMind2017/RFM-Based-Customer-Segmentation-in-the-Banking-Sector-Using-K-Means-Clustering.git)
   cd RFM-Based-Customer-Segmentation-in-the-Banking-Sector-Using-K-Means-Clustering

2. Initialize and activate the virtual environment:

python -m venv venv
# On Windows:

venv\Scripts\activate
# On Mac/Linux:

source venv/bin/activate

3. Install dependencies:
pip install -r requirements.txt

🚀 Pipeline Execution Sequence
Execute the modules in the exact sequential order defined below:

python 01_rfm_aggregation.py
python 02_kmeans_clustering.py
python 03_assign_clusters.py
python 04_spatial_visualization.py
python 05_pairplot_matrix.py

📊 Analytical MethodologyFeature Engineering: Constructs Recency (days since last transaction), Frequency (total transaction volume), and Monetary (aggregate transaction amount in INR) matrices aggregated per customer ID.Normalization: Heavy financial distribution right-skewness is treated using a non-linear $log(x + 1)$ mapping to establish symmetric data distributions.Isotropic Scaling: Features are standardized using a Z-score scaler (StandardScaler) to eliminate distance matrix bias during multidimensional K-Means Euclidean space routing.Hyperparameter Optimization: Employs the dual-validation intersection of Within-Cluster Sum of Squares (Inertia) and Silhouette Analysis to select $K=5$ as the optimal configuration.

