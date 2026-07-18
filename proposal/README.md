> **Note:** This documents the proposal-stage implementation, kept for reference. The final eight-stage pipeline is in `bank_rfm/` — see the root README.

# Bank Customer Segmentation Pipeline using K-Means Clustering

An end-to-end, high-throughput machine learning pipeline designed to aggregate, normalize, optimize, and segment bank transactional data into highly distinct customer behavioral archetypes.

## ðŸ“Œ Project Architecture

The repository is built using a strict modular approach to maintain an isolated environment and fully reproducible execution sequence:

```text
ARM-Demo/
â”œâ”€â”€ 01_rfm_aggregation.py       # Date parsing, feature engineering, Log1p transform, and Z-score scaling
â”œâ”€â”€ 02_kmeans_clustering.py  # Dual-axis elbow evaluation (WCSS vs. Silhouette Coefficient)
â”œâ”€â”€ 03_assign_clusters.py       # Final model training and cluster assignment extraction
â”œâ”€â”€ 04_spatial_visualization.py # Interactive 3D Plotly visualization generation
â”œâ”€â”€ 05_pairplot_matrix.py       # Publication-grade 2D Seaborn pairplots
â””â”€â”€ requirements.txt            # Project dependencies

ðŸ› ï¸ Installation & Environment Setup
Clone the repository:
## ðŸ› ï¸ Installation & Environment Setup

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

4. Create a data folder named "data" inside the project folder, and copy paste the csv file I have shared with you.

5. Pipeline Execution Sequence
python 01_rfm_aggregation.py
python 02_kmeans_clustering.py
python 03_assign_clusters.py
python 04_spatial_visualization.py
python 05_pairplot_matrix.py

Once all the scripts are run, the project structure should look like:
ARM-Demo/
â”œâ”€â”€ data/
â”‚   â”œâ”€â”€ bank_transactions.csv   # Raw source transactional dataset
â”‚   â””â”€â”€ processed/
â”‚       â”œâ”€â”€ final_bank_segments.csv # Labeled customer profiles
â”‚       â””â”€â”€ rfm_scaled.csv          # Isotropic standardized feature space
â”œâ”€â”€ venv/                       # Isolated virtual environment
â”œâ”€â”€ 01_rfm_aggregation.py       # Date parsing, feature aggregation, and scaling
â”œâ”€â”€ 02_cluster_optimization.py  # Dual-axis elbow evaluation (WCSS vs. Silhouette)
â”œâ”€â”€ 03_assign_clusters.py       # Final model training and cluster extraction
â”œâ”€â”€ 04_spatial_visualization.py # Interactive 3D Plotly visualization generation
â”œâ”€â”€ 05_pairplot_matrix.py       # Publication-grade 2D Seaborn pairplots
â”œâ”€â”€ .gitignore                  # Git tracking exclusion rule-set
â””â”€â”€ requirements.txt            # Project dependencies


ðŸ“Š Analytical MethodologyFeature Engineering: Constructs Recency (days since last transaction), Frequency (total transaction volume), and Monetary (aggregate transaction amount in INR) matrices aggregated per customer ID.Normalization: Heavy financial distribution right-skewness is treated using a non-linear $log(x + 1)$ mapping to establish symmetric data distributions.Isotropic Scaling: Features are standardized using a Z-score scaler (StandardScaler) to eliminate distance matrix bias during multidimensional K-Means Euclidean space routing.Hyperparameter Optimization: Employs the dual-validation intersection of Within-Cluster Sum of Squares (Inertia) and Silhouette Analysis to select $K=5$ as the optimal configuration.


