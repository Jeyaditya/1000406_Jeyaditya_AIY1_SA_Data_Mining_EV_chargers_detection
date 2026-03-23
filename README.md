# SmartCharging Analytics: Uncovering EV Behavior Patterns

**Candidate Name:** Jeyaditya (JD)
**Candidate Registration Number:** 1000406
**CRS Name:** Artificial Intelligence
**Course Name:** Data Mining Summative Assessment
**School Name:** Jain Vidyalaya IB World School

---

## 🚀 Live Application
Access the interactive dashboard here: 
**[SmartCharging Analytics Dashboard](https://1000406jeyadityaaiy1sadataminingevchargersdetection.streamlit.app/)**

---

## 📌 Project Rationale & Scope
This project addresses the critical challenge of optimizing global Electric Vehicle (EV) infrastructure.  By applying advanced data mining techniques to charging datasets, the application identifies usage trends, segments station behaviors, and audits data integrity to support strategic expansion and improved customer experience.

### Primary Objectives:
*  **Behavioral Segmentation:** Grouping stations into "Daily Commuters," "Occasional Users," and "Heavy Users" using K-Means clustering.
*  **Anomaly Detection:** Identifying geographic outliers and statistical usage spikes to flag faulty hardware or reporting errors.
*  **Association Rule Mining:** Discovering hidden relationships between "Eco-Friendly" status and "High Demand" patterns.
*  **Interactive Deployment:** Providing stakeholders with a real-time Streamlit intelligence dashboard.

---

## 🛠️ Data Preparation & Preprocessing
To ensure **Distinguished** data quality, the following pipeline was implemented in the source code:

* **Spatial Integrity Auditing:** A custom bounding-box algorithm isolates stations mapped to (0,0) or Atlantic Ocean coordinates, preventing geographic visualization bias.
*  **Missing Value Management:** * `Reviews (Rating)`: Imputed using the **Median** to maintain statistical distribution.
    * `Renewable Energy Source`: Null values filled with **'No'** to ensure conservative "Eco-Friendly" reporting.
* **Feature Engineering:** * Created `High_Demand` binary flags based on median usage thresholds.
    * Created `Eco_Friendly` flags for Association Rule Mining.
*  **Data Normalization:** Continuous variables (`Cost`, `Usage`, `Capacity`) were scaled using the **StandardScaler** to ensure scale-invariance for the clustering algorithm:
  $$x_{scaled} = \frac{x - \mu}{\sigma}$$

---

## 📊 Exploratory Data Analysis (EDA)
The "Story of the Data" is visualized through three core lenses in the dashboard:
* **Executive Summary:** High-level metrics showing Active Stations, Mean Daily Usage, and the count of Association Rules discovered.
* **Geographic Intelligence:** A global map color-coded by behavioral segment (Blue: Commuters, Green: Occasional, Red: Heavy Users).
* **Usage vs. Capacity:** A scatter plot analysis exploring the relationship between a station's capacity and its actual utilization, sized by cost.

---

## 🤖 Advanced Analytics & Machine Learning

### 1. Behavioral Clustering (K-Means)
Using the **K-Means algorithm** on scaled features (`Usage`, `Cost`, `Capacity`), stations are segmented into three distinct operational profiles:
1. **Daily Commuters:** Moderate, stable usage.
2. **Occasional Users:** Low-frequency charging points.
3.  **Heavy Users:** High-demand hubs requiring prioritized maintenance and infrastructure expansion[cite: 2].

### 2. Association Rule Mining (Apriori)
The application utilizes the **Apriori Algorithm** to identify frequent patterns between station attributes:
* **Feature Set:** `{High_Demand, Eco_Friendly}`.
*  **Metrics:** Rules are evaluated based on **Support, Confidence, and Lift** to determine the strength of the relationship between renewable energy and user demand[cite: 2].

### 3. Anomaly Detection (Z-Score)
To safeguard hardware and data integrity, a statistical outlier detection model identifies abnormal usage:
* **Formula:** $Z = \frac{x - \mu}{\sigma}$
*  **Threshold:** Stations with an absolute **Z-score > 2.5** for `Usage Stats` are flagged as "Anomalies" for maintenance review[cite: 2].

---

## 📂 Repository Structure
* `app.py`: The primary Streamlit application containing the full data pipeline and UI.
* `detailed_ev_charging_stations.csv`: The raw dataset used for analysis.
* `requirements.txt`: Environment dependencies including `pandas`, `scikit-learn`, `plotly`, and `mlxtend`.
* `README.md`: Comprehensive documentation and insights.

---

## ⚙️ Local Installation
1. Clone this repository.
2. Install dependencies: `pip install -r requirements.txt`
3. Run the application: `streamlit run app.py`

## 📚 References
* [Data-to-Viz: Visualization Guide](https://www.data-to-viz.com/)
* [Neptune AI: K-Means Clustering](https://neptune.ai/blog/k-means-clustering)
* [Scikit-learn Documentation](https://scikit-learn.org/)

---
**Facilitator:** Arul Jyoti# SmartCharging Analytics: Uncovering EV Behavior Patterns

 **Candidate Name:** Jeyaditya (JD) 
 **Candidate Registration Number:** 1000406 
 **CRS Name:** Artificial Intelligence 
 **Course Name:** Data Mining Summative Assessment 
 **School Name:** Jain Vidyalaya IB World School 

---

## 🚀 Live Application
Access the interactive dashboard here: 
 **[SmartCharging Analytics Dashboard](https://1000406jeyadityaaiy1sadataminingevchargersdetection.streamlit.app/)** 

---

## 📌 Project Rationale & Scope
 This project addresses the real-world challenge of optimizing Electric Vehicle (EV) infrastructure.  By analyzing global charging datasets, the application identifies usage trends, segments user behavior, and detects infrastructure anomalies to support data-driven expansion and pricing strategies.

### Primary Objectives:
*  **Behavioral Segmentation:** Grouping stations/users based on charging duration and frequency.
*  **Anomaly Detection:** Identifying faulty hardware or unusual spikes in demand.
*  **Trend Discovery:** Uncovering associations between station features (e.g., Renewable Energy) and high usage.
*  **Interactive Deployment:** Providing stakeholders with a Streamlit-based intelligence dashboard.

---

## 🛠️ Data Preparation & Preprocessing
 To ensure "Distinguished" data quality, the following steps were performed:
*  **Spatial Integrity Auditing:** Implemented a bounding-box algorithm to isolate "Atlantic Ocean" coordinates (0,0), ensuring geographic visualizations remain accurate.
*  **Handling Missing Values:** Managed null values in critical columns like `Reviews (Rating)`, `Renewable Energy Source`, and `Connector Types` using statistical imputation.
*  **Feature Encoding:** Converted categorical data such as `Charger Type` (AC Level 1, AC Level 2, DC Fast) and `Station Operator` into numerical formats for algorithmic processing.
* **Data Normalization:** Continuous variables (Cost, Usage, Capacity) were scaled to ensure scale-invariance using:
   $$x_{scaled} = \frac{x - \mu}{\sigma}$$ 

---

## 📊 Exploratory Data Analysis (EDA)
 The "Story of the Data" was uncovered through the following visualizations:
*  **Demand Distribution (Histograms):** Analyzed `Usage Stats` (avg users/day) to identify demand levels across the network.
*  **Temporal Growth (Line Charts):** Correlated `Usage Stats` with `Installation Year` to visualize the scaling of EV adoption.
*  **Operational Insights (Boxplots):** Evaluated `Cost (USD/kWh)` across different `Station Operators` to identify pricing anomalies.
*  **User Sentiment (Scatter Plots):** Discovered the relationship between `Reviews (Rating)` and station utilization.

---

## 🤖 Advanced Analytics & Machine Learning

### Behavioral Clustering (K-Means)
 Using the **Elbow Method** to determine the optimal number of clusters, the dataset was segmented into three distinct profiles:
1. **Daily Commuters:** Moderate, consistent usage patterns.
2. **Occasional Users:** Low-frequency, short-duration charging.
3.  **Heavy Users:** High-demand stations requiring prioritized maintenance.

### Association Rule Mining (Apriori)
 The system utilized the Apriori algorithm to identify high-confidence infrastructure patterns:
*  **Identified Rule:** `{DC Fast Chargers} + {Renewable Energy Source} → {Higher Average Users}`.
*  **Metrics:** Evaluated rules based on **Support, Confidence, and Lift** to ensure statistical significance.

### Anomaly Detection (Z-Score)
 To safeguard infrastructure, a statistical outlier detection model was implemented:
* **Formula:** $Z = \frac{x - \mu}{\sigma}$
*  **Threshold:** Stations with an absolute Z-score > 2.5 for usage or maintenance frequency are flagged for review.

---

## 📂 Repository Structure
*  `app.py`: Main Streamlit application file.
*  `requirements.txt`: List of necessary libraries (Pandas, Scikit-learn, etc.).
*  `ev_charging_data.csv`: The dataset used for the project.
*  `analysis_report.ipynb`: Detailed EDA and code implementation.
*  `README.md`: Project documentation and insights.

---

## ⚙️ Local Installation
1. Clone this repository.
2. Install dependencies: `pip install -r requirements.txt`
3. Run the app: `streamlit run app.py`

## 📚 References
*  [Data-to-Viz: Visualization Guide](https://www.data-to-viz.com/) 
*  [Neptune AI: K-Means Clustering](https://neptune.ai/blog/k-means-clustering) 
*  [Scikit-learn Documentation](https://scikit-learn.org/) 

---
 **Facilitator:** Arul Jyoti
