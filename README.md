# SmartCharging Analytics: Uncovering EV Behavior Patterns

**Candidate Name:** Jeyaditya
**Course:** Artificial Intelligence - Data Mining Summative Assessment

Link to access app - []

## Project Rationale
This project empowers stakeholders to leverage data mining techniques to solve real-world challenges within Electric Vehicle (EV) infrastructure. By exploring charging behavior across time-of-day, weekdays, and user profiles, this application provides actionable insights to improve station utilization and customer experience.

## Methodology & Architecture

### 1. Data Integrity and Spatial Auditing
Raw datasets often contain critical errors. During the initial exploratory phase, a significant anomaly was detected: several EV charging stations were mapped to coordinates in the middle of the Atlantic Ocean. 

Resolving this was a primary focus of the preprocessing stage. A rigorous spatial bounding-box algorithm was implemented to audit the latitude and longitude parameters. The system flags stations located at default (0,0) or known oceanic coordinates and isolates them. This ensures the integrity of the downstream Machine Learning models while providing stakeholders with a clear report of hardware reporting failures.

### 2. Behavioral Clustering (K-Means)
To group customers based on charging duration and frequency, K-Means clustering was applied. 
Prior to clustering, the continuous variables (Usage, Cost, and Capacity) were standardized using the following derivation to ensure scale-invariance:
$x_{scaled} = \frac{x - \mu}{\sigma}$

The stations were successfully segmented into three distinct behavioral profiles:
* Daily Commuters
* Occasional Users
* Heavy Users (Identified as high-priority expansion targets)

### 3. Statistical Anomaly Detection
To identify unusual consumption behaviors, such as overuse or abnormal charging durations, statistical z-score modeling was utilized. 
The derivation used to calculate the variance from the mean is:
$Z = \frac{x - \mu}{\sigma}$
Any station exhibiting a usage spike with an absolute Z-score greater than 2.5 is automatically flagged for maintenance review.

### 4. Association Rule Mining
The Apriori algorithm was deployed to discover hidden relationships between station features. By analyzing frequent itemsets, the system uncovered high-confidence rules, notably the strong association between DC Fast Chargers, Renewable Energy Sources, and higher average user demand.

## Deployment Details
The insights generated from this analysis are deployed via an interactive dashboard built on Streamlit Cloud. The interface allows users to seamlessly navigate through geographic intelligence maps, cluster visualizations, and raw data audits.

### Local Installation
1. Clone this repository.
2. Install the required dependencies: pip install -r requirements.txt
3. Execute the dashboard: streamlit run app.py

## Technologies Utilized
* **Data Processing:** Pandas, NumPy
* **Machine Learning:** Scikit-Learn (K-Means, StandardScaler), MLxtend (Apriori)
* **Visualization & Deployment:** Streamlit, Plotly Express

## Credits:
* Student name: A Jeyaditya
* Registration number: 1000406
* CRS Facillitator: Arul Jyoti
* School name: Jain Vidyalaya IB World School
