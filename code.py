# ==========================================================
# A DATA-DRIVEN ANALYSIS OF CRIME RATE
# AND ITS INFLUENCING FACTORS IN INDIA
# ==========================================================
# Authors:
# Suraj Subba (CSD25008)
# Somir Baruah (CSD25005)
# ==========================================================

# ==========================================================
# 1. IMPORT LIBRARIES
# ==========================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr, ttest_ind
from statsmodels.tsa.arima.model import ARIMA
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix



# ==========================================================
# 2. LOAD DATASET
# ==========================================================

df = pd.read_csv("dataset.csv")

print("First 5 Rows")
df.head()

# ==========================================================
# 3. BASIC INFORMATION
# ==========================================================

print("\nShape of Dataset")
print(df.shape)

print("\nColumns")
print(df.columns)

print("\nDataset Information")
print(df.info())


# ==========================================================
# 4. DATA CLEANING
# ==========================================================

# GDP column cleaning
df['gdp(billions)'] = (
    df['gdp(billions)']
    .astype(str)
    .str.replace(',', '')
)

df['gdp(billions)'] = pd.to_numeric(
    df['gdp(billions)'],
    errors='coerce'
)

# Remove duplicates
print("\nDuplicate Rows:", df.duplicated().sum())

df.drop_duplicates(inplace=True)

# Missing values
print("\nMissing Values")
print(df.isnull().sum())

# Fill missing numeric values
df.fillna(df.median(numeric_only=True), inplace=True)

# ==========================================================
# DESCRIPTIVE STATISTICS
# ==========================================================

df.describe().T

# ==========================================================
# 5. EXPLORATORY DATA ANALYSIS
# ==========================================================

# ==========================================================
# 5.1 Correlation Heatmap
# ==========================================================

numeric_df = df.select_dtypes(include=['int64', 'float64'])

plt.figure(figsize=(16,10))

sns.heatmap(
    numeric_df.corr(),
    annot=True,
    cmap='coolwarm'
)

plt.title("Correlation Heatmap")
plt.show()

# ==========================================================
# 5.2 Top Crime States
# ==========================================================

crime_state = (
    df.groupby('STATE/UT')['total crimes']
    .mean()
    .sort_values(ascending=False)
    .head(10)
)

plt.figure(figsize=(12,6))

crime_state.plot(kind='bar')

plt.title("Top 10 States by Total Crimes")
plt.ylabel("Average Crimes")
plt.xticks(rotation=45)

plt.show()

# ==========================================================
# 5.3 Poverty Distribution
# ==========================================================

plt.figure(figsize=(10,6))

sns.histplot(
    df['poverty rate (%)'],
    bins=15,
    kde=True
)

plt.title("Poverty Rate Distribution")
plt.show()

# ==========================================================
# 5.4 Literacy Distribution
# ==========================================================

plt.figure(figsize=(10,6))

sns.histplot(
    df['literacy rate(%)'],
    bins=15,
    kde=True
)

plt.title("Literacy Rate Distribution")
plt.show()

# ==========================================================
# 5.4 Literacy Distribution
# ==========================================================

plt.figure(figsize=(10,6))

sns.histplot(
    df['unemployment rate'],
    bins=15,
    kde=True
)

plt.title("Unemployment Rate Distribution")
plt.show()

# ==========================================================
# 6. CORRELATION ANALYSIS
# ==========================================================

# ==========================================================
# LITERACY VS CRIME
# ==========================================================

plt.figure(figsize=(8,6))

sns.scatterplot(
    x='literacy rate(%)',
    y='total crimes',
    data=df,
    hue='Year'
)

plt.title('Literacy Rate vs Total Crimes')
plt.show()

# ==========================================================
# POVERTY RATE VS CRIME
# ==========================================================

plt.figure(figsize=(8,6))

sns.scatterplot(
    x='poverty rate (%)',
    y='total crimes',
    data=df,
    hue='Year'
)

plt.title('Poverty Rate vs Total Crimes')
plt.show()

# ==========================================================
# Unemployment RATE VS CRIME
# ==========================================================

plt.figure(figsize=(8,6))

sns.scatterplot(
    x='unemployment rate',
    y='total crimes',
    data=df,
    hue='Year'
)

plt.title('Unemployment Rate vs Total Crimes')
plt.show()

# ==========================================================
# Unemployment RATE VS CRIME
# ==========================================================

plt.figure(figsize=(8,6))

sns.scatterplot(
    x='gdp(billions)',
    y='total crimes',
    data=df,
    hue='Year'
)

plt.title('GDP vs Total Crimes')
plt.show()

# ==========================================================
# Unemployment RATE VS CRIME
# ==========================================================

plt.figure(figsize=(8,6))

sns.scatterplot(
    x='Population(in lakhs)',
    y='total crimes',
    data=df,
    hue='Year'
)

plt.title('Population vs Total Crimes')
plt.show()

# ==========================================================
# Unemployment RATE VS CRIME
# ==========================================================

plt.figure(figsize=(8,6))

sns.scatterplot(
    x='population deprived of schooling',
    y='total crimes',
    data=df,
    hue='Year'
)

plt.title('Schooling deprivation vs Total Crimes')
plt.show()

# ==========================================================
# Unemployment RATE VS CRIME
# ==========================================================

plt.figure(figsize=(8,6))

sns.scatterplot(
    x='population deprived of Nutrition',
    y='total crimes',
    data=df,
    hue='Year'
)

plt.title('Nutrition deprivation vs Total Crimes')
plt.show()

# ==========================================================
# 6.3 Multi-Factor Correlation Analysis
# ==========================================================

features = [
    'poverty rate (%)',
    'literacy rate(%)',
    'unemployment rate',
    'gdp(billions)',
    'Population(in lakhs)',
    'population deprived of schooling',
    'population deprived of Nutrition'
]

print("\nFeature Correlation with Crime")

for feature in features:

    corr, p = pearsonr(
        df[feature],
        df['total crimes']
    )

    print(f"{feature}")
    print("Correlation:", corr)
    print("P-value:", p)
    print()

from scipy.stats import pearsonr
import pandas as pd
import matplotlib.pyplot as plt

# Features list
features = [
    'poverty rate (%)',
    'literacy rate(%)',
    'unemployment rate',
    'gdp(billions)',
    'Population(in lakhs)',
    'population deprived of schooling',
    'population deprived of Nutrition'
]

# Store correlation results
correlation_results = []

# Calculate correlations
for feature in features:
    
    corr, p = pearsonr(
        df[feature],
        df['total crimes']
    )
    
    correlation_results.append({
        'Feature': feature,
        'Correlation': corr,
        'P-value': p
    })

# Create dataframe
corr_df = pd.DataFrame(correlation_results)

# Sort by correlation strength
corr_df = corr_df.sort_values(by='Correlation')

# Plot
plt.figure(figsize=(10,6))

bars = plt.barh(
    corr_df['Feature'],
    corr_df['Correlation']
)

# Add correlation values on bars
for i, value in enumerate(corr_df['Correlation']):
    plt.text(
        value,
        i,
        f"{value:.2f}",
        va='center'
    )

plt.xlabel('Pearson Correlation')
plt.ylabel('Features')
plt.title('Feature Correlation with Total Crimes')

plt.grid(axis='x', linestyle='--', alpha=0.5)

plt.show()

# ==========================================================
# CRIME AGAINST WOMEN
# ==========================================================

women_crime = df.groupby('STATE/UT')['crime against women'].mean()

women_crime = women_crime.sort_values(ascending=False)

plt.figure(figsize=(14,7))
women_crime.head(10).plot(kind='bar', color='red')

plt.title('Top States in Crime Against Women')
plt.ylabel('Cases')
plt.xticks(rotation=75)
plt.show()

# ==========================================================
# CRIME AGAINST CHILDREN
# ==========================================================

child_crime = df.groupby('STATE/UT')['crime against children'].mean()

child_crime = child_crime.sort_values(ascending=False)

plt.figure(figsize=(14,7))
child_crime.head(10).plot(kind='bar', color='orange')

plt.title('Top States in Crime Against Children')
plt.ylabel('Cases')
plt.xticks(rotation=75)
plt.show()

# ==========================================================
# 7. CRIME ANALYTICS
# ==========================================================

# ==========================================================
# 7.1 Crime Category Analysis
# ==========================================================

crime_columns = [
    'Murder',
    'Kidnapping and Abduction',
    'crime against women',
    'crime against children',
    'crime committed by juveniles'
]

crime_totals = df[crime_columns].sum()

plt.figure(figsize=(12,6))

crime_totals.plot(kind='bar')

plt.title("Crime Category Comparison")
plt.ylabel("Total Cases")
plt.xticks(rotation=20)

plt.show()

# ==========================================================
# 7.2 State-wise Crime Heatmap
# ==========================================================

state_crime = df.pivot_table(
    values='total crimes',
    index='STATE/UT',
    columns='Year'
)

plt.figure(figsize=(12,10))

sns.heatmap(
    state_crime,
    cmap='Reds'
)

plt.title("State-wise Crime Heatmap")
plt.show()


# ==========================================================
# YEARLY CRIME TREND
# ==========================================================

yearly_crime = df.groupby('Year')['total crimes'].sum()

plt.figure(figsize=(8,5))
plt.plot(yearly_crime.index, yearly_crime.values, marker='o')

plt.title('Crime Trend Over Years')
plt.xlabel('Year')
plt.ylabel('Total Crimes')
plt.grid(True)
plt.show()

crime_columns = [
    'Murder',
    'Kidnapping and Abduction',
    'crime against women',
    'crime against children',
    'crime committed by juveniles'
]

crime_year = df.groupby('Year')[crime_columns].sum()

for col in crime_columns:
    plt.figure(figsize=(8,4))
    crime_year[col].plot(marker='o')

    plt.title(f"{col} Trend Over Years")
    plt.ylabel("Number of Crimes")
    plt.xlabel("Year")

    plt.grid(True)
    plt.show()

# ==========================================================
# 8. FEATURE ENGINEERING
# ==========================================================

# ==========================================================
# 8.1 Crime Per Lakh Population
# ==========================================================

df['Crime per Lakh Population'] = (
    df['total crimes'] /
    df['Population(in lakhs)']
)

df[['STATE/UT','Crime per Lakh Population']].head()

# ==========================================================
# CRIME RATE PER POPULATION
# ==========================================================

rate_df = df.groupby('STATE/UT')['Crime per Lakh Population'].mean()
rate_df = rate_df.sort_values(ascending=False)

plt.figure(figsize=(14,7))
rate_df.head(15).plot(kind='bar', color='purple')

plt.title('Crime Rate Per Lakh Population')
plt.ylabel('Crime Rate')
plt.xticks(rotation=75)
plt.show()

# ==========================================================
# 8.2 GDP Per Capita
# ==========================================================

df['GDP per Capita'] = (
    df['gdp(billions)'] /
    df['Population(in lakhs)']
)

print(df[[
    'STATE/UT',
    'GDP per Capita'
]].head())

# ==========================================================
# 9. MACHINE LEARNING MODELS
# ==========================================================

ml_features = [
    'poverty rate (%)',
    'population deprived of schooling',
    'population deprived of Nutrition',
    'unemployment rate',
    'literacy rate(%)',
    'gdp(billions)',
    'Population(in lakhs)',
    'Crime per Lakh Population'
]

X = df[ml_features]
y = df['total crimes']

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# ==========================================================
# 9.1 Linear Regression
# ==========================================================

lr = LinearRegression()

lr.fit(X_train, y_train)

pred = lr.predict(X_test)

print("\nLINEAR REGRESSION")

print("MAE:",
      mean_absolute_error(y_test, pred))

print("RMSE:",
      np.sqrt(mean_squared_error(y_test, pred)))

print("R2 Score:",
      r2_score(y_test, pred))


# ==========================================================
# 9.2 Random Forest
# ==========================================================

rf = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

rf.fit(X_train, y_train)

rf_pred = rf.predict(X_test)

print("\nRANDOM FOREST")

print("MAE:",
      mean_absolute_error(y_test, rf_pred))

print("RMSE:",
      np.sqrt(mean_squared_error(y_test, rf_pred)))

print("R2 Score:",
      r2_score(y_test, rf_pred))

# ==========================================================
# 9.3 Feature Importance
# ==========================================================

importance = pd.Series(
    rf.feature_importances_,
    index=ml_features
)

importance = importance.sort_values(
    ascending=False
)

plt.figure(figsize=(10,6))

importance.plot(kind='bar')

plt.title("Feature Importance")
plt.ylabel("Importance Score")

plt.show()

print("\nFeature Importance")
print(importance)


# ==========================================================
# OUTLIER DETECTION
# ==========================================================

features = df[[
    'total crimes',
    'population deprived of schooling',
    'poverty rate (%)',
    'population deprived of Nutrition',
    'unemployment rate',
    'gdp(billions)',
    'literacy rate(%)',
    'Population(in lakhs)',
    'total crimes',
    'Murder',
    'Kidnapping and Abduction',
    'crime against women',
    'crime against children',
    'crime committed by juveniles'
]]

iso = IsolationForest(contamination=0.1, random_state=42)

df['outlier'] = iso.fit_predict(features)

outliers = df[df['outlier'] == -1]

print(outliers[['STATE/UT', 'total crimes']])

# ==========================================================
# K-MEANS CLUSTERING
# ==========================================================

features = [
    "poverty rate (%)",
    "unemployment rate",
    "literacy rate(%)",
    "Population(in lakhs)",
    "total crimes",
    "crime against women",
    "crime against children"
]

# Standardize data
scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

# KMeans
kmeans = KMeans(
    n_clusters=3,
    random_state=42
)

clusters = kmeans.fit_predict(X_scaled)

# Add cluster labels
df.loc[X.index, "Cluster"] = clusters

# Show results
print(df[["STATE/UT", "Cluster"]].head())

# Visualization

plt.figure(figsize=(10,7))

# Plot each cluster separately
for cluster in sorted(df["Cluster"].dropna().unique()):
    
    cluster_data = df[df["Cluster"] == cluster]
    
    plt.scatter(
        cluster_data["poverty rate (%)"],
        cluster_data["total crimes"],
        label=f"Cluster {int(cluster)}"
    )

plt.title("KMeans Clustering of States")

plt.xlabel("Poverty Rate (%)")

plt.ylabel("Total Crimes")

# Add legend
plt.legend()

plt.show()

# Add cluster labels to dataframe
df['Cluster'] = kmeans.labels_

# Display states in each cluster
for cluster in sorted(df['Cluster'].unique()):
    print(f"\n===== Cluster {cluster} =====")
    print(df[df['Cluster'] == cluster]['STATE/UT'].values)

K-Means clustering was applied to group states with similar socio-economic 
and crime characteristics. The clustering analysis helped identify high-risk 
regions and revealed hidden patterns in crime distribution across India.
    
K-Means clustering can be very useful in your crime and socio-economic dataset because 
it helps you group states or regions with similar crime and socio-economic characteristics 
without needing predefined labels.

Instead of predicting something, K-Means helps answer questions like:

Which states have similar crime patterns?
Which regions are socio-economically similar?
Which states belong to high-risk crime clusters?
Which states have similar combinations of poverty, unemployment, and crime?

# Selected columns
columns = [
    "poverty rate (%)",
    "unemployment rate",
    "literacy rate(%)",
    "Population(in lakhs)",
    "total crimes"
]

sns.pairplot(df[columns])
plt.show()

sns.pairplot(
    df,
    vars=columns,
    hue="Cluster"
)

plt.show()

Pairplot visualization was used to examine pairwise relationships among socio-economic and 
crime-related variables. The analysis helped identify trends, correlations, clusters, and 
potential outliers across different states. 
A pairplot is very useful in your crime and socio-economic dataset because it helps us
visualize relationships between multiple variables simultaneously. Instead of checking one 
graph at a time, a pairplot provides a complete overview of:

correlations
trends
distributions
clusters
outliers

across all important factors in a single visualization.

For our dataset, which contains variables such as:

poverty rate
unemployment rate
literacy rate
population
total crimes
crimes against women
crimes against children

a pairplot helps identify how these variables interact with each other.

# Group yearly total crimes
yearly_crime = df.groupby('Year')['total crimes'].sum()

import matplotlib.pyplot as plt

plt.figure(figsize=(10,5))

plt.plot(
    yearly_crime.index,
    yearly_crime.values,
    marker='o',
    label='Historical'
)

plt.xlabel('Year')
plt.ylabel('Total Crimes')
plt.title('Historical Crime Trend')

plt.legend()
plt.grid(True)

plt.show()

from statsmodels.tsa.arima.model import ARIMA

# Train model
model = ARIMA(yearly_crime, order=(1,1,1))

model_fit = model.fit()

# Forecast next 5 years
forecast = model_fit.forecast(steps=5)


# Future years
future_years = range(
    yearly_crime.index.max() + 1,
    yearly_crime.index.max() + 6
)

plt.figure(figsize=(10,6))

# Historical data
plt.plot(
    yearly_crime.index,
    yearly_crime.values,
    marker='o',
    linewidth=2,
    label='Historical'
)

# Forecast data
plt.plot(
    future_years,
    forecast,
    marker='o',
    linewidth=2,
    label='Forecast'
)

plt.xlabel('Year')
plt.ylabel('Total Crimes')

plt.title('ARIMA Crime Forecasting')

plt.legend()
plt.grid(True)

plt.show()

print("ARIMA(1,1,1) Model")
print("Forecasts next 5 years based on historical crime patterns.")

if forecast.iloc[-1] > yearly_crime.values[-1]:
    print("Forecast suggests an increase in crime trend.")
else:
    print("Forecast suggests a decrease in crime trend.")

from sklearn.preprocessing import MinMaxScaler
# --------------------------------------------
# FEATURES FOR CRIME RISK INDEX
# --------------------------------------------

crime_features = [
    'total crimes',
    'Murder',
    'Kidnapping and Abduction',
    'crime against women',
    'crime against children',
    'crime committed by juveniles',
    'poverty rate (%)',
    'unemployment rate'
]

# --------------------------------------------
# NORMALIZE FEATURES
# --------------------------------------------

scaler = MinMaxScaler()

scaled_features = scaler.fit_transform(df[crime_features])

# --------------------------------------------
# ASSIGN WEIGHTS
# --------------------------------------------

weights = np.array([
    0.30,  # total crimes
    0.15,  # murder
    0.10,  # kidnapping
    0.15,  # crimes against women
    0.10,  # crimes against children
    0.05,  # juvenile crimes
    0.10,  # poverty
    0.05   # unemployment
])

# --------------------------------------------
# CALCULATE CRIME RISK INDEX
# --------------------------------------------

df['Crime Risk Index'] = (
    scaled_features * weights
).sum(axis=1) * 100

# --------------------------------------------
# AVERAGE RISK BY STATE
# --------------------------------------------

state_risk = (
    df.groupby('STATE/UT')['Crime Risk Index']
    .mean()
    .reset_index()
)

# Sort from highest risk to lowest
state_risk = state_risk.sort_values(
    by='Crime Risk Index',
    ascending=False
)

# --------------------------------------------
# TOP 5 SAFEST STATES
# --------------------------------------------

safest_states = (
    state_risk.sort_values(by='Crime Risk Index')
    .head(5)
)

print("\nTop 5 Safest States:")
print(safest_states)

# --------------------------------------------
# ANOMALY DETECTION
# --------------------------------------------

anomaly_features = [
    'total crimes',
    'Murder',
    'Kidnapping and Abduction',
    'crime against women',
    'crime against children',
    'crime committed by juveniles',
    'poverty rate (%)',
    'unemployment rate'
]

# Isolation Forest Model
iso = IsolationForest(
    contamination=0.05,
    random_state=42
)

# Fit model
df['Anomaly'] = iso.fit_predict(df[anomaly_features])

# -1 = anomaly
anomalies = df[df['Anomaly'] == -1]

print("\nDetected Anomalies:")
print(
    anomalies[
        ['STATE/UT', 'Year', 'Crime Risk Index']
    ]
)

# --------------------------------------------
# VISUALIZATION
# --------------------------------------------

top10 = state_risk.head(10)

plt.figure(figsize=(10, 6))

plt.barh(
    top10['STATE/UT'][::-1],
    top10['Crime Risk Index'][::-1]
)

plt.xlabel("Crime Risk Index")
plt.ylabel("States")
plt.title("Top 10 States - Crime Risk Index")

plt.grid(axis='x', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()


# ==========================================================
# 18. FINAL INSIGHTS
# ==========================================================

print("\nSTATE WITH HIGHEST CRIME")

print(
    df.groupby('STATE/UT')['total crimes']
    .mean()
    .idxmax()
)

print("\nSTATE WITH HIGHEST LITERACY")

print(
    df.groupby('STATE/UT')['literacy rate(%)']
    .mean()
    .idxmax()
)

print("\nSTATE WITH HIGHEST POVERTY")

print(
    df.groupby('STATE/UT')['poverty rate (%)']
    .mean()
    .idxmax()
)


# ==========================================================
# STATE RANKING
# ==========================================================

ranking_df = df.groupby('STATE/UT')['total crimes'].mean().reset_index()

ranking_df['Rank'] = ranking_df['total crimes'].rank(ascending=False)

ranking_df.sort_values('Rank').head(10)

# 32. Recommendations Section

## Policy Recommendations

### 1. Improve Education

Higher literacy may help reduce crime.

### 2. Poverty Reduction Programs

States with higher poverty may need stronger welfare support.

### 3. Women Safety Programs

Crime against women needs stricter law enforcement.

### 4. Child Protection Measures

High child crime regions need awareness and monitoring.

### 5. Employment Opportunities

Reducing unemployment may lower crime growth.



# 33. Final Conclusion

## Conclusion

This project successfully analyzed crime trends across Indian states using advanced Data Science and Machine Learning techniques.

Major findings include:

* Population significantly affects total crime volume.
* Poverty and unemployment influence crime patterns.
* Literacy rate shows relationship with crime reduction.
* Certain states consistently show higher crime concentration.
* Random Forest provided better prediction performance than Linear Regression.
* Clustering revealed groups of states with similar socio-economic and crime characteristics.

The project demonstrates practical applications of:

* Data Cleaning
* Exploratory Data Analysis
* Statistical Testing
* Machine Learning
* Clustering
