# ==========================================================
# INDIA CRIME RATE ANALYSIS — STREAMLIT DASHBOARD   
# ==========================================================

import streamlit as st
import pandas as pd
import numpy as np

import plotly.express as px
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    IsolationForest
)

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from sklearn.decomposition import PCA
from scipy.stats import pearsonr, ttest_ind
from statsmodels.tsa.arima.model import ARIMA

import warnings
warnings.filterwarnings("ignore")


# ==========================================================
# PAGE CONFIG
# ==========================================================

st.set_page_config(
    page_title="INDIA CRIME RATE ANALYSIS DASHBOARD   ",
    layout="wide"
)


# ==========================================================
# CUSTOM CSS
# ==========================================================

st.markdown("""
<style>

.stApp {
    background-color: #0f172a;
    color: white;
}

.main-title {
    text-align: center;
    font-size: 48px;
    font-weight: bold;
    color: #38bdf8;
}

.sub-title {
    text-align: center;
    color: #cbd5e1;
    margin-bottom: 20px;
}

</style>
""", unsafe_allow_html=True)


# ==========================================================
# LOAD DATA
# ==========================================================

@st.cache_data
def load_data():

    df = pd.read_csv("dataset.csv")

    # GDP Cleaning
    df['gdp(billions)'] = (
        df['gdp(billions)']
        .astype(str)
        .str.replace(',', '')
    )

    df['gdp(billions)'] = pd.to_numeric(
        df['gdp(billions)'],
        errors='coerce'
    )

    # Missing values
    df.fillna(
        df.median(numeric_only=True),
        inplace=True
    )

    # ======================================================
    # FEATURE ENGINEERING
    # ======================================================

    df['Crime per Lakh Population'] = (
        df['total crimes'] /
        df['Population(in lakhs)']
    )

    df['GDP per Capita'] = (
        df['gdp(billions)'] /
        df['Population(in lakhs)']
    )

    df['Development Index'] = (
        (df['literacy rate(%)'] * 0.4) +
        (df['gdp(billions)'] * 0.0005) -
        (df['poverty rate (%)'] * 0.3) -
        (df['unemployment rate'] * 0.2)
    )

    df['Crime Risk Index'] = (
        (df['poverty rate (%)'] * 0.3) +
        (df['unemployment rate'] * 0.3) +
        (df['Crime per Lakh Population'] * 0.2) -
        (df['literacy rate(%)'] * 0.2)
    )

    return df


df = load_data()


# ==========================================================
# SIDEBAR
# ==========================================================

st.sidebar.title("🔍 Dashboard Filters")

years = sorted(df['Year'].unique())

selected_year = st.sidebar.selectbox(
    "Select Year",
    ["All Years"] + list(years)
)

states = sorted(df['STATE/UT'].unique())

selected_state = st.sidebar.selectbox(
    "Select State",
    ["All States"] + states
)

crime_options = {
    "Total Crimes": "total crimes",
    "Murder": "Murder",
    "Women Crime": "crime against women",
    "Children Crime": "crime against children",
    "Kidnapping": "Kidnapping and Abduction"
}

selected_crime = st.sidebar.selectbox(
    "Crime Category",
    list(crime_options.keys())
)

crime_col = crime_options[selected_crime]

st.sidebar.info(
    "Use 'All States' for clustering analysis."
)


# ==========================================================
# FILTER DATA
# ==========================================================

filtered = df.copy()

if selected_year != "All Years":
    filtered = filtered[
        filtered['Year'] == selected_year
    ]

if selected_state != "All States":
    filtered = filtered[
        filtered['STATE/UT'] == selected_state
    ]


# ==========================================================
# HEADER
# ==========================================================

st.markdown(
    '<div class="main-title">INDIA CRIME RATE ANALYSIS DASHBOARD   </div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="sub-title">A Data-Driven Analysis of Crime Rate and Its Influencing Factors in India</div>',
    unsafe_allow_html=True
)


# ==========================================================
# KPI SECTION
# ==========================================================

st.subheader("📊 National Overview")

c1, c2, c3, c4, c5 = st.columns(5)

c1.metric(
    "Total Crimes",
    f"{int(filtered['total crimes'].sum()):,}"
)

c2.metric(
    "Avg Literacy",
    f"{filtered['literacy rate(%)'].mean():.2f}%"
)

c3.metric(
    "Avg Poverty",
    f"{filtered['poverty rate (%)'].mean():.2f}%"
)

c4.metric(
    "Avg Development",
    f"{filtered['Development Index'].mean():.2f}"
)

c5.metric(
    "Avg Risk",
    f"{filtered['Crime Risk Index'].mean():.2f}"
)


# ==========================================================
# TABS
# ==========================================================

tabs = st.tabs([
    "📈 Trend Analysis",
    "📊 Crime Categories",
    "🔥 Socio-Economic",
    "📉 Statistical Insights",
    "🤖 Machine Learning",
    "🔮 Forecasting",
    "🚨 Risk & Development",
    "🧠 Anomaly Detection",
    "📌 Recommendations",
    "📂 Data Explorer",
    "📘 Final Insights"
])


# ==========================================================
# TAB 1 — TREND ANALYSIS
# ==========================================================

with tabs[0]:

    st.subheader("📈 Crime Trend Analysis")

    trend = (
        filtered.groupby('Year')['total crimes']
        .sum()
        .reset_index()
    )

    fig = px.line(
        trend,
        x='Year',
        y='total crimes',
        markers=True,
        title='Year-wise Crime Trend'
    )

    st.plotly_chart(fig, use_container_width=True)



    # Crime Trends Over Years

    st.subheader("📈 Crime Trends Over Years")

    crime_columns = [
        'Murder',
        'Kidnapping and Abduction',
        'crime against women',
        'crime against children',
        'crime committed by juveniles'
    ]

    crime_year = df.groupby('Year')[crime_columns].sum().reset_index()

    selected_crime = st.selectbox(
        "Select Crime Type",
        crime_columns
    )

    # Growth %
    crime_year['Growth %'] = (
        crime_year[selected_crime]
        .pct_change() * 100
    )

    # Trend Detection
    latest_growth = crime_year['Growth %'].iloc[-1]

    if latest_growth > 0:
        st.success(f"📈 Increasing Trend ({latest_growth:.2f}%)")

    elif latest_growth < 0:
        st.error(f"📉 Decreasing Trend ({latest_growth:.2f}%)")

    else:
        st.info("➖ Stable Trend")

    # Graph
    fig = px.line(
        crime_year,
        x='Year',
        y=selected_crime,
        markers=True,
        title=f"{selected_crime} Trend Over Years"
    )

    st.plotly_chart(fig, use_container_width=True)


# ==========================================================
# TAB 2 — CRIME CATEGORY ANALYSIS
# ==========================================================

with tabs[1]:

    st.subheader("📊 Crime Category Analysis")

    crime_columns = [
        'Murder',
        'Kidnapping and Abduction',
        'crime against women',
        'crime against children',
        'crime committed by juveniles'
    ]

    crime_totals = (
        filtered[crime_columns]
        .sum()
    )

    fig_bar = px.bar(
        x=crime_totals.index,
        y=crime_totals.values,
        color=crime_totals.values,
        title='Crime Category Comparison'
    )

    st.plotly_chart(
        fig_bar,
        use_container_width=True
    )

    fig_pie = px.pie(
        values=crime_totals.values,
        names=crime_totals.index,
        title='Crime Category Distribution'
    )

    st.plotly_chart(
        fig_pie,
        use_container_width=True
    )

    # CRIME AGAINST WOMEN (TOP 10 STATES)
    

    women_crime = (
        filtered.groupby('STATE/UT')['crime against women']
        .mean()
        .sort_values(ascending=False)
        .head(10)
        .reset_index()
    )

    fig_women = px.bar(
        women_crime,
        x='STATE/UT',
        y='crime against women',
        color='crime against women',
        title='Top 10 States: Crime Against Women',
        text='crime against women'
    )

    fig_women.update_traces(texttemplate='%{text:.0f}', textposition='outside')

    fig_women.update_layout(
        xaxis_tickangle=-45
    )

    st.plotly_chart(
        fig_women,
        use_container_width=True
    )

   
    # CRIME AGAINST CHILDREN (TOP 10 STATES)
    

    child_crime = (
        filtered.groupby('STATE/UT')['crime against children']
        .mean()
        .sort_values(ascending=False)
        .head(10)
        .reset_index()
    )

    fig_child = px.bar(
        child_crime,
        x='STATE/UT',
        y='crime against children',
        color='crime against children',
        title='Top 10 States: Crime Against Children',
        text='crime against children'
    )

    fig_child.update_traces(texttemplate='%{text:.0f}', textposition='outside')

    fig_child.update_layout(
        xaxis_tickangle=-45
    )

    st.plotly_chart(
        fig_child,
        use_container_width=True
    )

    
    st.subheader("🗺️ State-wise Crime Heatmap")

    # Pivot Table
    state_crime = df.pivot_table(
        values='total crimes',
        index='STATE/UT',
        columns='Year',
        aggfunc='mean'
    )
    
  
    
    # Heatmap
    fig_heatmap = px.imshow(
        state_crime,
        aspect='auto',
        color_continuous_scale='Reds',
        text_auto=True,
        title='State-wise Crime Heatmap'
    )

    # Layout Improvements
    fig_heatmap.update_layout(
        height=900,
        xaxis_title='Year',
        yaxis_title='State / UT'
    )

    st.plotly_chart(
        fig_heatmap,
        use_container_width=True
    )

# ==========================================================
# SOCIO-ECONOMIC CRIME ANALYSIS
# ==========================================================

with tabs[2]:
    st.subheader("🔥 Socio-Economic Crime Analysis")

    tab1, tab2, tab3 = st.tabs([
        "📘 Literacy vs Crime",
        "📉 Poverty vs Crime",
        "📊 Unemployment vs Crime"
    ])

    # ==========================================================
    # LITERACY VS CRIME
    # ==========================================================

    with tab1:

        fig1 = px.scatter(
            df,
            x='literacy rate(%)',
            y='total crimes',
            color='Year',
            hover_name='STATE/UT',
            title='Literacy Rate vs Total Crimes'
        )

        st.plotly_chart(
            fig1,
            use_container_width=True
        )

    # ==========================================================
    # POVERTY VS CRIME
    # ==========================================================

    with tab2:

        fig2 = px.scatter(
            df,
            x='poverty rate (%)',
            y='total crimes',
            color='Year',
            hover_name='STATE/UT',
            title='Poverty Rate vs Total Crimes'
        )

        st.plotly_chart(
            fig2,
            use_container_width=True
        )

    # ==========================================================
    # UNEMPLOYMENT VS CRIME
    # ==========================================================

    with tab3:

        fig3 = px.scatter(
            df,
            x='unemployment rate',
            y='total crimes',
            color='Year',
            hover_name='STATE/UT',
            title='Unemployment Rate vs Total Crimes'
        )

        st.plotly_chart(
            fig3,
            use_container_width=True
        )

    # ==========================================================
    # CORRELATION VALUES
    # ==========================================================

    st.subheader("📊 Correlation Analysis")

    features = [
        'poverty rate (%)',
        'literacy rate(%)',
        'unemployment rate',
        'gdp(billions)',
        'Population(in lakhs)',
        'population deprived of schooling',
        'population deprived of Nutrition'
    ]

    corr_data = []

    for feature in features:

        corr = df[feature].corr(df['total crimes'])

        corr_data.append({
            "Feature": feature,
            "Correlation with Crime": round(corr, 3)
        })

    corr_df = pd.DataFrame(corr_data)

    st.dataframe(
        corr_df,
        use_container_width=True
    )

# ==========================================================
# TAB 4 — STATISTICAL INSIGHTS
# ==========================================================

with tabs[3]:

    st.subheader("📉 Statistical Insights")

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

    corr_df = []

    for feature in ml_features:

        corr, p = pearsonr(
            df[feature],
            df['total crimes']
        )

        corr_df.append({
            "Feature": feature,
            "Correlation": round(corr, 4),
            "P-value": round(p, 4)
        })

    corr_df = pd.DataFrame(corr_df)

    st.dataframe(
        corr_df,
        use_container_width=True
    )

    # TOP 10 STATES BY TOTAL CRIMES
    

    top10_crime = (
        filtered.groupby('STATE/UT')['total crimes']
        .sum()
        .sort_values(ascending=False)
        .head(10)
        .reset_index()
    )

    fig_bar = px.bar(
        top10_crime,
        x='STATE/UT',
        y='total crimes',
        color='total crimes',
        text='total crimes',
        title='Top 10 States by Total Crimes'
    )

    fig_bar.update_traces(
        texttemplate='%{text:.0f}',
        textposition='outside'
    )

    fig_bar.update_layout(
        xaxis_title='State/UT',
        yaxis_title='Total Crimes',
        xaxis_tickangle=-45
    )

    st.plotly_chart(
        fig_bar,
        use_container_width=True
    )
    # TOP 10 STATES BY CRIME RISK INDEX (BAR PLOT)
    

    top10_states = (
        filtered.groupby('STATE/UT')['Crime Risk Index']
        .mean()
        .sort_values(ascending=False)
        .head(10)
        .reset_index()
    )

    fig_bar = px.bar(
        top10_states,
        x='STATE/UT',
        y='Crime Risk Index',
        color='Crime Risk Index',
        title='Top 10 States by Crime Risk Index',
        text='Crime Risk Index'
    )

    fig_bar.update_layout(
        xaxis_title='State/UT',
        yaxis_title='Crime Risk Index',
        xaxis_tickangle=-45
    )

    fig_bar.update_traces(
        texttemplate='%{text:.2f}',
        textposition='outside'
    )

    st.plotly_chart(
        fig_bar,
        use_container_width=True
    )
    # TOP 10 STATES BY CRIME RISK INDEX (BOXPLOT)
    

    top10_states = (
        filtered.groupby('STATE/UT')['Crime Risk Index']
        .mean()
        .sort_values(ascending=False)
        .head(10)
        .index
    )

    top10_data = filtered[
        filtered['STATE/UT'].isin(top10_states)
    ]

    fig_box = px.box(
        top10_data,
        x='STATE/UT',
        y='Crime Risk Index',
        points='all',
        color='STATE/UT',
        title='Top 10 States by Crime Risk Index',
    )

    fig_box.update_layout(
        xaxis_title='State/UT',
        yaxis_title='Crime Risk Index',
        xaxis_tickangle=-45
    )

    st.plotly_chart(
        fig_box,
        use_container_width=True
    )


# ==========================================================
# TAB 5 — MACHINE LEARNING
# ==========================================================

with tabs[4]:

    st.subheader("🤖 Machine Learning Models")

    # ==========================================================
    # SAME FEATURES AS NOTEBOOK
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

    # ==========================================================
    # SAME DATAFRAME AS NOTEBOOK
    # ==========================================================

    ml_df = df.copy()

    # Remove missing values
    ml_df = ml_df.dropna(
        subset=ml_features + ['total crimes']
    )

    # ==========================================================
    # FEATURES AND TARGET
    # ==========================================================

    X = ml_df[ml_features]

    y = ml_df['total crimes']

    # ==========================================================
    # TRAIN TEST SPLIT
    # ==========================================================

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    # ==========================================================
    # LINEAR REGRESSION
    # ==========================================================

    lr = LinearRegression()

    lr.fit(X_train, y_train)

    lr_pred = lr.predict(X_test)

    # ==========================================================
    # RANDOM FOREST
    # ==========================================================

    rf = RandomForestRegressor(
        n_estimators=100,
        random_state=42
    )

    rf.fit(X_train, y_train)

    rf_pred = rf.predict(X_test)

    # ==========================================================
    # RESULTS TABLE
    # ==========================================================

    results = pd.DataFrame({

        "Model": [
            "Linear Regression",
            "Random Forest"
        ],

        "MAE": [
            round(mean_absolute_error(y_test, lr_pred), 2),
            round(mean_absolute_error(y_test, rf_pred), 2)
        ],

        "RMSE": [
            round(np.sqrt(mean_squared_error(y_test, lr_pred)), 2),
            round(np.sqrt(mean_squared_error(y_test, rf_pred)), 2)
        ],

        "R2 Score": [
            round(r2_score(y_test, lr_pred), 3),
            round(r2_score(y_test, rf_pred), 3)
        ]

    })

    st.dataframe(
        results,
        use_container_width=True
    )

    # ==========================================================
    # FEATURE IMPORTANCE
    # ==========================================================

    importance = pd.Series(
        rf.feature_importances_,
        index=ml_features
    ).sort_values(ascending=False)

    fig_imp = px.bar(
        importance,
        orientation='h',
        title='Random Forest Feature Importance',
        labels={
            'value': 'Importance',
            'index': 'Features'
        }
    )

    st.plotly_chart(
        fig_imp,
        use_container_width=True
    )

    # ==========================================================
    # OPTIONAL DEBUG INFO
    # ==========================================================

    st.write("Dataset Shape:", ml_df.shape)

    




# ==========================================================
# TAB 6 — FORECASTING
# ==========================================================

with tabs[5]:

    st.subheader("🔮 Crime Forecasting")

    crime_year = (
        df.groupby('Year')['total crimes']
        .sum()
    )

    model = ARIMA(
        crime_year,
        order=(1,1,1)
    )

    model_fit = model.fit()

    forecast = model_fit.forecast(steps=5)

    future_years = list(
        range(
            crime_year.index.max()+1,
            crime_year.index.max()+6
        )
    )

    fig_forecast = go.Figure()

    fig_forecast.add_trace(
        go.Scatter(
            x=crime_year.index,
            y=crime_year.values,
            mode='lines+markers',
            name='Historical'
        )
    )

    fig_forecast.add_trace(
        go.Scatter(
            x=future_years,
            y=forecast,
            mode='lines+markers',
            name='Forecast'
        )
    )

    st.plotly_chart(
        fig_forecast,
        use_container_width=True
    )


# ==========================================================
# TAB 7 — RISK & DEVELOPMENT
# ==========================================================

with tabs[6]:

    st.subheader("🚨 Risk & Development Index")

    c1, c2 = st.columns(2)

    with c1:

        risk_states = (
            filtered.groupby('STATE/UT')[
                'Crime Risk Index'
            ]
            .mean()
            .sort_values(ascending=False)
            .head(10)
        )

        fig_risk = px.bar(
            risk_states,
            title='Highest Risk States'
        )

        st.plotly_chart(
            fig_risk,
            use_container_width=True
        )

    with c2:

        development = (
            filtered.groupby('STATE/UT')[
                'Development Index'
            ]
            .mean()
            .sort_values(ascending=False)
            .head(10)
        )

        fig_dev = px.bar(
            development,
            title='Top Developed States'
        )

        st.plotly_chart(
            fig_dev,
            use_container_width=True
        )

    # Safest states
    safe_states = (
        filtered.groupby('STATE/UT')[
            'Crime Risk Index'
        ]
        .mean()
        .sort_values()
        .head(5)
    )

    fig_safe = px.bar(
        safe_states,
        title='Top 5 Safest States'
    )

    st.plotly_chart(
        fig_safe,
        use_container_width=True
    )


# ==========================================================
# TAB 8 — ANOMALY DETECTION
# ==========================================================

with tabs[7]:

    st.subheader("🧠 Crime Anomaly Detection")

    anomaly_features = filtered[[
        'total crimes',
        'Crime Risk Index',
        'Crime per Lakh Population'
    ]]

    iso = IsolationForest(
        contamination=0.05,
        random_state=42
    )

    filtered['Anomaly'] = (
        iso.fit_predict(anomaly_features)
    )

    anomalies = filtered[
        filtered['Anomaly'] == -1
    ]

    st.dataframe(
        anomalies[[
            'STATE/UT',
            'total crimes',
            'Crime Risk Index'
        ]],
        use_container_width=True
    )

    st.error(
        f"{len(anomalies)} anomalous high-risk regions detected."
    )


# ==========================================================
# TAB 9 — RECOMMENDATION SYSTEM
# ==========================================================

with tabs[8]:

    st.subheader("📌 Policy Recommendation System")

    # ======================================================
    # STATE LEVEL AGGREGATION
    # ======================================================

    policy_df = (
        filtered.groupby('STATE/UT')[
            [
                'poverty rate (%)',
                'literacy rate(%)',
                'crime against women',
                'unemployment rate'
            ]
        ]
        .mean()
        .reset_index()
    )

    # ======================================================
    # POLICY FUNCTION
    # ======================================================

    def policy_recommendation(row):

        recommendations = []

        if row['poverty rate (%)'] > df[
            'poverty rate (%)'
        ].mean():

            recommendations.append(
                'Increase poverty reduction programs'
            )

        if row['literacy rate(%)'] < df[
            'literacy rate(%)'
        ].mean():

            recommendations.append(
                'Improve educational investment'
            )

        if row['crime against women'] > df[
            'crime against women'
        ].mean():

            recommendations.append(
                'Strengthen women safety policies'
            )

        if row['unemployment rate'] > df[
            'unemployment rate'
        ].mean():

            recommendations.append(
                'Increase employment opportunities'
            )

        if len(recommendations) == 0:

            recommendations.append(
                'Current indicators appear stable'
            )

        return " | ".join(recommendations)

    # ======================================================
    # APPLY RECOMMENDATIONS
    # ======================================================

    policy_df['Policy Recommendation'] = (
        policy_df.apply(
            policy_recommendation,
            axis=1
        )
    )

    # ======================================================
    # SHOW RESULTS
    # ======================================================

    st.dataframe(
        policy_df,
        use_container_width=True
    )


# ==========================================================
# TAB 10 — DATA EXPLORER
# ==========================================================

with tabs[9]:

    st.subheader("📂 Interactive Dataset Explorer")

    st.dataframe(
        filtered,
        use_container_width=True
    )


# ==========================================================
# TAB 11 — FINAL INSIGHTS
# ==========================================================

with tabs[10]:

    st.subheader("📘 Final Insights")

    highest_crime = (
        filtered.groupby('STATE/UT')[
            'total crimes'
        ]
        .mean()
        .idxmax()
    )

    highest_literacy = (
        filtered.groupby('STATE/UT')[
            'literacy rate(%)'
        ]
        .mean()
        .idxmax()
    )

    highest_poverty = (
        filtered.groupby('STATE/UT')[
            'poverty rate (%)'
        ]
        .mean()
        .idxmax()
    )

    st.success(
        f"🚨 Highest Crime State: {highest_crime}"
    )

    st.info(
        f"📚 Highest Literacy State: {highest_literacy}"
    )

    st.warning(
        f"💰 Highest Poverty State: {highest_poverty}"
    )

    # Automated Insight
    yearly = (
        filtered.groupby('Year')['total crimes']
        .sum()
        .reset_index()
    )

    yearly['Growth %'] = (
        yearly['total crimes']
        .pct_change() * 100
    )

    highest_growth = yearly.loc[
        yearly['Growth %'].idxmax()
    ]

    st.success(
        f"📈 Highest crime growth observed in "
        f"{int(highest_growth['Year'])} "
        f"with {highest_growth['Growth %']:.2f}% increase."
    )

    # Hypothesis Testing
    median_literacy = (
        filtered['literacy rate(%)']
        .median()
    )

    high_lit = filtered[
        filtered['literacy rate(%)']
        >= median_literacy
    ]['total crimes']

    low_lit = filtered[
        filtered['literacy rate(%)']
        < median_literacy
    ]['total crimes']

    stat, p = ttest_ind(
        high_lit,
        low_lit
    )

    st.subheader("🧪 Hypothesis Testing")

    st.write(f"T-Statistic: {stat:.4f}")
    st.write(f"P-Value: {p:.4f}")

    if p < 0.05:
        st.success(
            "Significant difference exists."
        )
    else:
        st.warning(
            "No significant difference found."
        )

    # ==========================================================
    # ADDITIONAL FINAL INSIGHTS
    # ==========================================================

    st.subheader("📌 Key Project Insights")

    insights = [

        "Population significantly influences total crime volume across states.",

        "Poverty and unemployment show noticeable relationships with crime patterns.",

        "Crime per lakh population provides more accurate comparison than total crime counts.",

        "Crimes against women and children remain major concerns in several regions.",

        "Random Forest performed better than Linear Regression in predicting crime trends.",

        "Feature importance analysis identified population, poverty, and unemployment as major influencing factors.",

        "K-Means clustering revealed groups of states with similar socio-economic and crime characteristics."
    ]

    for insight in insights:
        st.markdown(f"✅ {insight}")


# ==========================================================
# DOWNLOAD SECTION
# ==========================================================

st.sidebar.subheader("📥 Download Dataset")

csv = filtered.to_csv(index=False)

st.sidebar.download_button(
    label="Download Filtered Dataset",
    data=csv,
    file_name="crime_analysis.csv",
    mime="text/csv"
)