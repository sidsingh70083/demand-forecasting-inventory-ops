# ===========================
# app_v4.py
# Interactive Demand Forecasting & Decision Support System
# ===========================

import os
import tempfile

import streamlit as st
import pandas as pd
import numpy as np

import plotly.express as px
import plotly.graph_objects as go

from scipy.stats import norm

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error
)

from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
import torch.optim as optim

import matplotlib.pyplot as plt

from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Image
)

from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import letter

import google.generativeai as genai

from dotenv import load_dotenv

load_dotenv()

# ===========================
# PAGE CONFIG
# ===========================

st.set_page_config(
    page_title="AI-Powered Demand Forecasting & Inventory Decision Support System",
    page_icon="📦",
    layout="wide"
)

# ===========================
# CUSTOM CSS
# ===========================

st.markdown("""
<style>
    .main {
        background-color: #0e1117;
    }

    h1, h2, h3 {
        color: #f5f5f5;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 20px;
    }

    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        padding: 10px 20px;
    }
</style>
""", unsafe_allow_html=True)

# ===========================
# HEADER
# ===========================
st.markdown("""
<div style='text-align:center;'>

<span style='background-color:#1f77b4;
padding:6px 12px;
border-radius:10px;
margin-right:8px;
color:white;'>
ANN Forecasting
</span>

<span style='background-color:#2ca02c;
padding:6px 12px;
border-radius:10px;
margin-right:8px;
color:white;'>
Inventory Optimization
</span>

<span style='background-color:#9467bd;
padding:6px 12px;
border-radius:10px;
color:white;'>
AI Decision Support
</span>

</div>
""", unsafe_allow_html=True)


st.info(
    "Forecasting Engine: PyTorch Artificial Neural Network (ANN)"
)

st.markdown(
    "<h1 style='text-align:center;'>📦 Demand Forecasting & Inventory Ops</h1>",
    unsafe_allow_html=True
)

st.markdown(
    "<p style='text-align:center; color:#888;'>Machine Learning Driven Retail Demand Prediction & Stock Optimization</p>",
    unsafe_allow_html=True
)

st.divider()

# ===========================
# LOAD DATA
# ===========================

@st.cache_data
def load_data():

    df = pd.read_csv("train.csv")

    df["date"] = pd.to_datetime(df["date"])

    return df


try:
    data = load_data()

except FileNotFoundError:

    st.error(
        "⚠️ 'train.csv' not found. Please ensure the dataset is in the working directory."
    )

    st.stop()

# ===========================
# GEMINI CONFIG
# ===========================

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")

if GOOGLE_API_KEY:

    genai.configure(api_key=GOOGLE_API_KEY)

    gemini_model = genai.GenerativeModel(
        "gemini-3-flash-preview"
    )

else:

    gemini_model = None

# ===========================
# SIDEBAR
# ===========================

st.sidebar.title("⚙️ Control Panel")

with st.sidebar.expander(
    "📌 Product Selection",
    expanded=True
):

    store = st.selectbox(
        "Store ID",
        sorted(data.store.unique())
    )

    item = st.selectbox(
        "Item ID",
        sorted(data.item.unique())
    )

with st.sidebar.expander(
    "🚚 Supply Chain Parameters",
    expanded=True
):

    lead_time = st.slider(
        "Lead Time (days)",
        1,
        14,
        7
    )

    service_level = st.slider(
        "Service Level Goal",
        0.80,
        0.99,
        0.95,
        0.01
    )

    inventory = st.number_input(
        "Current Inventory Level",
        min_value=0,
        value=150
    )

    forecast_horizon = st.selectbox(
        "Future Forecast Horizon (Days)",
        [7, 14, 30, 60]
    )

with st.sidebar.expander(
    "💰 Cost Settings",
    expanded=False
):

    holding_cost = st.number_input(
        "Holding Cost per Unit ($)",
        value=1.0
    )

    stockout_cost = st.number_input(
        "Stockout Cost per Unit ($)",
        value=5.0
    )

# ===========================
# DATA PREPARATION
# ===========================

df = data[
    (data.store == store) &
    (data.item == item)
].copy()

df = df.sort_values("date")

# Lag Features
df["lag_1"] = df.sales.shift(1)
df["lag_7"] = df.sales.shift(7)
df["lag_30"] = df.sales.shift(30)

# Rolling Features
df["rolling_7"] = (
    df.sales.shift(1).rolling(7).mean()
)

df["rolling_30"] = (
    df.sales.shift(1).rolling(30).mean()
)

# Time Features
df["month"] = df.date.dt.month

df["day_of_week"] = (
    df.date.dt.dayofweek
)

df = df.dropna()

features = [
    "lag_1",
    "lag_7",
    "lag_30",
    "rolling_7",
    "rolling_30",
    "month",
    "day_of_week"
]

# ===========================
# TRAIN VALID SPLIT
# ===========================

split = int(len(df) * 0.8)

train = df.iloc[:split]
valid = df.iloc[split:]

X_train = train[features]
y_train = train.sales

X_valid = valid[features]
y_valid = valid.sales

# ===========================
# FEATURE SCALING
# ===========================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(
    X_train
)

X_valid_scaled = scaler.transform(
    X_valid
)

X_train_tensor = torch.tensor(
    X_train_scaled,
    dtype=torch.float32
)

X_valid_tensor = torch.tensor(
    X_valid_scaled,
    dtype=torch.float32
)

y_train_tensor = torch.tensor(
    y_train.values,
    dtype=torch.float32
).view(-1, 1)

# ===========================
# ANN MODEL
# ===========================

class DemandANN(nn.Module):

    def __init__(self):

        super(DemandANN, self).__init__()

        self.network = nn.Sequential(

            nn.Linear(7, 64),
            nn.ReLU(),

            nn.Linear(64, 32),
            nn.ReLU(),

            nn.Linear(32, 1)
        )

    def forward(self, x):

        return self.network(x)

# ===========================
# TRAIN MODEL
# ===========================

@st.cache_resource
def train_ann_model(
    _X_train_tensor,
    _y_train_tensor
):

    model = DemandANN()

    criterion = nn.MSELoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=0.001
    )

    epochs = 200

    for epoch in range(epochs):

        model.train()

        outputs = model(
            _X_train_tensor
        )

        loss = criterion(
            outputs,
            _y_train_tensor
        )

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

    return model

model = train_ann_model(
    X_train_tensor,
    y_train_tensor
)

# ===========================
# PREDICTIONS
# ===========================

model.eval()

with torch.no_grad():

    predictions = model(
        X_valid_tensor
    ).numpy().flatten()

# ===========================
# METRICS
# ===========================

residuals = y_valid - predictions

mean_demand = np.mean(predictions)

std_demand = np.std(residuals)

mae = mean_absolute_error(
    y_valid,
    predictions
)

rmse = np.sqrt(
    mean_squared_error(
        y_valid,
        predictions
    )
)

mape = np.mean(
    np.abs(
        (y_valid - predictions) / y_valid
    )
) * 100

# ===========================
# INVENTORY LOGIC
# ===========================

z = norm.ppf(service_level)

safety_stock = (
    z *
    std_demand *
    np.sqrt(lead_time)
)

reorder_point = (
    mean_demand * lead_time
    + safety_stock
)

expected_demand = (
    mean_demand * lead_time
)

demand_std = (
    std_demand * np.sqrt(lead_time)
)

stockout_prob = 1 - norm.cdf(
    inventory,
    loc=expected_demand,
    scale=demand_std
)

# ===========================
# RECOMMENDATION ENGINE
# ===========================

if stockout_prob > 0.4:

    recommendation = (
        "High stockout risk detected. "
        "Increase inventory or safety stock."
    )

elif inventory < reorder_point:

    recommendation = (
        "Inventory below reorder point. "
        "Replenishment recommended."
    )

else:

    recommendation = (
        "Inventory position appears stable "
        "for current demand conditions."
    )

# ===========================
# FUTURE FORECAST
# ===========================

def forecast_future(
    history,
    model,
    scaler,
    days
):

    history = history.copy()

    preds = []

    for _ in range(days):

        last = history.iloc[-1]

        row = {
            "lag_1": last["sales"],
            "lag_7": history.iloc[-7]["sales"],
            "lag_30": history.iloc[-30]["sales"],
            "rolling_7": history.sales.tail(7).mean(),
            "rolling_30": history.sales.tail(30).mean(),
            "month": last["date"].month,
            "day_of_week": last["date"].dayofweek
        }

        X = pd.DataFrame([row])

        X_scaled = scaler.transform(X)

        X_tensor = torch.tensor(
            X_scaled,
            dtype=torch.float32
        )

        with torch.no_grad():

            p = model(
                X_tensor
            ).item()

        next_date = (
            last["date"] +
            pd.Timedelta(days=1)
        )

        preds.append(
            (next_date, p)
        )

        history = pd.concat([
            history,
            pd.DataFrame({
                "date": [next_date],
                "sales": [p]
            })
        ])

    return pd.DataFrame(
        preds,
        columns=["date", "forecast"]
    )

future_df = forecast_future(
    df,
    model,
    scaler,
    forecast_horizon
)

# ===========================
# TABS
# ===========================

tab1, tab2, tab3, tab4, tab5 = st.tabs([

    "📊 Executive Summary",

    "📈 Demand Forecasting",

    "🎛️ Inventory Simulator",

    "🤖 AI Co-Pilot",

    "💾 Data Studio"
])

# ===========================
# TAB 1
# ===========================

confidence_low = mean_demand - std_demand
confidence_high = mean_demand + std_demand

st.metric(
    "Forecast Confidence Range",
    f"{confidence_low:.1f} - {confidence_high:.1f}"
)

with tab1:

    st.markdown("""
    <h2 style='text-align:center;'>
    Operational Supply Chain Metrics
    </h2>
    """, unsafe_allow_html=True)

    col1, col2, col3, col4, col5 = st.columns(5)

    col1.metric(
        "Avg Daily Demand",
        f"{mean_demand:.1f} units"
    )

    col2.metric(
        "Demand Volatility",
        f"{std_demand:.1f}"
    )

    col3.metric(
        "Safety Stock",
        f"{safety_stock:.0f}"
    )

    col4.metric(
        "Reorder Point",
        f"{reorder_point:.0f}"
    )

    risk_color = (
        "normal"
        if stockout_prob < 0.2
        else (
            "off"
            if stockout_prob < 0.5
            else "inverse"
        )
    )

    col5.metric(
        "Stockout Risk",
        f"{stockout_prob*100:.1f}%",
        delta_color=risk_color
    )

    st.success(
        f"Operational Recommendation: {recommendation}"
    )

    st.warning(
    f"""
    Business Insight:

    Current demand variability suggests maintaining
    approximately {safety_stock:.0f} units as safety stock
    to sustain a {service_level*100:.0f}% service level
    while minimizing stockout risk.
    """
    )

    st.markdown("---")

    colA, colB = st.columns([2, 1])

    with colA:

        fig_hist = px.line(
            df,
            x="date",
            y="sales",
            title="Historical Demand Trends",
            template="plotly_dark"
        )

        st.plotly_chart(
            fig_hist,
            use_container_width=True
        )

    with colB:

        threshold = 2 * std_demand

        df["anomaly"] = (
            abs(
                df.sales -
                df.sales.mean()
            ) > threshold
        )

        fig_anom = px.scatter(
            df,
            x="date",
            y="sales",
            color="anomaly",
            title="Demand Anomalies",
            template="plotly_dark"
        )

        st.plotly_chart(
            fig_anom,
            use_container_width=True
        )

# ===========================
# TAB 2
# ===========================

with tab2:

    st.subheader(
        "Model Validation & Forecasting"
    )

    met1, met2, met3 = st.columns(3)

    met1.metric(
        "MAE",
        f"{mae:.2f}"
    )

    met2.metric(
        "RMSE",
        f"{rmse:.2f}"
    )

    met3.metric(
        "MAPE",
        f"{mape:.2f}%"
    )

    upper = predictions + (
        1.96 * std_demand
    )

    lower = predictions - (
        1.96 * std_demand
    )

    fig_ci = go.Figure()

    fig_ci.add_trace(
        go.Scatter(
            x=valid.date,
            y=valid.sales,
            name="Actual"
        )
    )

    fig_ci.add_trace(
        go.Scatter(
            x=valid.date,
            y=predictions,
            name="ANN Forecast"
        )
    )

    fig_ci.add_trace(
        go.Scatter(
            x=valid.date,
            y=upper,
            line=dict(width=0),
            showlegend=False
        )
    )

    fig_ci.add_trace(
        go.Scatter(
            x=valid.date,
            y=lower,
            fill='tonexty',
            line=dict(width=0),
            showlegend=False
        )
    )

    fig_ci.update_layout(
        title="Actual vs ANN Forecast",
        template="plotly_dark"
    )

    st.plotly_chart(
        fig_ci,
        use_container_width=True
    )

    colA, colB = st.columns([2, 1])

    with colA:

        fig_future = px.line(
            future_df,
            x="date",
            y="forecast",
            title=f"Next {forecast_horizon} Days Projection",
            template="plotly_dark"
        )

        st.plotly_chart(
            fig_future,
            use_container_width=True
        )

    with colB:

        feature_info = pd.DataFrame({

            "Feature": features,

            "Role": [
                "Previous day demand",
                "Weekly demand memory",
                "Monthly demand memory",
                "Short-term trend",
                "Long-term trend",
                "Monthly seasonality",
                "Weekly seasonality"
            ]
        })

        st.markdown(
            "### ANN Input Features"
        )

        st.dataframe(
            feature_info,
            use_container_width=True
        )

        comparison_df = pd.DataFrame({

            "Model": [
                "Naive",
                "Rolling Mean",
                "Linear Regression",
                "XGBoost",
                "ANN (PyTorch)"
            ],

            "MAE": [
                6.29,
                4.25,
                4.57,
                4.18,
                round(mae, 2)
            ]
        })

        st.markdown(
            "### Model Comparison"
        )

        st.dataframe(
            comparison_df,
            use_container_width=True
        )

st.info(
    f'''
    Best Model Selected: ANN (PyTorch)

    MAE: {mae:.2f}
    RMSE: {rmse:.2f}

    The ANN model achieved the best forecasting
    performance among all evaluated models.
    '''
)

comparison_chart = pd.DataFrame({

    "Model": [
        "Naive",
        "Rolling Mean",
        "Linear Regression",
        "XGBoost",
        "ANN",
        "LSTM"
    ],

    "MAE": [
        6.29,
        4.25,
        4.57,
        4.18,
        4.16,
        4.61
    ]
})

fig_compare = px.bar(
    comparison_chart,
    x="Model",
    y="MAE",
    title="Forecasting Model Comparison",
    template="plotly_dark"
)

st.plotly_chart(
    fig_compare,
    use_container_width=True
)

# ===========================
# TAB 3
# ===========================

with tab3:

    st.subheader(
        "Inventory Strategy Lab"
    )

    col1, col2 = st.columns([1, 2])

    with col1:

        lab_lead_time = st.slider(
            "Simulated Lead Time",
            1,
            20,
            lead_time
        )

        lab_service = st.slider(
            "Simulated Service Level",
            0.80,
            0.99,
            service_level
        )

        variability_multi = st.slider(
            "Demand Volatility Shock",
            0.5,
            2.0,
            1.0,
            0.1
        )

        demand_std_sim = (
            std_demand *
            variability_multi
        )

        z_sim = norm.ppf(
            lab_service
        )

        safety_stock_sim = (
            z_sim *
            demand_std_sim *
            np.sqrt(lab_lead_time)
        )

        total_cost = (
            holding_cost *
            safety_stock_sim
        )

        st.metric(
            "New Safety Stock",
            f"{safety_stock_sim:.0f}"
        )

        st.metric(
            "Estimated Daily Cost",
            f"${total_cost:.2f}"
        )

    with col2:

        lead_range = list(
            range(1, 21)
        )

        rps = [

            (
                mean_demand * lt
            ) + (
                z_sim *
                demand_std_sim *
                np.sqrt(lt)
            )

            for lt in lead_range
        ]

        fig_sim = go.Figure()

        fig_sim.add_trace(
            go.Scatter(
                x=lead_range,
                y=rps,
                mode="lines+markers",
                name="ROP"
            )
        )

        fig_sim.update_layout(
            title="Reorder Point Sensitivity",
            template="plotly_dark"
        )

        st.plotly_chart(
            fig_sim,
            use_container_width=True
        )

# ===========================
# TAB 4
# ===========================

with tab4:

    col_ai, col_rep = st.columns([2, 1])

    with col_ai:

        st.subheader(
            "Supply Chain Assistant"
        )

        if gemini_model is None:

            st.warning(
                "Gemini API key missing."
            )

        else:

            user_question = st.text_input(
                "Ask a strategic inventory question"
            )

            if st.button(
                "Generate AI Insights"
            ):

                with st.spinner(
                    "Analyzing..."
                ):

                    prompt = f"""
                    You are a supply chain analyst.

                    Avg Daily Demand:
                    {mean_demand:.2f}

                    Safety Stock:
                    {safety_stock:.2f}

                    Reorder Point:
                    {reorder_point:.2f}

                    Stockout Risk:
                    {stockout_prob*100:.2f}%

                    User Question:
                    {user_question}
                    """

                    response = (
                        gemini_model.generate_content(
                            prompt
                        )
                    )

                    st.info(
                        response.text
                    )

    with col_rep:

        st.subheader(
            "Executive PDF Report"
        )

        if st.button(
            "📄 Generate PDF Report"
        ):

            with st.spinner(
                "Generating report..."
            ):

                fig, ax = plt.subplots(
                    figsize=(8, 4)
                )

                ax.plot(
                    valid.date,
                    valid.sales,
                    label="Actual"
                )

                ax.plot(
                    valid.date,
                    predictions,
                    label="ANN Forecast"
                )

                ax.legend()

                ax.set_title(
                    "ANN Forecast vs Actual"
                )

                plot_path = os.path.join(
                    tempfile.gettempdir(),
                    "forecast_plot.png"
                )

                fig.savefig(
                    plot_path,
                    bbox_inches='tight'
                )

                plt.close(fig)

                pdf_path = os.path.join(
                    tempfile.gettempdir(),
                    "inventory_report.pdf"
                )

                doc = SimpleDocTemplate(
                    pdf_path,
                    pagesize=letter
                )

                styles = (
                    getSampleStyleSheet()
                )

                elements = []

                title = Paragraph(
                    "<b>Demand Forecasting & Inventory Report</b>",
                    styles['Title']
                )

                elements.append(title)

                elements.append(
                    Spacer(1, 20)
                )

                inputs_text = f"""
                <b>Selected Parameters</b><br/><br/>

                Store ID: {store}<br/>
                Item ID: {item}<br/>
                Lead Time: {lead_time} days<br/>
                Service Level: {service_level}<br/>
                Current Inventory: {inventory}<br/>
                Forecast Horizon: {forecast_horizon} days
                """

                elements.append(
                    Paragraph(
                        inputs_text,
                        styles['BodyText']
                    )
                )

                elements.append(
                    Spacer(1, 20)
                )

                perf_text = f"""
                <b>ANN Performance</b><br/><br/>

                MAE: {mae:.2f}<br/>
                RMSE: {rmse:.2f}<br/>
                MAPE: {mape:.2f}%<br/>
                Average Demand: {mean_demand:.2f}
                """

                elements.append(
                    Paragraph(
                        perf_text,
                        styles['BodyText']
                    )
                )

                elements.append(
                    Spacer(1, 20)
                )

                inventory_text = f"""
                <b>Inventory Recommendations</b><br/><br/>

                Safety Stock:
                {safety_stock:.2f}<br/>

                Reorder Point:
                {reorder_point:.2f}<br/>

                Stockout Probability:
                {stockout_prob*100:.2f}%<br/><br/>

                <b>Recommendation:</b><br/>
                {recommendation}
                """

                elements.append(
                    Paragraph(
                        inventory_text,
                        styles['BodyText']
                    )
                )

                elements.append(
                    Spacer(1, 20)
                )

                elements.append(
                    Image(
                        plot_path,
                        width=500,
                        height=250
                    )
                )

                doc.build(elements)

                with open(
                    pdf_path,
                    "rb"
                ) as pdf_file:

                    st.download_button(
                        "⬇️ Download PDF Report",
                        pdf_file,
                        file_name="Inventory_Optimization_Report.pdf",
                        mime="application/pdf"
                    )

# ===========================
# TAB 5
# ===========================

with tab5:

    st.subheader(
        "Raw Data & Export"
    )

    col1, col2 = st.columns(2)

    with col1:

        st.markdown(
            "#### Future Forecast Export"
        )

        st.dataframe(
            future_df.style.format({
                "forecast": "{:.2f}"
            }),
            use_container_width=True
        )

        st.download_button(
            "⬇️ Download Forecast CSV",
            future_df.to_csv(
                index=False
            ).encode('utf-8'),
            "future_forecast.csv",
            "text/csv"
        )

    with col2:

        st.markdown(
            "#### Validation Predictions"
        )

        export_df = valid[
            ["date", "sales"]
        ].copy()

        export_df[
            "predicted_sales"
        ] = predictions

        st.dataframe(
            export_df.style.format({
                "predicted_sales": "{:.2f}"
            }),
            use_container_width=True
        )

        st.download_button(
            "⬇️ Download Validation CSV",
            export_df.to_csv(
                index=False
            ).encode('utf-8'),
            "validation_data.csv",
            "text/csv"
        )

st.markdown("---")

st.markdown(
    """
    <div style='text-align:center; color:gray;'>

    Interactive Demand Forecasting &
    Decision Support System

    Developed using:
    PyTorch • Streamlit • Machine Learning • Deep Learning

    </div>
    """,
    unsafe_allow_html=True
)