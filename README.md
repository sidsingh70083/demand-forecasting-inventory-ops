<div align="center">

# 📦 Demand Forecasting & Inventory Ops

**Know what will sell, when to reorder, and how much stock you actually need.**

An end-to-end machine learning system that forecasts retail demand and turns those forecasts into inventory decisions, wrapped in an interactive Streamlit app with AI-written recommendations.

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-ANN_·_LSTM-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-Gradient_Boosting-189FDD?style=for-the-badge)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Gemini](https://img.shields.io/badge/Gemini-AI_Recommendations-8E75B2?style=for-the-badge&logo=googlegemini&logoColor=white)

</div>

---

## The problem

Retailers lose money in two directions: too much stock ties up cash and storage, too little means empty shelves and lost sales. Most small teams still plan with spreadsheets and gut feel. Forecasting alone is not enough either. A manager needs to know **when to reorder, how much safety stock to hold, and how risky the current position is.**

## The result

| Model | MAE | RMSE |
| --- | ---: | ---: |
| Naive forecast (baseline) | 6.29 | 7.72 |
| Rolling mean | 4.25 | 5.35 |
| Linear regression | 4.57 | 5.72 |
| XGBoost | 4.18 | 5.25 |
| **Feed-forward neural network (ANN)** | **4.17** | **5.14** |
| LSTM | 4.62 | 5.86 |

The ANN cut forecast error by **34% against the naive baseline** and was chosen for deployment. One honest takeaway: the "more advanced" LSTM did worse. Good lag and rolling features captured most of the signal, so a simpler network generalised better.

## What the app does

- **Executive summary:** sales KPIs and demand trends at a glance
- **Demand forecasting:** future demand per store and item, with forecast vs actual charts
- **Inventory simulation:** safety stock, reorder point and stockout probability for any service level and lead time you choose
- **AI recommendations:** Gemini turns the numbers into plain-language next steps for an inventory manager
- **PDF reports:** one-click report generation for the chosen parameters

## Data

[Store Item Demand Forecasting Challenge](https://www.kaggle.com/competitions/demand-forecasting-kernels-only) on Kaggle: **913,000 daily sales records** across 10 stores and 50 items (2013–2017).

## How it was built

```
notebooks/
  01–02   problem framing and exploratory analysis (trends, seasonality, weekly patterns)
  03–08   baselines, linear regression and XGBoost with lag, rolling and calendar features
  09–14   model comparison, panel models, rolling cross-validation, feature importance, residuals
  15–17   inventory decision logic, simulation and the end-to-end pipeline
  18–20   deep learning: feed-forward ANN and LSTM in PyTorch, final comparison
app.py    the Streamlit decision-support app
```

## Run it locally

```bash
git clone https://github.com/sidsingh70083/demand-forecasting-inventory-ops.git
cd demand-forecasting-inventory-ops
pip install -r requirements.txt
```

1. Download `train.csv` from the Kaggle link above into the project folder.
2. Optional, for AI recommendations: copy `.env.example` to `.env` and add your Gemini API key.
3. Start the app:

```bash
streamlit run app.py
```

## Tech stack

Python · pandas · NumPy · scikit-learn · XGBoost · PyTorch · SciPy · Plotly · Matplotlib · Streamlit · Google Gemini · ReportLab

---

<div align="center">

Built by **[Siddharth Singh](https://www.linkedin.com/in/siddharth2025)** · Final-year project, Manipal Institute of Technology

</div>
