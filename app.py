"""
================================================================================
 End-to-End Sales Forecasting & Demand Intelligence — Streamlit Dashboard
================================================================================
A 4-page interactive dashboard built for the Superstore sales dataset:

    Page 1 — Sales Overview Dashboard
    Page 2 — Forecast Explorer (SARIMA, by Category or Region)
    Page 3 — Anomaly Report (Isolation Forest + Z-Score)
    Page 4 — Product Demand Segments (K-Means + PCA)

Run locally with:
    streamlit run app.py

Deploy for free on Streamlit Community Cloud by pointing it at this repo/file.
================================================================================
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.statespace.sarimax import SARIMAX

# ------------------------------------------------------------------------------
# Page configuration
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="Sales Forecasting & Demand Intelligence",
    page_icon="📊",
    layout="wide",
)

# ------------------------------------------------------------------------------
# Data loading & caching
# ------------------------------------------------------------------------------
@st.cache_data(show_spinner="Loading and preparing data...")
def load_data():
    df = pd.read_csv("train.csv")
    df["Order Date"] = pd.to_datetime(df["Order Date"])
    df["Ship Date"] = pd.to_datetime(df["Ship Date"])
    df["Order Year"] = df["Order Date"].dt.year
    df["Order Month"] = df["Order Date"].dt.month
    df["Order Quarter"] = df["Order Date"].dt.quarter

    def season(m):
        if m in (12, 1, 2):
            return "Winter"
        if m in (3, 4, 5):
            return "Spring"
        if m in (6, 7, 8):
            return "Summer"
        return "Fall"

    df["Season"] = df["Order Month"].apply(season)
    df["Shipping Delay (days)"] = (df["Ship Date"] - df["Order Date"]).dt.days
    return df


@st.cache_data(show_spinner=False)
def get_monthly_series(_df, category=None, region=None):
    d = _df
    if category and category != "All":
        d = d[d["Category"] == category]
    if region and region != "All":
        d = d[d["Region"] == region]
    monthly = d.set_index("Order Date").resample("MS")["Sales"].sum().asfreq("MS").fillna(0)
    return monthly


@st.cache_data(show_spinner="Fitting SARIMA model...")
def run_sarima_forecast(series, horizon):
    n_test = min(3, max(1, len(series) // 6))
    train, test = series.iloc[:-n_test], series.iloc[-n_test:]
    model = SARIMAX(train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12),
                     enforce_stationarity=False, enforce_invertibility=False)
    fit = model.fit(disp=False)

    # Metrics on the last known holdout window
    test_pred = fit.get_forecast(steps=len(test)).predicted_mean
    mae = mean_absolute_error(test, test_pred)
    rmse = mean_squared_error(test, test_pred) ** 0.5

    # Refit on full history, then forecast the requested horizon into the future
    full_model = SARIMAX(series, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12),
                          enforce_stationarity=False, enforce_invertibility=False)
    full_fit = full_model.fit(disp=False)
    fc_obj = full_fit.get_forecast(steps=horizon)
    forecast = fc_obj.predicted_mean
    ci = fc_obj.conf_int(alpha=0.05)
    return forecast, ci, mae, rmse


# ------------------------------------------------------------------------------
# NEW: per-step RMSE / MSE curve over the holdout window (Page 2 addition)
# ------------------------------------------------------------------------------
@st.cache_data(show_spinner="Computing RMSE/MSE curve...")
def compute_error_curve(series):
    n_test = min(3, max(1, len(series) // 6))
    train, test = series.iloc[:-n_test], series.iloc[-n_test:]
    model = SARIMAX(train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12),
                     enforce_stationarity=False, enforce_invertibility=False)
    fit = model.fit(disp=False)
    test_pred = fit.get_forecast(steps=len(test)).predicted_mean

    steps, mse_curve, rmse_curve = [], [], []
    for i in range(1, len(test) + 1):
        mse_i = mean_squared_error(test.iloc[:i], test_pred.iloc[:i])
        steps.append(i)
        mse_curve.append(mse_i)
        rmse_curve.append(mse_i ** 0.5)

    return pd.DataFrame({"Step": steps, "MSE": mse_curve, "RMSE": rmse_curve})


@st.cache_data(show_spinner="Detecting anomalies...")
def detect_anomalies(_df):
    weekly = _df.set_index("Order Date").resample("W")["Sales"].sum().reset_index()
    weekly.columns = ["week", "sales"]

    iso = IsolationForest(contamination=0.07, random_state=42)
    weekly["iso_flag"] = iso.fit_predict(weekly[["sales"]]) == -1

    window = 8
    weekly["rolling_mean"] = weekly["sales"].rolling(window, min_periods=4, center=True).mean()
    weekly["rolling_std"] = weekly["sales"].rolling(window, min_periods=4, center=True).std()
    weekly["zscore"] = (weekly["sales"] - weekly["rolling_mean"]) / weekly["rolling_std"]
    weekly["z_flag"] = weekly["zscore"].abs() > 2
    return weekly


@st.cache_data(show_spinner="Segmenting products...")
def cluster_products(_df):
    rows = []
    for name, g in _df.groupby("Sub-Category"):
        g_monthly = g.set_index("Order Date").resample("MS")["Sales"].sum()
        total_sales = g_monthly.sum()
        yearly = g.groupby("Order Year")["Sales"].sum()
        yoy_growth = yearly.pct_change().mean() * 100 if len(yearly) > 1 else 0
        volatility = g_monthly.std()
        avg_order_val = g["Sales"].sum() / g["Order ID"].nunique()
        rows.append({
            "Sub-Category": name, "Total Sales": total_sales,
            "YoY Growth %": yoy_growth, "Volatility": volatility,
            "Avg Order Value": avg_order_val
        })
    seg = pd.DataFrame(rows)
    feature_cols = ["Total Sales", "YoY Growth %", "Volatility", "Avg Order Value"]
    X = StandardScaler().fit_transform(seg[feature_cols].fillna(0))

    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    seg["Cluster"] = kmeans.fit_predict(X)

    pca = PCA(n_components=2)
    coords = pca.fit_transform(X)
    seg["PCA1"], seg["PCA2"] = coords[:, 0], coords[:, 1]

    profile = seg.groupby("Cluster")[feature_cols].mean()
    label_map = {}
    for c in profile.index:
        row = profile.loc[c]
        if row["Total Sales"] > profile["Total Sales"].median() and row["Volatility"] < profile["Volatility"].median():
            label_map[c] = "High Volume, Stable Demand"
        elif row["YoY Growth %"] > 50:
            label_map[c] = "Growing Demand (High Volatility)"
        elif row["Total Sales"] < profile["Total Sales"].median() and row["Volatility"] < profile["Volatility"].median():
            label_map[c] = "Low Volume, Stable Demand"
        else:
            label_map[c] = "High Value, Low Volume"
    seg["Cluster Label"] = seg["Cluster"].map(label_map)
    return seg


STRATEGY = {
    "High Volume, Stable Demand": "Maintain steady base stock with simple reorder-point rules; safe to negotiate bulk-purchase discounts.",
    "High Value, Low Volume": "Use just-in-time / made-to-order procurement rather than holding expensive inventory on shelves.",
    "Growing Demand (High Volatility)": "Track closely month-to-month; keep a modest safety-stock buffer and re-evaluate quarterly.",
    "Low Volume, Stable Demand": "Minimize inventory holding — order in small, infrequent batches to save warehouse space.",
}

df = load_data()

# ------------------------------------------------------------------------------
# Sidebar navigation
# ------------------------------------------------------------------------------
st.sidebar.title("📦 Sales Intelligence")
page = st.sidebar.radio(
    "Navigate to:",
    ["Sales Overview", "Forecast Explorer", "Anomaly Report", "Product Demand Segments"],
)
st.sidebar.markdown("---")
st.sidebar.caption(
    "Built for the End-to-End Sales Forecasting & Demand Intelligence System project. "
    f"Data spans {df['Order Date'].min().date()} to {df['Order Date'].max().date()}."
)

# ==============================================================================
# PAGE 1 — Sales Overview Dashboard
# ==============================================================================
if page == "Sales Overview":
    st.title("📊 Sales Overview Dashboard")
    st.caption("A Monday-morning snapshot of company performance for a business manager.")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Revenue", f"${df['Sales'].sum():,.0f}")
    col2.metric("Total Orders", f"{df['Order ID'].nunique():,}")
    col3.metric("Total Profit", f"${df['Profit'].sum():,.0f}")
    col4.metric("Avg Order Value", f"${df['Sales'].sum()/df['Order ID'].nunique():,.2f}")

    st.markdown("### Total Sales by Year")
    yearly = df.groupby("Order Year")["Sales"].sum().reset_index()
    fig_year = px.bar(yearly, x="Order Year", y="Sales", text_auto=".2s",
                       color="Sales", color_continuous_scale="Blues")
    fig_year.update_layout(showlegend=False, coloraxis_showscale=False)
    st.plotly_chart(fig_year, use_container_width=True)

    st.markdown("### Monthly Sales Trend")
    monthly_all = df.set_index("Order Date").resample("MS")["Sales"].sum().reset_index()
    fig_trend = px.line(monthly_all, x="Order Date", y="Sales", markers=True)
    st.plotly_chart(fig_trend, use_container_width=True)

    st.markdown("### Sales by Region & Category")
    c1, c2 = st.columns(2)
    region_filter = c1.multiselect("Filter Region(s)", df["Region"].unique().tolist(),
                                    default=df["Region"].unique().tolist())
    category_filter = c2.multiselect("Filter Category(ies)", df["Category"].unique().tolist(),
                                      default=df["Category"].unique().tolist())
    filtered = df[df["Region"].isin(region_filter) & df["Category"].isin(category_filter)]

    c3, c4 = st.columns(2)
    with c3:
        reg_cat = filtered.groupby(["Region", "Category"])["Sales"].sum().reset_index()
        fig_reg = px.bar(reg_cat, x="Region", y="Sales", color="Category", barmode="group")
        st.plotly_chart(fig_reg, use_container_width=True)
    with c4:
        cat_totals = filtered.groupby("Category")["Sales"].sum().reset_index()
        fig_pie = px.pie(cat_totals, names="Category", values="Sales", hole=0.45)
        st.plotly_chart(fig_pie, use_container_width=True)

# ==============================================================================
# PAGE 2 — Forecast Explorer
# ==============================================================================
elif page == "Forecast Explorer":
    st.title("🔮 Forecast Explorer")
    st.caption("SARIMA forecasts — the best-performing model identified in the notebook analysis.")

    col1, col2, col3 = st.columns(3)
    dim = col1.selectbox("Forecast by:", ["Category", "Region", "Company-Wide"])

    if dim == "Category":
        options = ["All"] + sorted(df["Category"].unique().tolist())
        selection = col2.selectbox("Select Category", options)
        series = get_monthly_series(df, category=selection)
    elif dim == "Region":
        options = ["All"] + sorted(df["Region"].unique().tolist())
        selection = col2.selectbox("Select Region", options)
        series = get_monthly_series(df, region=selection)
    else:
        selection = "Company-Wide"
        series = get_monthly_series(df)

    horizon = col3.slider("Forecast Horizon (months ahead)", 1, 3, 3)

    forecast, ci, mae, rmse = run_sarima_forecast(series, horizon)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=series.index, y=series.values, name="Historical Sales",
                              line=dict(color="steelblue")))
    fig.add_trace(go.Scatter(x=forecast.index, y=forecast.values, name="Forecast",
                              line=dict(color="crimson", dash="dash"), mode="lines+markers"))
    fig.add_trace(go.Scatter(x=list(forecast.index) + list(forecast.index[::-1]),
                              y=list(ci.iloc[:, 1]) + list(ci.iloc[:, 0][::-1]),
                              fill="toself", fillcolor="rgba(220,20,60,0.15)",
                              line=dict(color="rgba(255,255,255,0)"), name="95% CI"))
    fig.update_layout(title=f"Sales Forecast — {selection} ({dim})", yaxis_title="Sales ($)")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Forecast Values")
    fc_table = pd.DataFrame({
        "Month": forecast.index.strftime("%b %Y"),
        "Forecast ($)": forecast.values.round(0),
        "Lower 95% CI": ci.iloc[:, 0].values.round(0),
        "Upper 95% CI": ci.iloc[:, 1].values.round(0),
    })
    st.dataframe(fc_table, use_container_width=True, hide_index=True)

    st.markdown("### Model Accuracy (evaluated on the most recent holdout window)")
    m1, m2 = st.columns(2)
    m1.metric("MAE", f"${mae:,.0f}")
    m2.metric("RMSE", f"${rmse:,.0f}")

    # -- NEW: RMSE / MSE curve over the holdout steps --------------------------
    st.markdown("### RMSE & MSE Curve (cumulative error across holdout steps)")
    curve_df = compute_error_curve(series)

    fig_err = go.Figure()
    fig_err.add_trace(go.Scatter(
        x=curve_df["Step"], y=curve_df["RMSE"], name="RMSE",
        mode="lines+markers", line=dict(color="crimson")
    ))
    fig_err.add_trace(go.Scatter(
        x=curve_df["Step"], y=curve_df["MSE"], name="MSE",
        mode="lines+markers", line=dict(color="steelblue"), yaxis="y2"
    ))
    fig_err.update_layout(
        title="Cumulative RMSE & MSE over Holdout Steps",
        xaxis_title="Holdout Step",
        yaxis=dict(title="RMSE ($)"),
        yaxis2=dict(title="MSE", overlaying="y", side="right"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig_err, use_container_width=True)
    st.caption(
        "Each point shows RMSE/MSE computed cumulatively over the first N steps "
        "of the holdout window — useful for seeing whether error grows as the "
        "forecast horizon extends further into the holdout period."
    )

# ==============================================================================
# PAGE 3 — Anomaly Report
# ==============================================================================
elif page == "Anomaly Report":
    st.title("🚨 Anomaly Report")
    st.caption("Weeks where sales were unusually high or low, detected two independent ways.")

    weekly = detect_anomalies(df)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=weekly["week"], y=weekly["sales"], name="Weekly Sales",
                              line=dict(color="steelblue")))
    iso_pts = weekly[weekly["iso_flag"]]
    z_pts = weekly[weekly["z_flag"]]
    fig.add_trace(go.Scatter(x=iso_pts["week"], y=iso_pts["sales"], mode="markers",
                              marker=dict(color="red", size=11, symbol="x"),
                              name="Isolation Forest Anomaly"))
    fig.add_trace(go.Scatter(x=z_pts["week"], y=z_pts["sales"], mode="markers",
                              marker=dict(color="orange", size=15, symbol="circle-open", line=dict(width=2)),
                              name="Z-Score Anomaly"))
    fig.update_layout(title="Weekly Sales with Detected Anomalies", yaxis_title="Sales ($)")
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    col1.metric("Isolation Forest Anomalies", int(weekly["iso_flag"].sum()))
    col2.metric("Z-Score Anomalies", int(weekly["z_flag"].sum()))

    st.markdown("### Detected Anomaly Weeks")
    method = st.radio("Show anomalies from:", ["Isolation Forest", "Z-Score", "Both (agreement only)"], horizontal=True)
    if method == "Isolation Forest":
        table = weekly[weekly["iso_flag"]]
    elif method == "Z-Score":
        table = weekly[weekly["z_flag"]]
    else:
        table = weekly[weekly["iso_flag"] & weekly["z_flag"]]

    table = table[["week", "sales"]].sort_values("sales", ascending=False).copy()
    table["week"] = table["week"].dt.date
    table.columns = ["Week", "Sales ($)"]
    st.dataframe(table, use_container_width=True, hide_index=True)

# ==============================================================================
# PAGE 4 — Product Demand Segments
# ==============================================================================
elif page == "Product Demand Segments":
    st.title("🧩 Product Demand Segments")
    st.caption("K-Means clustering of product sub-categories by demand behavior.")

    seg = cluster_products(df)

    fig = px.scatter(seg, x="PCA1", y="PCA2", color="Cluster Label", text="Sub-Category",
                      size="Total Sales", size_max=40, hover_data=["YoY Growth %", "Volatility", "Avg Order Value"])
    fig.update_traces(textposition="top center")
    fig.update_layout(title="Demand Segmentation (PCA-reduced view)")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Sub-Categories by Demand Cluster")
    display_cols = ["Sub-Category", "Cluster Label", "Total Sales", "YoY Growth %", "Volatility", "Avg Order Value"]
    st.dataframe(seg[display_cols].sort_values("Cluster Label").round(1), use_container_width=True, hide_index=True)

    st.markdown("### Recommended Stocking Strategy per Cluster")
    for label, strat in STRATEGY.items():
        with st.expander(f"📦 {label}"):
            members = seg[seg["Cluster Label"] == label]["Sub-Category"].tolist()
            st.write(f"**Products:** {', '.join(members)}")
            st.write(f"**Strategy:** {strat}")