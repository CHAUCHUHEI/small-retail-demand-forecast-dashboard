import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from datetime import datetime

# --- page setup 
st.set_page_config(page_title="Retail Forecast Tool", layout="wide")
st.title("Retail Demand & Inventory Dashboard")

# session flags 
if "uploaded_once" not in st.session_state:
    st.session_state["uploaded_once"] = False
if "applied_category" not in st.session_state:
    st.session_state["applied_category"] = "All"
if "run_forecast" not in st.session_state:
    st.session_state["run_forecast"] = False
if "forecast_settings" not in st.session_state:
    st.session_state["forecast_settings"] = {}
if "df_future" not in st.session_state:
    st.session_state["df_future"] = None

# show upload notes only before first upload
if not st.session_state["uploaded_once"]:
    st.markdown("""
        ### Before uploading your CSV

        Please ensure your file contains at least these columns:
        - Date (YYYY-MM-DD)
        - Product ID
        - Units Sold
        - Price

        Adding these optional columns will make the forecasts more accurate:
        - Category
        - Inventory Level (without it, suggestions show as N/A)
        - Weather / Promo / Seasonality
        - Units Ordered
        - Discount

        Each row should represent the daily sales of one product.
    """)

# --- file upload
file = st.file_uploader("Upload your sales CSV", type=["csv"])

if file:
    st.session_state["uploaded_once"] = True

    # read data
    df = pd.read_csv(file)
    
    # for Business insight block
    df_original = df.copy()
    if "Date" in df_original.columns:
        df_original["Date"] = pd.to_datetime(df_original["Date"], errors="coerce")
        df_original["DayOfWeek"] = df_original["Date"].dt.dayofweek
        df_original["Month"] = df_original["Date"].dt.month

    # required + optional columns
    req_cols = ["Date", "Product ID", "Units Sold", "Price"]
    opt_cols = ["Category", "Inventory Level", "Weather Condition",
                "Holiday/Promotion", "Seasonality", "Units Ordered", "Discount"]

    missing_req = [c for c in req_cols if c not in df.columns]
    if missing_req:
        st.error(f"Missing required columns: {missing_req}. Fix and re-upload.")
        st.stop()

    has_cat = "Category" in df.columns
    has_inv = "Inventory Level" in df.columns

    # fill optional columns with defaults
    for c in opt_cols:
        if c not in df.columns:
            if c == "Holiday/Promotion":
                df[c] = 0
            elif c in ["Units Ordered", "Discount"]:
                df[c] = 0.0
            elif c == "Inventory Level":
                df[c] = np.nan
            elif c == "Category":
                df[c] = "Unknown"
            else:
                df[c] = "Unknown"

    if not has_inv:
        st.info("No 'Inventory Level' column found. Reorder suggestions will appear as N/A.")
    if not has_cat:
        st.info("No 'Category' column found. Category views will be unavailable.")

    # --- basic date features
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["DayOfWeek"] = df["Date"].dt.dayofweek
    df["Month"] = df["Date"].dt.month

    # copy for modelling (sorted)
    df_model = df.copy().sort_values(["Product ID", "Date"])

    # -- filters
    st.subheader("Filters")

    cat_opts = ["All"] + list(df["Category"].unique()) if has_cat else ["All"]
    cat_pick = st.selectbox("Category filter", cat_opts)

    if not has_cat:
        st.info("Category filter limited because original data has no Category column.")

    if st.button("Apply filters"):
        st.session_state["applied_category"] = cat_pick
        st.session_state["run_forecast"] = False
        st.session_state["df_future"] = None
        st.success("Filters applied. Adjust settings and run forecast.")

    # filtering function 
    def filter_by_cat(df_in):
        if not has_cat:
            return df_in.copy()
        if st.session_state["applied_category"] == "All":
            return df_in.copy()
        return df_in[df_in["Category"] == st.session_state["applied_category"]].copy()

    df_filt = filter_by_cat(df)
    if df_filt.empty:
        st.warning("No data left after filtering.")
        st.stop()

    # -- forecast settings 
    st.subheader("Forecast settings")

    col_a, col_b, col_c = st.columns(3)
    risk_mode = col_a.selectbox("Risk level", ["Conservative", "Neutral", "Aggressive"])
    cash_cycle = col_b.number_input("Cash flow cycle (days)", 7, 120, 30, step=5)
    lead_time = col_c.slider("Lead time (days)", 1, 30, 14)
    h = st.slider("Forecast horizon (days)", 7, 60, 14)

    # service factor calc 
    base_z = {"Conservative": 2.05, "Neutral": 1.65, "Aggressive": 1.04}[risk_mode]
    cash_adj = max(0.7, min(1.8, 1 + (cash_cycle - 30) / 60))
    svc = max(0.8, min(3.0, round(base_z * cash_adj, 2)))

    st.markdown(f"""
        **Current settings**
        - Risk: {risk_mode}
        - Lead time: {lead_time} days
        - Horizon: {h} days
        - Cash cycle: {cash_cycle} days
        - Service factor: {svc}
    """)

    run_msg = st.empty()

    # run + reset buttons
    btn_left, btn_right = st.columns([1, 3])
    if btn_left.button("Run Forecast"):
        st.session_state["run_forecast"] = True
        st.session_state["forecast_settings"] = {
            "risk_type": risk_mode,
            "cash_cycle_days": cash_cycle,
            "lead_time": lead_time,
            "forecast_horizon": h,
            "service_factor": svc,
        }
        st.session_state["df_future"] = None
        run_msg.info("Running forecast...")

    if btn_right.button("Reset run flag"):
        st.session_state["run_forecast"] = False
        st.session_state["df_future"] = None
        st.info("Run flag reset. Adjust settings and run again.")

    # ---My Store Business Insights
    with st.expander("🏪 My Store Business Insights", expanded=False):

        chart_option = st.selectbox(
            "Choose Business insight",
            [
                "Weekly Sales Pattern",
                "Weather Impact",
                "Promotion / Holiday Impact",
                "Seasonality Trend"
            ]
        )

        df_fp = df_original.copy()

        # ensure required columns exist 
        if "Date" in df_fp.columns:
            df_fp["Date"] = pd.to_datetime(df_fp["Date"], errors="coerce")
            if "DayOfWeek" not in df_fp.columns:
                df_fp["DayOfWeek"] = df_fp["Date"].dt.dayofweek
            if "Month" not in df_fp.columns:
                df_fp["Month"] = df_fp["Date"].dt.month
        else:
            if "DayOfWeek" not in df_fp.columns:
                df_fp["DayOfWeek"] = 0
            if "Month" not in df_fp.columns:
                df_fp["Month"] = 0

        if "Weather Condition" not in df_fp.columns:
            df_fp["Weather Condition"] = "Unknown"
        if "Holiday/Promotion" not in df_fp.columns:
            df_fp["Holiday/Promotion"] = 0
        if "Seasonality" not in df_fp.columns:
            df_fp["Seasonality"] = "Unknown"

        # 1. Weekly Sales Pattern 
        if chart_option == "Weekly Sales Pattern":
            st.subheader("Weekly Sales Rhythm")

            if "DayOfWeek" in df_fp.columns and "Units Sold" in df_fp.columns:
                day_sales = df_fp.groupby('DayOfWeek')['Units Sold'].mean()
                if not day_sales.empty:
                    day_labels = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']

                    plt.figure(figsize=(7,3))
                    sns.lineplot(
                        x=day_labels[:len(day_sales)],
                        y=day_sales.values,
                        marker='o',
                        linewidth=2,
                        color=sns.color_palette("viridis", 7)[4]
                    )
                    plt.ylabel("Avg Units Sold")
                    plt.xlabel("")
                    st.pyplot(plt)

                    st.markdown(
                        f"- **Peak Day:** **{day_labels[int(day_sales.idxmax())]}** with **{day_sales.max():.0f} units**  \n"
                        f"- **Slowest Day:** **{day_labels[int(day_sales.idxmin())]}** with **{day_sales.min():.0f} units**  \n"
                        f"- **Weekly Avg:** **{day_sales.mean():.0f} units/day**"
                    )
                else:
                    st.info("Not enough data to compute weekly pattern.")
            else:
                st.info("Required columns for weekly pattern are missing.")

        # 2. Weather Impact
        elif chart_option == "Weather Impact":
            st.subheader("Weather vs Sales Performance")

            # check if weather data actually exists
            weather_available = (
                "Weather Condition" in df_fp.columns 
                and df_fp["Weather Condition"].notna().sum() > 0
                and df_fp["Weather Condition"].nunique() > 1
            )

            if not weather_available:
                st.info("Weather data not found in your CSV — cannot compute weather impact.")
            else:
                weather_sales = (
                    df_fp.groupby('Weather Condition')['Units Sold']
                    .mean()
                    .sort_values()
                )

                if not weather_sales.empty:
                    plt.figure(figsize=(7,3))
                    sns.barplot(
                        x=weather_sales.values,
                        y=weather_sales.index,
                        palette="coolwarm"
                    )
                    plt.xlabel("Avg Units Sold")
                    plt.ylabel("")
                    st.pyplot(plt)

                    st.markdown(
                        f"- **Best Weather:** **{weather_sales.idxmax()}** → **{weather_sales.max():.0f} units**  \n"
                        f"- **Worst Weather:** **{weather_sales.idxmin()}** → **{weather_sales.min():.0f} units**  \n"
                        f"- **Weather Gap:** **{(weather_sales.max()-weather_sales.min()):.0f} units**"
                    )
                else:
                    st.info("Not enough data to compute weather impact.")


        # 3. Promotion Impact
        elif chart_option == "Promotion / Holiday Impact":
            st.subheader("Promotion & Holiday Lift")

            # check if promo data actually exists
            promo_available = (
                "Holiday/Promotion" in df_fp.columns
                and df_fp["Holiday/Promotion"].notna().sum() > 0
                and df_fp["Holiday/Promotion"].nunique() > 1
                and "Units Sold" in df_fp.columns
            )

            if not promo_available:
                st.info("Promotion data not found in your csv — cannot compute promotion impact.")
            else:
                promo_sales = (
                    df_fp.groupby('Holiday/Promotion')['Units Sold']
                    .mean()
                    .sort_values()
                )

                if not promo_sales.empty:
                    # rename index safely
                    if 0 in promo_sales.index and 1 in promo_sales.index:
                        promo_sales.index = ['No Promo', 'Promo']
                    else:
                        promo_sales.index = [str(i) for i in promo_sales.index]

                    plt.figure(figsize=(6,3))
                    sns.barplot(
                        x=promo_sales.index,
                        y=promo_sales.values,
                        palette="viridis"
                    )
                    plt.ylabel("Avg Units Sold")
                    plt.xlabel("")
                    st.pyplot(plt)

                    # compute lift only if both groups exist
                    if 'Promo' in promo_sales.index and 'No Promo' in promo_sales.index:
                        lift = promo_sales['Promo'] - promo_sales['No Promo']
                        lift_pct = lift / promo_sales['No Promo'] * 100 if promo_sales['No Promo'] != 0 else np.nan

                        st.markdown(
                            f"- **Promotion Lift:** **+{lift:.0f} units**  \n"
                            f"- **Lift Percentage:** **+{lift_pct:.1f}%**  \n"
                            f"- **Business Signal:** Promotions materially change demand"
                        )
                    else:
                        st.info("Not enough promo/no-promo variation to compute lift.")
                else:
                    st.info("Not enough data to compute promotion impact.")


        # 4. Seasonality 
        elif chart_option == "Seasonality Trend":
            st.subheader("Seasonality Demand Cycle")

            # check if seasonality data actually exists
            seasonality_available = (
                "Month" in df_fp.columns
                and df_fp["Month"].notna().sum() > 0
                and df_fp["Month"].nunique() > 1
                and "Units Sold" in df_fp.columns
            )

            if not seasonality_available:
                st.info("Seasonality data not found or insufficient — cannot compute seasonality trend.")
            else:
                month_sales = (
                    df_fp.groupby('Month')['Units Sold']
                    .mean()
                    .sort_values()
                )

                if not month_sales.empty:
                    plt.figure(figsize=(7,3))
                    sns.lineplot(
                        x=month_sales.index,
                        y=month_sales.values,
                        marker='o',
                        linewidth=2,
                        color=sns.color_palette("magma", 1)[0]
                    )
                    plt.xlabel("Month")
                    plt.ylabel("Avg Units Sold")
                    st.pyplot(plt)

                    st.markdown(
                        f"- **Peak Month:** **{int(month_sales.idxmax())}** → **{month_sales.max():.0f} units**  \n"
                        f"- **Low Month:** **{int(month_sales.idxmin())}** → **{month_sales.min():.0f} units**  \n"
                        f"- **Seasonal Spread:** **{(month_sales.max()-month_sales.min()):.0f} units**"
                    )
                else:
                    st.info("Not enough data to compute seasonality.")


    # if nothing to reuse and not running → stop
    if st.session_state["df_future"] is None and not st.session_state["run_forecast"]:
        st.info("Adjust settings and click Run Forecast.")
        st.stop()

    # --- reuse existing forecast if available 
    if st.session_state["df_future"] is not None and not st.session_state["run_forecast"]:
        future_df = st.session_state["df_future"]
        df_eval = st.session_state.get("df_eval")
        mae = st.session_state.get("mae")
        r2 = st.session_state.get("r2")

        fs = st.session_state["forecast_settings"]
        risk_mode = fs["risk_type"]
        cash_cycle = fs["cash_cycle_days"]
        lead_time = fs["lead_time"]
        h = fs["forecast_horizon"]
        svc = fs["service_factor"]

    else:
        # --- model training 
        with st.spinner("Training model..."):
            # lag features per product
            for lag in [1, 7]:
                df_model[f"Sales_Lag_{lag}"] = df_model.groupby("Product ID")["Units Sold"].shift(lag)

            # rolling stats for short-term behaviour
            df_model["Sales_RollingMean_7"] = df_model.groupby("Product ID")["Units Sold"].transform(
                lambda x: x.rolling(7, min_periods=1).mean()
            )
            df_model["Sales_RollingStd_7"] = df_model.groupby("Product ID")["Units Sold"].transform(
                lambda x: x.rolling(7, min_periods=1).std()
            )

            df_model.fillna(0, inplace=True)

            cat_cols = ["Product ID", "Category", "Weather Condition", "Holiday/Promotion", "Seasonality"]
            le_map = {}
            for c in cat_cols:
                le = LabelEncoder()
                df_model[c] = le.fit_transform(df_model[c].astype(str))
                le_map[c] = le

            feat_cols = [
                "Sales_Lag_1",
                "Sales_Lag_7",
                "Sales_RollingMean_7",
                "Sales_RollingStd_7",
                "DayOfWeek",
                "Month",
                "Price",
                "Discount",
            ] + cat_cols

            X_raw = df_model[feat_cols]
            y = df_model["Units Sold"].astype(float)
            y_log = np.log1p(np.clip(y, 0, None))

            X_tr_raw, X_te_raw, y_tr_log, y_te_log = train_test_split(
                X_raw, y_log, test_size=0.2, shuffle=False
            )

            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X_tr_raw)
            X_te = scaler.transform(X_te_raw)

            rf = RandomForestRegressor(n_estimators=200, min_samples_leaf=2, random_state=42)
            xgb = XGBRegressor(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.08,
                subsample=0.9,
                colsample_bytree=0.9,
                random_state=42,
                objective="reg:squarederror",
                verbosity=0,
            )
            lr = LinearRegression()

            model = VotingRegressor([("rf", rf), ("xgb", xgb), ("lr", lr)], weights=[2, 3, 1])
            model.fit(X_tr, y_tr_log)

            y_pred_log = model.predict(X_te)
            y_te = np.expm1(y_te_log)
            y_pred = np.expm1(y_pred_log)

            df_eval = df_model.iloc[X_te_raw.index].copy()
            df_eval["Actual Units"] = y_te
            df_eval["Predicted Units"] = y_pred
            df_eval["Error"] = df_eval["Actual Units"] - df_eval["Predicted Units"]

            mae = mean_absolute_error(y_te, y_pred)
            r2 = r2_score(y_te, y_pred)

        # --rolling forecast
        with st.spinner("Generating rolling forecast..."):
            future_rows = []
            prod_ids = df_filt["Product ID"].unique()

            for pid in prod_ids:
                hist = df_filt[df_filt["Product ID"] == pid].sort_values("Date")
                if hist.empty:
                    continue

                sales_hist = hist["Units Sold"].astype(float).tolist()
                date_hist = hist["Date"].tolist()

                if has_inv:
                    inv_level = float(hist["Inventory Level"].iloc[-1])
                else:
                    inv_level = np.nan

                last_row = hist.iloc[-1]

                for _ in range(h):
                    cur_date = date_hist[-1] + pd.Timedelta(days=1)

                    # last 30 days window
                    recent = sales_hist[-30:] if len(sales_hist) else [0.0]

                    def get_lag(k: int) -> float:
                        if len(recent) >= k:
                            return float(recent[-k])
                        return float(recent[-1]) if recent else 0.0

                    lag1 = get_lag(1)
                    lag7 = get_lag(7)

                    recent7 = recent[-7:] if len(recent) else [0.0]
                    roll_mean7 = float(np.mean(recent7))
                    roll_std7 = float(np.std(recent7)) if len(recent7) > 1 else 0.0

                    feat_row = {
                        "Sales_Lag_1": lag1,
                        "Sales_Lag_7": lag7,
                        "Sales_RollingMean_7": roll_mean7,
                        "Sales_RollingStd_7": roll_std7,
                        "DayOfWeek": cur_date.dayofweek,
                        "Month": cur_date.month,
                        "Price": float(last_row["Price"]),
                        "Discount": float(last_row["Discount"]),
                        "Product ID": pid,
                        "Category": last_row["Category"],
                        "Weather Condition": last_row.get("Weather Condition", "Unknown"),
                        "Holiday/Promotion": last_row.get("Holiday/Promotion", 0),
                        "Seasonality": last_row.get("Seasonality", "Unknown"),
                    }

                    enc_row = feat_row.copy()
                    for c in cat_cols:
                        val = str(enc_row[c])
                        try:
                            enc_row[c] = le_map[c].transform([val])[0]
                        except Exception:
                            enc_row[c] = 0

                    X_row = pd.DataFrame([enc_row])[feat_cols]
                    X_row_scaled = scaler.transform(X_row)

                    pred_log = model.predict(X_row_scaled)[0]
                    pred_units = max(float(np.expm1(pred_log)), 0.0)

                    # promo uplift rule
                    if str(last_row.get("Holiday/Promotion", 0)) in ["1", "True", "true", "yes", "Yes"]:
                        pred_units *= 1.2

                    safety = max(svc * roll_std7, 20.0)

                    if has_inv:
                        reorder_pt = pred_units * (lead_time / 7.0) + safety
                        reorder_qty = max(reorder_pt - inv_level, 0.0)
                        stock_flag = "Reorder" if reorder_qty > 0 else "OK"
                    else:
                        reorder_pt = np.nan
                        reorder_qty = np.nan
                        stock_flag = "N/A"

                    future_rows.append(
                        {
                            "Product ID": pid,
                            "Category": last_row["Category"],
                            "Forecast Date": cur_date,
                            "Forecast Units": pred_units,
                            "Inventory Level": inv_level,
                            "Safety Stock": safety,
                            "Reorder Suggestion_Daily": reorder_qty,
                            "Stock Status_Daily": stock_flag,
                        }
                    )

                    if has_inv:
                        inv_level = max(inv_level - pred_units, 0)
                    sales_hist.append(pred_units)
                    date_hist.append(cur_date)

        future_df = pd.DataFrame(future_rows)
        st.session_state["df_future"] = future_df
        st.session_state["df_eval"] = df_eval
        st.session_state["mae"] = mae
        st.session_state["r2"] = r2
        st.session_state["run_forecast"] = False

        run_msg.empty()

    # --summary table
    grp = (
        future_df.groupby(["Product ID", "Category"], as_index=False)
        .agg(
            {
                "Forecast Units": "sum",
                "Reorder Suggestion_Daily": "sum",
                "Forecast Date": ["min", "max"],
            }
        )
    )

    grp.columns = [
        "Product ID",
        "Category",
        "Total Forecast Units",
        "Total Reorder Suggestion",
        "Forecast Start",
        "Forecast End",
    ]

    grp["Avg Daily Forecast"] = grp["Total Forecast Units"] / h

    if has_inv:
        grp["Stock Status"] = np.where(grp["Total Reorder Suggestion"] > 0, "Reorder", "OK")
    else:
        grp["Stock Status"] = "N/A"

    export_df = filter_by_cat(grp)

    for c in ["Total Forecast Units", "Avg Daily Forecast", "Total Reorder Suggestion"]:
        export_df[c] = export_df[c].fillna(0).round(1)

    cols_out = [
        "Product ID",
        "Category",
        "Total Forecast Units",
        "Avg Daily Forecast",
        "Total Reorder Suggestion",
        "Stock Status",
        "Forecast Start",
        "Forecast End",
    ]
    export_df = export_df[cols_out]

    # --tabs for views 
    with st.container():
        tab1, tab2 = st.tabs(["Forecast overview", "Historical & model view"])

        # overview tab
        with tab1:
            st.subheader("Forecast summary by product and category")

            df_view = export_df.copy()
            df_view["Forecast Period"] = (
                df_view["Forecast Start"].dt.strftime("%Y-%m-%d")
                + " to "
                + df_view["Forecast End"].dt.strftime("%Y-%m-%d")
            )

            st.dataframe(
                df_view[
                    [
                        "Product ID",
                        "Category",
                        "Total Forecast Units",
                        "Total Reorder Suggestion",
                        "Stock Status",
                        "Forecast Period",
                    ]
                ]
            )

            if has_inv:
                st.subheader("Top reorder risk (by total suggested qty)")
                top_reorder = (
                    export_df.groupby(["Product ID", "Category"])["Total Reorder Suggestion"]
                    .sum()
                    .sort_values(ascending=False)
                    .head(10)
                    .reset_index()
                )
                top_reorder["Total Reorder Suggestion"] = (
                    top_reorder["Total Reorder Suggestion"].round(0).astype(int)
                )
                st.dataframe(top_reorder)
            else:
                st.info("Reorder suggestions are not available because Inventory Level is missing.")

            csv_bytes = export_df.to_csv(index=False).encode()
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            fname = f"Forecast_Summary_{ts}.csv"

            st.download_button(
                "Download forecast summary as CSV",
                data=csv_bytes,
                file_name=fname,
                mime="text/csv",
            )

            st.subheader("Future demand trend")
            prod_opts = ["All Products"] + sorted(future_df["Product ID"].unique().tolist())
            prod_sel = st.selectbox("Select product for trend view", prod_opts)

            if prod_sel == "All Products":
                trend_df = (
                    future_df.groupby("Forecast Date", as_index=False)["Forecast Units"]
                    .sum()
                    .rename(columns={"Forecast Units": "Total Forecast Units"})
                )
                y_col = "Total Forecast Units"
                title = "All products – total forecast units"
            else:
                trend_df = (
                    future_df[future_df["Product ID"] == prod_sel]
                    .groupby("Forecast Date", as_index=False)["Forecast Units"]
                    .sum()
                )
                y_col = "Forecast Units"
                title = f"Product {prod_sel} – forecast units"

            trend_df = trend_df.sort_values("Forecast Date")

            fig_tr, ax_tr = plt.subplots(figsize=(10, 4))
            ax_tr.plot(trend_df["Forecast Date"], trend_df[y_col], marker="o")
            ax_tr.set_title(title)
            ax_tr.set_xlabel("Forecast date")
            ax_tr.set_ylabel("Units")
            ax_tr.grid(True, alpha=0.3)
            st.pyplot(fig_tr)

        # historical & model tab
        with tab2:
            st.subheader("Historical and model analysis")

            st.markdown("### Top historical products (by total units sold)")
            hist_prod = (
                df.groupby("Product ID")["Units Sold"]
                .sum()
                .sort_values(ascending=False)
                .head(10)
            )

            fig_hist, ax_hist = plt.subplots(figsize=(8, 4))
            sns.barplot(x=hist_prod.values, y=hist_prod.index, ax=ax_hist, palette="viridis")
            ax_hist.set_xlabel("Total units sold (historical)")
            ax_hist.set_ylabel("Product ID")
            ax_hist.set_title("Top historical products")
            st.pyplot(fig_hist)

            st.markdown("### Historical total sales trend")
            hist_trend = (
                df.groupby("Date", as_index=False)["Units Sold"]
                .sum()
                .sort_values("Date")
            )

            fig_ts, ax_ts = plt.subplots(figsize=(10, 4))
            ax_ts.plot(hist_trend["Date"], hist_trend["Units Sold"], marker="o")
            ax_ts.set_xlabel("Date")
            ax_ts.set_ylabel("Total units sold")
            ax_ts.set_title("Historical total sales trend")
            ax_ts.grid(True, alpha=0.3)
            st.pyplot(fig_ts)

            st.markdown("### Historical sales heatmap by category and month")
            if has_cat:
                heat_df = (
                    df.groupby(["Category", "Month"])["Units Sold"]
                    .sum()
                    .unstack("Month")
                    .fillna(0)
                )

                fig_hm, ax_hm = plt.subplots(figsize=(8, 4))
                sns.heatmap(heat_df, cmap="viridis", annot=False, ax=ax_hm)
                ax_hm.set_title("Historical sales heatmap by category and month")
                ax_hm.set_xlabel("Month")
                ax_hm.set_ylabel("Category")
                st.pyplot(fig_hm)
            else:
                st.info("Cannot display heatmap because Category is missing in the data.")

            # --- BASIC MODEL PERFORMANCE (simple view)
            st.markdown(
                f"""
                ### Model Performance (Test Set)

                - **Average Error (MAE)**: {mae:.2f} units  
                - **Model Fit (R²)**: {r2:.3f}  (closer to 1.0 is better)
                """
            )

            if r2 > 0.5:
                st.success("The model performs reasonably well and can support decision-making.")
            elif r2 > 0:
                st.info("The model shows moderate predictive ability.")
            else:
                st.warning("The model struggles on the test set. Might be worth checking data quality or feature choices.")

            # --- ADVANCED (expander)
            with st.expander("Advanced Model Evaluation (Details & Charts)"):

                # --- compute RMSE
                rmse = np.sqrt(np.mean((df_eval["Actual Units"] - df_eval["Predicted Units"])**2))

                # --- MAPE (only where actual > 0)
                df_nonzero = df_eval[df_eval["Actual Units"] > 0].copy()
                if len(df_nonzero) > 0:
                    mape = np.mean(
                        np.abs((df_nonzero["Actual Units"] - df_nonzero["Predicted Units"]) / df_nonzero["Actual Units"])
                    ) * 100
                else:
                    mape = np.nan

                # --- sMAPE 
                smape = np.mean(
                    200 * np.abs(df_eval["Actual Units"] - df_eval["Predicted Units"]) /
                    (np.abs(df_eval["Actual Units"]) + np.abs(df_eval["Predicted Units"]) + 1e-9)
                )

                # --- baseline: naive forecast 
                df_eval_sorted = df_eval.sort_values("Date").copy()
                df_eval_sorted["Naive Forecast"] = df_eval_sorted["Actual Units"].shift(1)

                df_eval_sorted["Naive Forecast"].fillna(df_eval_sorted["Actual Units"].mean(), inplace=True)
                df_eval_sorted["Actual Units"].fillna(0, inplace=True)
                df_eval_sorted["Naive Forecast"].replace([np.inf, -np.inf], np.nan, inplace=True)
                df_eval_sorted["Naive Forecast"].fillna(df_eval_sorted["Actual Units"].mean(), inplace=True)

                baseline_mae = mean_absolute_error(df_eval_sorted["Actual Units"], df_eval_sorted["Naive Forecast"])
                baseline_rmse = np.sqrt(np.mean((df_eval_sorted["Actual Units"] - df_eval_sorted["Naive Forecast"])**2))

                st.markdown(
                    f"""
                    #### Metrics  
                    - **MAE**: {mae:.2f}  
                    - **RMSE**: {rmse:.2f}  
                    - **MAPE** (actual > 0 only): {mape:.2f}%  
                    - **sMAPE** (zero‑safe): {smape:.2f}%  
                    - **R²**: {r2:.3f}  

                    #### Baseline Comparison (Naive Forecast)
                    - **Baseline MAE**: {baseline_mae:.2f}  
                    - **Baseline RMSE**: {baseline_rmse:.2f}  
                    """
                )

                # Chart 1: Actual vs Predicted
                st.markdown("#### Actual vs Predicted (Scatter Plot)")
                fig1, ax1 = plt.subplots(figsize=(6,4))
                ax1.scatter(df_eval["Actual Units"], df_eval["Predicted Units"], alpha=0.6)
                ax1.plot([0, df_eval["Actual Units"].max()], 
                        [0, df_eval["Actual Units"].max()], 
                        color="red", linestyle="--")
                ax1.set_xlabel("Actual Units")
                ax1.set_ylabel("Predicted Units")
                ax1.set_title("Actual vs Predicted")
                st.pyplot(fig1)

                # Chart 2: Residual Plot
                st.markdown("#### Residual Plot")
                df_eval["Residual"] = df_eval["Actual Units"] - df_eval["Predicted Units"]
                fig2, ax2 = plt.subplots(figsize=(6,4))
                ax2.scatter(df_eval["Predicted Units"], df_eval["Residual"], alpha=0.6)
                ax2.axhline(0, color="red", linestyle="--")
                ax2.set_xlabel("Predicted Units")
                ax2.set_ylabel("Residual (Actual - Predicted)")
                ax2.set_title("Residual Plot")
                st.pyplot(fig2)

