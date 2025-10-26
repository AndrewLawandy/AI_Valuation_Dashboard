# app.py
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.linear_model import LinearRegression
import plotly.graph_objs as go
import math

# --- Page Config ---
st.set_page_config(page_title="AI Stock Valuation Dashboard", layout="wide", initial_sidebar_state="expanded")

# --- Minimal CSS for card layout and visual polish ---
st.markdown(
    """
    <style>
    /* App background */
    .stApp { background-color: #0E1117; color: #FAFAFA; }

    /* Card */
    .stock-card {
        background: linear-gradient(180deg, rgba(26,29,39,0.9), rgba(18,20,28,0.85));
        border: 1px solid rgba(255,255,255,0.04);
        border-radius: 12px;
        padding: 16px;
        margin-bottom: 16px;
        box-shadow: 0 6px 18px rgba(0,0,0,0.5);
    }
    .stock-card h2 { margin: 0; font-size: 20px; line-height: 1; }
    .stock-card .rec { font-size: 24px; font-weight:700; margin-top:6px; }
    .metric-grid { display:flex; gap:12px; flex-wrap:wrap; margin-top:8px; }
    .metric { min-width:120px; }
    .small-caption { color: #BFC9D9; font-size:12px; margin-top:6px; }

    /* Sidebar larger labels */
    .sidebar-heading { font-size:16px; font-weight:700; margin-bottom:6px; color:#FAFAFA; }
    .sidebar-sub { color:#BFC9D9; margin-bottom:12px; font-size:13px; }

    /* Make streamlit plots fit nicely inside cards */
    .stPlotlyChart > div { border-radius: 8px; overflow: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- Sidebar (styled) ---
st.sidebar.markdown("<div class='sidebar-heading'>🎯 Investment Strategy</div>", unsafe_allow_html=True)
st.sidebar.markdown("<div class='sidebar-sub'>Set your time horizon and risk preference</div>", unsafe_allow_html=True)
investment_horizon = st.sidebar.slider("Investment Horizon (days)", min_value=30, max_value=365, value=60, step=7)
risk_tolerance = st.sidebar.slider("Risk Tolerance (0.1 = Low, 1 = High)", 0.1, 1.0, 0.5)
portfolio_percent_invest = st.sidebar.slider("Portfolio % to invest in stocks", 0, 100, 70, step=5)
st.sidebar.markdown("---")
st.sidebar.markdown("<div class='sidebar-heading'>⚙️ Display</div>", unsafe_allow_html=True)
show_analyst = st.sidebar.checkbox("Show analyst consensus (if available)", value=True)
st.sidebar.markdown("---")

# --- Title / input ---
st.title("AI Stock Valuation Dashboard")
st.write("Enter stock tickers (comma-separated), e.g. `AAPL, TSLA, MSFT`")

tickers_input = st.text_input("Stock Tickers", value="AAPL, TSLA").upper()
tickers = [t.strip() for t in tickers_input.split(",") if t.strip()]
if not tickers:
    st.stop()

# --- Helper functions ---
def compute_technical_indicators(df):
    df = df.copy()
    df["MA50"] = df["Close"].rolling(50).mean()
    df["MA200"] = df["Close"].rolling(200).mean()
    delta = df["Close"].diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    roll_up = up.rolling(14).mean()
    roll_down = down.rolling(14).mean()
    rs = roll_up / roll_down
    df["RSI"] = 100 - (100 / (1 + rs))
    short_avg = df["Close"].ewm(span=12, adjust=False).mean()
    long_avg = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = short_avg - long_avg
    df["Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    return df

def linear_forecast(df, days=60):
    df = df.copy()
    df['DateOrdinal'] = pd.to_datetime(df.index).map(datetime.toordinal)
    X = df['DateOrdinal'].values.reshape(-1,1)
    y = df['Close'].values
    model = LinearRegression()
    model.fit(X, y)
    future_dates = pd.date_range(df.index[-1] + pd.Timedelta(days=1), periods=days)
    X_future = future_dates.map(datetime.toordinal).values.reshape(-1,1)
    forecast = model.predict(X_future)
    return future_dates, forecast

def compute_kelly(df, forecast):
    returns = df['Close'].pct_change().dropna()
    if len(returns) == 0:
        return 0.0
    p = (returns > 0).mean()
    expected_up = float(max(forecast)) - float(df['Close'].iloc[-1])
    expected_down = float(df['Close'].iloc[-1]) - float(min(forecast))
    if expected_down == 0:
        b = 1.0
    else:
        b = expected_up / expected_down if expected_down != 0 else 1.0
    if b == 0:
        return 0.0
    f = (b * p - (1 - p)) / b
    f = max(min(f, 1.0), 0.0)
    return float(f)

def fundamental_score(info):
    score = 0
    weights = {"PE":0.3, "PEG":0.2, "RevenueGrowth":0.25, "ROE":0.25}
    pe = info.get("trailingPE")
    if pe and pe < 25: score += weights["PE"]
    elif pe and pe > 40: score -= weights["PE"]
    peg = info.get("pegRatio")
    if peg and peg < 1: score += weights["PEG"]
    rev_growth = info.get("revenueGrowth")
    if rev_growth and rev_growth > 0.1: score += weights["RevenueGrowth"]
    roe = info.get("returnOnEquity")
    if roe and roe > 0.15: score += weights["ROE"]
    return max(min(score, 1.0), -1.0)

def ai_score(df, info, kelly_f):
    score = 0.0
    weights = {"RSI":0.2, "MA":0.15, "P/E":0.15, "52w":0.1, "MACD":0.15, "Kelly":0.25}
    try:
        rsi = float(df["RSI"].iloc[-1])
    except Exception:
        rsi = 50.0
    if rsi < 30: score += 1 * weights["RSI"]
    elif rsi > 70: score -= 1 * weights["RSI"]
    try:
        ma50 = float(df["MA50"].iloc[-1])
        ma200 = float(df["MA200"].iloc[-1])
        close = float(df["Close"].iloc[-1])
        if close > ma50 > ma200: score += 1 * weights["MA"]
        elif close < ma50 < ma200: score -= 1 * weights["MA"]
    except Exception:
        pass
    pe = info.get("trailingPE")
    if pe:
        if pe < 15: score += 1 * weights["P/E"]
        elif pe > 25: score -= 1 * weights["P/E"]
    high_52 = info.get("fiftyTwoWeekHigh")
    low_52 = info.get("fiftyTwoWeekLow")
    if high_52 and close >= 0.9 * high_52: score -= 1 * weights["52w"]
    if low_52 and close <= 1.1 * low_52: score += 1 * weights["52w"]
    try:
        macd = float(df["MACD"].iloc[-1])
        signal = float(df["Signal"].iloc[-1])
        if macd > signal: score += 1 * weights["MACD"]
        elif macd < signal: score -= 1 * weights["MACD"]
    except Exception:
        pass
    if kelly_f > 0.5: score += 1 * weights["Kelly"]
    elif kelly_f < 0.1: score -= 1 * weights["Kelly"]
    return float(score)

def recommendation_from_score(score):
    if score >= 0.6: return "Strong Buy", "#2ECC71"
    elif score >= 0.2: return "Buy", "#7CFC00"
    elif score > -0.2: return "Hold", "#FFD400"
    elif score > -0.6: return "Sell", "#FF7F50"
    else: return "Strong Sell", "#FF4136"

def compute_risk_indicator(df):
    volatility = df['Close'].pct_change().std() * math.sqrt(252) if len(df) > 1 else 0.0
    if volatility < 0.25: return "Low", "green"
    elif volatility < 0.5: return "Medium", "yellow"
    else: return "High", "red"

# --- Fetch & process data (safe) ---
data_dict, info_dict, ai_dict, forecast_dict, fund_dict, risk_dict = {}, {}, {}, {}, {}, {}

for ticker in tickers:
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period="1y")
        if df.empty:
            st.warning(f"No historical data for {ticker}")
            continue
        df = compute_technical_indicators(df)
        future_dates, forecast = linear_forecast(df, investment_horizon)
        kelly_f = compute_kelly(df, forecast)
        tech_score = ai_score(df, stock.info if stock.info else {}, kelly_f)
        fund_score = fundamental_score(stock.info if stock.info else {})
        combined_score = 0.6 * tech_score + 0.4 * fund_score
        rec_text, rec_color = recommendation_from_score(combined_score)
        risk_text, risk_color = compute_risk_indicator(df)

        data_dict[ticker] = df
        info_dict[ticker] = stock.info if stock.info else {}
        ai_dict[ticker] = {"score": combined_score, "rec": rec_text, "color": rec_color, "kelly": kelly_f}
        forecast_dict[ticker] = (future_dates, forecast)
        fund_dict[ticker] = fund_score
        risk_dict[ticker] = {"color": risk_color, "text": risk_text}
    except Exception as e:
        st.error(f"Error fetching {ticker}: {e}")

if not data_dict:
    st.stop()

# --- Tabs ---
tab1, tab2, tab3 = st.tabs(["Summary & Recommendations", "Price & Forecast", "Technical Indicators"])

# --- Tab 1: Card Grid + Big Pie on right ---
with tab1:
    st.subheader("Stock Recommendations")
    # Build left area (cards) and right area (global pie)
    left_col, right_col = st.columns([2.3, 1])  # more space for cards
    # Cards: two per row using chunking
    cards = list(data_dict.keys())

    with left_col:
        # iterate in pairs
        for i in range(0, len(cards), 2):
            cols = st.columns(2)
            for j in range(2):
                idx = i + j
                if idx >= len(cards):
                    cols[j].empty()
                    continue
                t = cards[idx]
                df = data_dict[t]
                info = info_dict.get(t, {})
                ai = ai_dict.get(t, {})
                rec_color = ai.get("color", "#FFFFFF")
                rec_text = ai.get("rec", "N/A")
                close = df["Close"].iloc[-1]
                prev_close = df["Close"].iloc[-2] if len(df) > 1 else close
                day_change = close - prev_close
                day_change_pct = (day_change / prev_close * 100) if prev_close != 0 else 0.0
                rsi_val = df["RSI"].iloc[-1] if "RSI" in df.columns else None
                analyst = info.get("recommendationKey") or info.get("recommendationMean") or info.get("recommendationNumerical") or info.get("averageAnalystRating") or None

                # Card HTML
                with cols[j]:
                    st.markdown("<div class='stock-card'>", unsafe_allow_html=True)
                    st.markdown(f"<h2 style='color:#FFFFFF'>{t}</h2>", unsafe_allow_html=True)
                    st.markdown(f"<div class='rec' style='color:{rec_color}'>{rec_text}</div>", unsafe_allow_html=True)
                    # metrics grid
                    st.markdown("<div class='metric-grid'>", unsafe_allow_html=True)
                    st.markdown(f"<div class='metric'><strong>Price</strong><div>${close:,.2f}</div></div>", unsafe_allow_html=True)
                    st.markdown(f"<div class='metric'><strong>Day Δ</strong><div>{day_change:+.2f} ({day_change_pct:+.2f}%)</div></div>", unsafe_allow_html=True)
                    st.markdown(f"<div class='metric'><strong>Kelly</strong><div>{ai.get('kelly', 0.0):.2f}</div></div>", unsafe_allow_html=True)
                    st.markdown(f"<div class='metric'><strong>Fund Score</strong><div>{fund_dict.get(t, 0.0):.2f}</div></div>", unsafe_allow_html=True)
                    st.markdown("</div>", unsafe_allow_html=True)

                    # RSI small chart and value
                    try:
                        rsi_series = df["RSI"].dropna()[-50:]
                        if len(rsi_series) > 3:
                            fig_rsi = go.Figure()
                            fig_rsi.add_trace(go.Scatter(x=rsi_series.index, y=rsi_series.values, mode='lines', line=dict(color='#F5B041')))
                            fig_rsi.update_layout(template='plotly_dark', margin=dict(l=0,r=0,t=10,b=10), height=90, xaxis=dict(visible=False))
                            st.plotly_chart(fig_rsi, use_container_width=True)
                        else:
                            st.markdown(f"<div class='small-caption'>RSI: {rsi_val:.1f}</div>", unsafe_allow_html=True)
                    except Exception:
                        pass

                    # Analyst consensus
                    if show_analyst:
                        analyst_label = str(analyst) if analyst else "N/A"
                        st.markdown(f"<div class='small-caption'>Analyst consensus: {analyst_label}</div>", unsafe_allow_html=True)

                    # small risk tag and more
                    risk_info = risk_dict.get(t, {})
                    st.markdown(f"<div class='small-caption'>Risk: <span style='color:{risk_info.get('color','white')}'>{risk_info.get('text','N/A')}</span></div>", unsafe_allow_html=True)

                    # separator
                    st.markdown("<hr style='border:0.5px solid rgba(255,255,255,0.06)'/>", unsafe_allow_html=True)

                    # quick bullets on indicators
                    try:
                        ma50 = df["MA50"].iloc[-1]
                        ma200 = df["MA200"].iloc[-1]
                        ma_text = "Bullish" if ma50 > ma200 else "Bearish" if ma50 < ma200 else "Flat"
                        st.markdown(f"**Trend (MA50 vs MA200):** {ma_text}", unsafe_allow_html=True)
                    except Exception:
                        pass
                    st.markdown("</div>", unsafe_allow_html=True)

    # Right column: global pie
    with right_col:
        st.subheader("Portfolio Allocation Guidance")
        stocks_pct = portfolio_percent_invest * (risk_tolerance)
        bonds_pct = (100 - portfolio_percent_invest) * 0.5 * (1 - risk_tolerance)
        mutual_pct = (100 - portfolio_percent_invest) * 0.3 * (1 - risk_tolerance)
        cash_pct = 100 - (stocks_pct + bonds_pct + mutual_pct)
        pie_labels = ["Stocks", "Bonds", "Mutual Funds", "Cash"]
        pie_values = [stocks_pct, bonds_pct, mutual_pct, cash_pct]
        # Stocks orange, Bonds grey, Mutual Funds blue, Cash green (darker tones)
        pie_colors = ["#D35400", "#7F8C8D", "#2980B9", "#27AE60"]
        fig_pie = go.Figure(data=[go.Pie(labels=pie_labels, values=pie_values,
                                         marker=dict(colors=pie_colors, line=dict(color='black', width=2)),
                                         textinfo='label+percent')])
        fig_pie.update_layout(template='plotly_dark', height=520,
                              legend=dict(orientation="v", x=1.02, y=0.5, bgcolor='rgba(0,0,0,0)', bordercolor='rgba(255,255,255,0.1)', borderwidth=1),
                              margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig_pie, use_container_width=True)

# --- Tab 2: Price & Forecast with Support Line and projection matching investment_horizon ---
with tab2:
    st.subheader("Price Chart, Forecast & Support Line")
    for t in data_dict.keys():
        df = data_dict[t]
        future_dates, forecast = forecast_dict[t]
        # support line: 20-day rolling low
        support_price = df['Close'].rolling(window=20, min_periods=1).min().iloc[-1]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=df["Close"], mode='lines', name=f"{t} Close", line=dict(color='white')))
        if "MA50" in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df["MA50"], mode='lines', name="50-Day MA", line=dict(dash='dash', color='#F1C40F')))
        if "MA200" in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df["MA200"], mode='lines', name="200-Day MA", line=dict(dash='dot', color='#D35400')))
        # forecast projection over investment_horizon (already computed)
        proj_x = future_dates
        proj_y = forecast
        fig.add_trace(go.Scatter(x=proj_x, y=proj_y, mode='lines', name=f"Forecast ({investment_horizon}d)", line=dict(color='#FF00FF')))
        # support line across historical + projection range
        full_x = list(df.index) + list(proj_x)
        full_support = [support_price] * len(full_x)
        fig.add_trace(go.Scatter(x=full_x, y=full_support, mode='lines', name='Support (20d low)', line=dict(color='#00FF7F', dash='dot')))
        fig.update_layout(template='plotly_dark', height=520, title=f"{t} Price & {investment_horizon}-day Forecast", xaxis_title="Date", yaxis_title="Price (USD)")
        st.plotly_chart(fig, use_container_width=True)

# --- Tab 3: Technical Indicators with bright MACD and Signal colors ---
with tab3:
    st.subheader("Technical Indicators: RSI & MACD")
    for t in data_dict.keys():
        df = data_dict[t]
        st.write(f"### {t}")
        col1, col2 = st.columns([1,1])
        with col1:
            # RSI
            if "RSI" in df.columns:
                fig_rsi = go.Figure()
                fig_rsi.add_trace(go.Scatter(x=df.index, y=df["RSI"], mode='lines', name='RSI', line=dict(color='#F5B041')))
                fig_rsi.update_layout(template='plotly_dark', height=280, yaxis=dict(range=[0,100]), title="RSI (14)")
                st.plotly_chart(fig_rsi, use_container_width=True)
                st.caption("RSI indicates overbought (>70) or oversold (<30).")
            else:
                st.write("RSI not available.")

        with col2:
            # MACD & Signal both in bright colors
            if "MACD" in df.columns and "Signal" in df.columns:
                fig_macd = go.Figure()
                fig_macd.add_trace(go.Scatter(x=df.index, y=df["MACD"], mode='lines', name='MACD', line=dict(color='#FF00FF', width=2)))
                fig_macd.add_trace(go.Scatter(x=df.index, y=df["Signal"], mode='lines', name='Signal', line=dict(color='#00FFFF', width=2)))
                fig_macd.update_layout(template='plotly_dark', height=280, title="MACD & Signal")
                st.plotly_chart(fig_macd, use_container_width=True)
                st.caption("MACD (magenta) and Signal (cyan). When MACD crosses above Signal it may indicate bullish momentum; crossing below may indicate bearish momentum.")
            else:
                st.write("MACD/Signal not available.")
