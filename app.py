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
st.set_page_config(
    page_title="AI Stock Valuation Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS / styling ---
st.markdown(
    """
    <style>
    .stApp { background-color: #0E1117; color: #FAFAFA; }
    .stock-card {
        background: linear-gradient(180deg, rgba(24,26,33,0.92), rgba(14,16,22,0.88));
        border: 1px solid rgba(255,255,255,0.04);
        border-radius: 12px;
        padding: 18px;
        margin-bottom: 18px;
        box-shadow: 0 6px 18px rgba(0,0,0,0.45);
    }
    .stock-card h2 { margin: 0; font-size: 20px; font-weight:700; color:#FFFFFF; }
    .rec-badge { font-size: 28px; font-weight:800; padding:6px 12px; border-radius:8px; display:inline-block; margin-top:8px; }
    .risk-badge { font-weight:700; padding:6px 10px; border-radius:8px; display:inline-block; margin-left:12px; }
    .metric-grid { display:flex; gap:12px; flex-wrap:wrap; margin-top:12px; }
    .metric { min-width:140px; color:#E6EEF3; }
    .small-caption { color: #BFC9D9; font-size:12px; margin-top:6px; }
    .analyst { font-size:16px; font-weight:700; color:#FFFFFF; margin-top:10px; }
    .summary { margin-top:10px; color:#DDE7F2; font-weight:600; }
    .bullets { margin-top:6px; color:#BFC9D9; margin-left:18px; }
    .delta-up { color:#2ECC71; font-weight:700; }
    .delta-down { color:#FF6B6B; font-weight:700; }
    .kelly-label { font-weight:700; color:#FAFAFA; }
    .indicator-label { color:#BFC9D9; font-size:13px; margin-top:6px; }
    .risk-bar { width:100%; height:16px; border-radius:8px; overflow:hidden; margin-top:8px; border:1px solid rgba(255,255,255,0.06); }
    .risk-seg { height:100%; float:left; }
    .seg-low { background:#2E86AB; }       /* blue-ish */
    .seg-med { background:#FFD400; }       /* gold */
    .seg-high { background:#8B2E3A; }      /* burgundy */
    .risk-indicator { position:relative; top:-20px; font-weight:700; text-align:center; color:#000; }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- Sidebar ---
st.sidebar.markdown("<h3 style='color:#FAFAFA'>🎯 Investment Strategy</h3>", unsafe_allow_html=True)
st.sidebar.markdown("<div style='color:#BFC9D9; margin-bottom:8px'>Set your time horizon and risk preference</div>", unsafe_allow_html=True)
investment_horizon = st.sidebar.slider("Investment Horizon (days)", min_value=30, max_value=365, value=60, step=7)
risk_tolerance = st.sidebar.slider("Risk Tolerance (0.1 = Low, 1 = High)", 0.1, 1.0, 0.5)
portfolio_percent_invest = st.sidebar.slider("Portfolio % to invest in stocks", 0, 100, 70, step=5)
st.sidebar.markdown("---")
st.sidebar.markdown("<h4 style='color:#FAFAFA'>⚙️ Display</h4>", unsafe_allow_html=True)
show_analyst = st.sidebar.checkbox("Show analyst consensus (if available)", value=True)
st.sidebar.markdown("<div style='color:#BFC9D9; font-size:13px'>Tip: increase Investment Horizon for longer-term forecasts.</div>", unsafe_allow_html=True)
st.sidebar.markdown("---")

# --- Title & input ---
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
    score = 0.0
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
    if rsi < 30: score += weights["RSI"]
    elif rsi > 70: score -= weights["RSI"]
    try:
        ma50 = float(df["MA50"].iloc[-1])
        ma200 = float(df["MA200"].iloc[-1])
        close = float(df["Close"].iloc[-1])
        if close > ma50 > ma200: score += weights["MA"]
        elif close < ma50 < ma200: score -= weights["MA"]
    except Exception:
        pass
    pe = info.get("trailingPE")
    if pe:
        if pe < 15: score += weights["P/E"]
        elif pe > 25: score -= weights["P/E"]
    high_52 = info.get("fiftyTwoWeekHigh")
    low_52 = info.get("fiftyTwoWeekLow")
    if high_52 and close >= 0.9 * high_52: score -= weights["52w"]
    if low_52 and close <= 1.1 * low_52: score += weights["52w"]
    try:
        macd = float(df["MACD"].iloc[-1])
        signal = float(df["Signal"].iloc[-1])
        if macd > signal: score += weights["MACD"]
        elif macd < signal: score -= weights["MACD"]
    except Exception:
        pass
    if kelly_f > 0.5: score += weights["Kelly"]
    elif kelly_f < 0.1: score -= weights["Kelly"]
    return float(score)

def recommendation_from_score(score):
    # Finance-oriented subtle palette (dark green, navy/gold, burgundy)
    if score >= 0.6:
        return "Strong Buy", "#0B6E3A"  # dark green
    elif score >= 0.2:
        return "Buy", "#0F8A5F"
    elif score > -0.2:
        return "Hold", "#123A66"  # navy (gold text used separately)
    elif score > -0.6:
        return "Sell", "#8B2E3A"  # burgundy
    else:
        return "Strong Sell", "#6A0E1A"

def compute_risk_indicator(df):
    volatility = df['Close'].pct_change().std() * math.sqrt(252) if len(df) > 1 else 0.0
    # Map volatility to a discrete tier and numeric score 0..1
    if volatility < 0.25:
        return "Low", "#2ECC71", 0.1
    elif volatility < 0.5:
        return "Medium", "#FFD400", 0.5
    else:
        return "High", "#FF4136", 0.9

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
        risk_text, risk_color, risk_numeric = compute_risk_indicator(df)

        data_dict[ticker] = df
        info_dict[ticker] = stock.info if stock.info else {}
        ai_dict[ticker] = {"score": combined_score, "rec": rec_text, "color": rec_color, "kelly": kelly_f}
        forecast_dict[ticker] = (future_dates, forecast)
        fund_dict[ticker] = fund_score
        risk_dict[ticker] = {"color": risk_color, "text": risk_text, "numeric": risk_numeric}
    except Exception as e:
        st.error(f"Error fetching {ticker}: {e}")

if not data_dict:
    st.stop()

# --- Tabs ---
tab1, tab2, tab3 = st.tabs(["Summary & Recommendations", "Price & Forecast", "Technical Indicators"])

# --- Tab 1: Cards + Pie (risk bar under badge, segmented tier bar) ---
with tab1:
    st.subheader("Stock Recommendations")
    left_col, right_col = st.columns([2.3, 1])

    cards = list(data_dict.keys())
    with left_col:
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
                # analyst consensus safe handling
                analyst = None
                if show_analyst:
                    analyst = info.get("recommendationKey") or info.get("recommendationMean") or info.get("averageAnalystRating") or info.get("recommendationNumerical")
                else:
                    analyst = None

                # risk info
                risk_info = risk_dict.get(t, {})
                risk_text = risk_info.get("text", "N/A")
                risk_color = risk_info.get("color", "#777")
                risk_numeric = risk_info.get("numeric", 0.5)

                # Build card
                with cols[j]:
                    st.markdown("<div class='stock-card'>", unsafe_allow_html=True)
                    st.markdown(f"<h2>{t}</h2>", unsafe_allow_html=True)

                    # Recommendation badge
                    if rec_text.lower().startswith("hold"):
                        # navy badge with gold text for hold
                        badge_style = "background:#123A66; color:#FFD700;"
                    else:
                        badge_style = f"background:{rec_color}; color:#FFFFFF;"
                    st.markdown(f"<div class='rec-badge' style='{badge_style}'>{rec_text}</div>", unsafe_allow_html=True)

                    # Risk bar placed directly under badge (segmented tier bar)
                    # Segments: Low | Medium | High (equal width visually)
                    seg_low_pct = 33
                    seg_med_pct = 34
                    seg_high_pct = 33
                    # position indicator percent (0..100) derived from risk_numeric
                    pos_pct = int(min(max(risk_numeric, 0.0), 1.0) * 100)
                    # color segments and highlight by overlaying transparent indicator
                    st.markdown(
                        "<div style='margin-top:10px;'>"
                        "<div style='display:flex; align-items:center;'>"
                        f"<div class='risk-bar' style='flex:1;'>"
                        f"<div class='risk-seg seg-low' style='width:{seg_low_pct}%; float:left;'></div>"
                        f"<div class='risk-seg seg-med' style='width:{seg_med_pct}%; float:left;'></div>"
                        f"<div class='risk-seg seg-high' style='width:{seg_high_pct}%; float:left;'></div>"
                        "</div>"
                        f"<div style='width:110px; text-align:center; margin-left:12px;'><span style='font-weight:700; color:{risk_color}; background:rgba(255,255,255,0.06); padding:6px 8px; border-radius:8px'>{risk_text}</span></div>"
                        "</div>"
                        # indicator pointer - small textual overlay showing numeric position
                        f"<div style='margin-top:6px; color:#BFC9D9; font-size:13px'>Risk position: {pos_pct}% (low→high)</div>"
                        "</div>",
                        unsafe_allow_html=True,
                    )

                    # Price / delta / Kelly / Fund score grid
                    st.markdown("<div class='metric-grid'>", unsafe_allow_html=True)
                    st.markdown(f"<div class='metric'><strong>Price</strong><div>${close:,.2f}</div></div>", unsafe_allow_html=True)
                    # clearer delta with arrows and label
                    if day_change >= 0:
                        arrow = "▲"
                        delta_class = "delta-up"
                    else:
                        arrow = "▼"
                        delta_class = "delta-down"
                    st.markdown(f"<div class='metric'><strong>Today Change</strong><div class='{delta_class}'>{arrow} {abs(day_change):.2f} USD ({day_change_pct:+.2f}%)</div></div>", unsafe_allow_html=True)
                    st.markdown(f"<div class='metric'><strong>Kelly Criterion</strong><div>{ai.get('kelly',0.0):.2f}</div></div>", unsafe_allow_html=True)
                    st.markdown(f"<div class='metric'><strong>Fund Score</strong><div>{fund_dict.get(t,0.0):.2f}</div></div>", unsafe_allow_html=True)
                    st.markdown("</div>", unsafe_allow_html=True)

                    # Analyst consensus (bigger if available and enabled)
                    if show_analyst:
                        analyst_label = str(analyst) if analyst else "N/A"
                        st.markdown(f"<div class='analyst'>Analyst consensus: {analyst_label}</div>", unsafe_allow_html=True)

                    # Short summary sentence + 3 bullet points
                    summary_parts = []
                    if "RSI" in df.columns:
                        if df["RSI"].iloc[-1] > 70:
                            summary_parts.append("Short-term overbought")
                        elif df["RSI"].iloc[-1] < 30:
                            summary_parts.append("Short-term oversold")
                        else:
                            summary_parts.append("RSI neutral")
                    try:
                        ma50 = df["MA50"].iloc[-1]
                        ma200 = df["MA200"].iloc[-1]
                        if ma50 > ma200:
                            summary_parts.append("Trend bullish (MA50 > MA200)")
                        elif ma50 < ma200:
                            summary_parts.append("Trend bearish (MA50 < MA200)")
                    except Exception:
                        pass
                    short_summary = " / ".join(summary_parts[:2]) if summary_parts else "No short-term alerts"
                    st.markdown(f"<div class='summary'>{short_summary}</div>", unsafe_allow_html=True)

                    bullets = []
                    if "RSI" in df.columns:
                        bullets.append(f"RSI: {df['RSI'].iloc[-1]:.1f}")
                    bullets.append(f"Analyst: {analyst_label if show_analyst else 'hidden'}")
                    bullets.append(f"Kelly Criterion: {ai.get('kelly',0.0):.2f}")
                    st.markdown("<ul class='bullets'>", unsafe_allow_html=True)
                    for b in bullets:
                        st.markdown(f"<li style='margin-bottom:4px'>{b}</li>", unsafe_allow_html=True)
                    st.markdown("</ul>", unsafe_allow_html=True)

                    # mini RSI chart labeled
                    if "RSI" in df.columns:
                        rsi_series = df["RSI"].dropna()[-60:]
                        if len(rsi_series) > 3:
                            fig_rsi = go.Figure()
                            fig_rsi.add_trace(go.Scatter(x=rsi_series.index, y=rsi_series.values, mode='lines', line=dict(color='#F5B041'), name='RSI'))
                            fig_rsi.update_layout(template='plotly_dark', margin=dict(l=0,r=0,t=20,b=10), height=100, xaxis=dict(visible=False), legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1))
                            st.plotly_chart(fig_rsi, use_container_width=True)
                            st.markdown("<div class='indicator-label'>Mini RSI (14) — overbought >70 / oversold <30</div>", unsafe_allow_html=True)
                        else:
                            st.markdown(f"<div class='small-caption'>RSI: {rsi_val:.1f}</div>", unsafe_allow_html=True)

                    st.markdown("</div>", unsafe_allow_html=True)

    # Right: pie chart
    with right_col:
        st.subheader("Portfolio Allocation Guidance")
        stocks_pct = portfolio_percent_invest * (risk_tolerance)
        bonds_pct = (100 - portfolio_percent_invest) * 0.5 * (1 - risk_tolerance)
        mutual_pct = (100 - portfolio_percent_invest) * 0.3 * (1 - risk_tolerance)
        cash_pct = 100 - (stocks_pct + bonds_pct + mutual_pct)
        pie_labels = ["Stocks", "Bonds", "Mutual Funds", "Cash"]
        pie_values = [stocks_pct, bonds_pct, mutual_pct, cash_pct]
        pie_colors = ["#D35400", "#7F8C8D", "#2980B9", "#27AE60"]
        fig_pie = go.Figure(data=[go.Pie(labels=pie_labels, values=pie_values,
                                         marker=dict(colors=pie_colors, line=dict(color='black', width=2)),
                                         textinfo='label+percent')])
        fig_pie.update_layout(template='plotly_dark', height=520,
                              legend=dict(orientation="v", x=1.02, y=0.5, bgcolor='rgba(0,0,0,0)', bordercolor='rgba(255,255,255,0.08)', borderwidth=1),
                              margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig_pie, use_container_width=True)

# --- Tab 2: Price & Forecast with labels and legend ---
with tab2:
    st.subheader("Price Chart, Forecast & Support")
    for t in data_dict.keys():
        df = data_dict[t]
        future_dates, forecast = forecast_dict[t]
        support_price = df['Close'].rolling(window=20, min_periods=1).min().iloc[-1]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=df["Close"], mode='lines', name='Historical Close', line=dict(color='#FFFFFF', width=2)))
        if "MA50" in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df["MA50"], mode='lines', name='50-Day MA', line=dict(dash='dash', color='#F1C40F')))
        if "MA200" in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df["MA200"], mode='lines', name='200-Day MA', line=dict(dash='dot', color='#D35400')))
        fig.add_trace(go.Scatter(x=future_dates, y=forecast, mode='lines', name=f'Forecast ({investment_horizon}d)', line=dict(color='#FF00FF', width=2)))
        full_x = list(df.index) + list(future_dates)
        full_support = [support_price] * len(full_x)
        fig.add_trace(go.Scatter(x=full_x, y=full_support, mode='lines', name='Support (20d low)', line=dict(color='#00FF7F', dash='dot')))
        fig.update_layout(template='plotly_dark', height=560, title=f"{t} — Historical Price & Forecast",
                          xaxis_title="Date", yaxis_title="Price (USD)",
                          legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1))
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("<div style='color:#BFC9D9; font-size:13px'>Legend: White = Historical Close | Yellow = 50-Day MA | Orange = 200-Day MA | Magenta = Forecast | Green dotted = Support (20-day low)</div>", unsafe_allow_html=True)

# --- Tab 3: Technical indicators with bright MACD & Signal ---
with tab3:
    st.subheader("Technical Indicators")
    for t in data_dict.keys():
        df = data_dict[t]
        st.write(f"### {t}")
        col1, col2 = st.columns([1,1])
        with col1:
            if "RSI" in df.columns:
                fig_rsi = go.Figure()
                fig_rsi.add_trace(go.Scatter(x=df.index, y=df["RSI"], mode='lines', name='RSI', line=dict(color='#F5B041', width=2)))
                fig_rsi.update_layout(template='plotly_dark', height=300, yaxis=dict(range=[0,100]), title="RSI (14)")
                st.plotly_chart(fig_rsi, use_container_width=True)
                st.caption("RSI indicates overbought (>70) or oversold (<30).")
            else:
                st.write("RSI not available.")

        with col2:
            if "MACD" in df.columns and "Signal" in df.columns:
                fig_macd = go.Figure()
                fig_macd.add_trace(go.Scatter(x=df.index, y=df["MACD"], mode='lines', name='MACD', line=dict(color='#FF00FF', width=2)))
                fig_macd.add_trace(go.Scatter(x=df.index, y=df["Signal"], mode='lines', name='Signal', line=dict(color='#00FFFF', width=2)))
                fig_macd.update_layout(template='plotly_dark', height=300, title="MACD & Signal")
                st.plotly_chart(fig_macd, use_container_width=True)
                st.caption("MACD (magenta) and Signal (cyan). MACD crossing above Signal suggests bullish momentum; crossing below suggests bearish momentum.")
            else:
                st.write("MACD/Signal not available.")
