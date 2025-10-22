# app.py
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.linear_model import LinearRegression
import plotly.graph_objs as go

# --- Page Config ---
st.set_page_config(page_title="AI Stock Valuation Dashboard", layout="wide")

# --- Dark Theme CSS ---
st.markdown("""
<style>
.stApp {background-color: #0E1117; color: #FAFAFA;}
.css-1d391kg {background-color: #1A1D27;}
.stTitle, .stText {color: #FAFAFA;}
.stButton>button {background-color: #3B3F50; color: #FAFAFA;}
</style>
""", unsafe_allow_html=True)

# --- Title ---
st.title("AI Stock Valuation Dashboard")
st.write("Enter stock tickers (comma-separated, e.g., AAPL, TSLA, MSFT)")

# --- User input ---
tickers_input = st.text_input("Stock Tickers", value="AAPL, TSLA").upper()
tickers = [t.strip() for t in tickers_input.split(",") if t.strip()]

# --- Sidebar Filters ---
st.sidebar.header("Portfolio Settings & Risk")
investment_horizon = st.sidebar.slider("Investment Horizon (days)", min_value=30, max_value=180, value=60, step=10)
risk_tolerance = st.sidebar.slider("Risk Tolerance (0.1=Low, 1=High)", 0.1, 1.0, 0.5)
portfolio_percent_invest = st.sidebar.slider("Portfolio % to Invest in Stocks", 10, 100, 70, step=5)

# --- Helper Functions ---
def compute_technical_indicators(df):
    df["MA50"] = df["Close"].rolling(50).mean()
    df["MA200"] = df["Close"].rolling(200).mean()
    
    delta = df["Close"].diff()
    up = delta.clip(lower=0)
    down = -1*delta.clip(upper=0)
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
    future_dates = pd.date_range(df.index[-1]+pd.Timedelta(days=1), periods=days)
    X_future = future_dates.map(datetime.toordinal).values.reshape(-1,1)
    forecast = model.predict(X_future)
    return future_dates, forecast

def compute_kelly(df, forecast):
    returns = df['Close'].pct_change().dropna()
    p = (returns > 0).mean()
    expected_up = max(forecast) - df['Close'].iloc[-1]
    expected_down = df['Close'].iloc[-1] - min(forecast)
    b = expected_up / expected_down if expected_down != 0 else 1
    f = (b*p - (1-p)) / b if b != 0 else 0
    f = max(min(f, 1.0), 0.0)
    return f

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
    
    return score

def ai_score(df, info, kelly_f):
    score = 0
    weights = {"RSI":0.2, "MA":0.15, "P/E":0.15, "52w":0.1, "MACD":0.15, "Kelly":0.25}
    
    rsi = df["RSI"].iloc[-1]
    if rsi < 30: score += 1*weights["RSI"]
    elif rsi > 70: score -= 1*weights["RSI"]
    
    ma50 = df["MA50"].iloc[-1]
    ma200 = df["MA200"].iloc[-1]
    close = df["Close"].iloc[-1]
    if close > ma50 > ma200: score += 1*weights["MA"]
    elif close < ma50 < ma200: score -= 1*weights["MA"]
    
    pe = info.get("trailingPE")
    if pe:
        if pe < 15: score += 1*weights["P/E"]
        elif pe > 25: score -= 1*weights["P/E"]
    
    high_52 = info.get("fiftyTwoWeekHigh")
    low_52 = info.get("fiftyTwoWeekLow")
    if high_52 and close >= 0.9*high_52: score -= 1*weights["52w"]
    if low_52 and close <= 1.1*low_52: score += 1*weights["52w"]
    
    macd = df["MACD"].iloc[-1]
    signal = df["Signal"].iloc[-1]
    if macd > signal: score += 1*weights["MACD"]
    elif macd < signal: score -= 1*weights["MACD"]
    
    if kelly_f > 0.5: score += 1*weights["Kelly"]
    elif kelly_f < 0.1: score -= 1*weights["Kelly"]
    
    return score

def recommendation_from_score(score):
    if score >= 0.6: return "Strong Buy", "green"
    elif score >= 0.2: return "Buy", "lime"
    elif score > -0.2: return "Hold", "yellow"
    elif score > -0.6: return "Sell", "orange"
    else: return "Strong Sell", "red"

def compute_risk_indicator(df):
    volatility = df['Close'].pct_change().std() * np.sqrt(252)
    if volatility < 0.25: return "green", "Low Risk"
    elif volatility < 0.5: return "yellow", "Medium Risk"
    else: return "red", "High Risk"

# --- Fetch & Process Data ---
if tickers:
    data_dict, info_dict, ai_dict, forecast_dict, fund_dict, risk_dict, kelly_dict = {}, {}, {}, {}, {}, {}, {}
    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            df = stock.history(period="1y")
            if df.empty:
                st.warning(f"No data for {ticker}")
                continue
            df = compute_technical_indicators(df)
            future_dates, forecast = linear_forecast(df, investment_horizon)
            kelly_f = compute_kelly(df, forecast)
            tech_score = ai_score(df, stock.info, kelly_f)
            fund_score = fundamental_score(stock.info)
            combined_score = 0.6*tech_score + 0.4*fund_score
            rec, color = recommendation_from_score(combined_score)
            risk_color, risk_text = compute_risk_indicator(df)
            
            data_dict[ticker] = df
            info_dict[ticker] = stock.info
            ai_dict[ticker] = {"score": combined_score, "rec": rec, "color": color, "kelly": kelly_f}
            forecast_dict[ticker] = (future_dates, forecast)
            fund_dict[ticker] = fund_score
            risk_dict[ticker] = {"color": risk_color, "text": risk_text}
        except Exception as e:
            st.error(f"Error fetching {ticker}: {e}")

    if data_dict:
        tab1, tab2, tab3 = st.tabs(["Summary & Recommendations", "Price & Forecast", "Technical Indicators"])

        # --- Tab 1: Summary & Recommendations ---
        with tab1:
            st.subheader("Stock Recommendations")
            left_col, right_col = st.columns([2,1])  # Pie chart bigger
            with left_col:
                for t in tickers:
                    df = data_dict[t]
                    info = info_dict[t]
                    rec_color = ai_dict[t]["color"]
                    rec_text = ai_dict[t]["rec"]
                    close = df["Close"].iloc[-1]
                    risk_color = risk_dict[t]["color"]
                    risk_text = risk_dict[t]["text"]
                    fund_score_val = fund_dict[t]
                    kelly_f = ai_dict[t]["kelly"]

                    st.markdown(f"<h1 style='color:{rec_color}'>{rec_text}</h1>", unsafe_allow_html=True)
                    st.metric(label=f"{t} Current Price", value=f"${close:.2f}")
                    st.markdown(f"**Risk:** <span style='color:{risk_color}'>{risk_text}</span>", unsafe_allow_html=True)
                    st.caption(f"Fundamental Score: {fund_score_val:.2f} (0-1)")
                    st.caption(f"Kelly Fraction: {kelly_f:.2f}")

            # --- Enlarged Portfolio Pie Chart with Dark Borders & Legend beside ---
            with right_col:
                st.subheader("Portfolio Allocation Guidance")
                stocks = portfolio_percent_invest * risk_tolerance
                bonds = (100 - portfolio_percent_invest) * 0.5 * (1-risk_tolerance)
                mutual_funds = (100 - portfolio_percent_invest) * 0.3 * (1-risk_tolerance)
                cash = 100 - (stocks + bonds + mutual_funds)
                pie_labels = ["Stocks", "Bonds", "Mutual Funds", "Cash"]
                pie_values = [stocks, bonds, mutual_funds, cash]
                pie_colors = ["#2ECC71", "#3498DB", "#E67E22", "#95A5A6"]

                fig_pie = go.Figure(data=[go.Pie(
                    labels=pie_labels,
                    values=pie_values,
                    marker=dict(colors=pie_colors, line=dict(color='black', width=3)),  # Darker borders
                    textinfo='label+percent',
                )])
                fig_pie.update_layout(
                    template='plotly_dark',
                    height=700,
                    legend=dict(
                        orientation="v",
                        yanchor="top",
                        y=1,
                        xanchor="left",
                        x=1.05  # Pie chart directly beside legend
                    ),
                    margin=dict(t=50, b=0, l=0, r=0)
                )
                st.plotly_chart(fig_pie, use_container_width=True, key="global_pie_right")

        # --- Tab 2: Price & Forecast ---
        with tab2:
            st.subheader("Interactive Price Chart with Forecast")
            for t in tickers:
                df = data_dict[t]
                future_dates, forecast = forecast_dict[t]
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df.index, y=df["Close"], mode='lines', name=f"{t} Close"))
                fig.add_trace(go.Scatter(x=df.index, y=df["MA50"], mode='lines', name=f"{t} 50-Day MA", line=dict(dash='dash')))
                fig.add_trace(go.Scatter(x=df.index, y=df["MA200"], mode='lines', name=f"{t} 200-Day MA", line=dict(dash='dot')))
                fig.add_trace(go.Scatter(x=future_dates, y=forecast, mode='lines', name=f"{t} Forecast", line=dict(color='magenta')))
                fig.update_layout(template='plotly_dark', height=500, title=f"{t} Price & Forecast")
                st.plotly_chart(fig, use_container_width=True, key=f"price_{t}")

        # --- Tab 3: Technical Indicators ---
        with tab3:
            st.subheader("Technical Indicators Explained")
            for t in tickers:
                df = data_dict[t]
                st.write(f"### {t}")
                col1, col2 = st.columns(2)
                with col1:
                    st.line_chart(df["RSI"], use_container_width=True)
                    st.caption("RSI indicates overbought (>70) or oversold (<30).")
                    st.line_chart(df["MACD"], use_container_width=True)
                    st.caption("MACD shows momentum. Above signal line = bullish.")
                with col2:
                    st.line_chart(df["Signal"], use_container_width=True)
                    st.caption("MACD Signal Line: used with MACD for buy/sell signals.")
