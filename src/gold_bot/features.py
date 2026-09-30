from __future__ import annotations

import streamlit as st
import pandas as pd

from .config import CONFIG
from .data import fetch_market_data
from .features import compute_indicators
from .regime import classify_regime
from .strategy import score_signal


st.set_page_config(page_title="Gold AI Monitor", layout="wide")

st.title("Gold AI Trading Monitor")

if "df" not in st.session_state:
    with st.spinner("Downloading market data..."):
        df = fetch_market_data(symbol=CONFIG.symbol, period="3mo", interval=CONFIG.data_interval)
        df = compute_indicators(df)
        df = classify_regime(df)
        df = score_signal(df, CONFIG.min_confidence)
        st.session_state.df = df

else:
    df = st.session_state.df

latest = df.iloc[-1]

st.metric("Latest Close", f"{latest['close']:.2f}")
st.metric("Regime", latest["regime"])
st.metric("Signal", latest["signal"])
st.metric("Confidence", f"{latest['signal_confidence']:.2f}")

st.line_chart(df[["close", "ema_fast", "ema_slow"]])

st.dataframe(df.tail(25)[["timestamp", "close", "rsi", "regime", "signal", "signal_confidence"]])
