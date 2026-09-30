from __future__ import annotations

import pandas as pd
import streamlit as st

from .config import CONFIG
from .data import fetch_market_data
from .regime import classify_regime
from .strategy import score_signal


st.set_page_config(page_title="Gold AI Monitor", layout="wide")

st.title("Gold Intelligence Monitor")

if "df" not in st.session_state:
    with st.spinner("Downloading and processing gold data..."):
        df = fetch_market_data(symbol=CONFIG.symbol, period="3mo", interval=CONFIG.data_interval)
        df = classify_regime(df)
        df = score_signal(df, CONFIG.min_confidence)
        st.session_state.df = df

else:
    df = st.session_state.df

latest = df.iloc[-1]
st.metric("Close", f"{latest['close']:.2f}")
st.metric("Regime", latest["regime"])
st.metric("Signal", latest["signal"])
st.metric("Confidence", f"{latest['signal_confidence']:.2f}")

left, right = st.columns(2)
with left:
    st.line_chart(df[["close", "ema_fast", "ema_slow"]])
with right:
    st.line_chart(df[["rsi"]])

st.dataframe(df.tail(30)[["close", "rsi", "regime", "signal", "signal_confidence", "signal_reason"]])
