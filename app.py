import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from risk import (
    estimate_params,
    simulate_terminal_returns,
    simulate_paths,
    portfolio_pnl,
    var_es,
    parametric_var,
    historical_var,
)

st.set_page_config(page_title="Monte Carlo VaR", layout="centered")
st.title("Monte Carlo VaR: two stocks")

# ---------- Inputs ----------
with st.sidebar:
    st.header("Inputs")
    t1 = st.text_input("Ticker 1", "AAPL").upper().strip()
    t2 = st.text_input("Ticker 2", "MSFT").upper().strip()
    t3 = st.text_input("Ticker 3", "MSFT").upper().strip()
    t4 = st.text_input("Ticker 4", "MSFT").upper().strip()
    t5 = st.text_input("Ticker 5", "MSFT").upper().strip()
    Pricet1 = st.number_input("average entry")
    QT1= st.number_input("Quantity")
    Pricet2 = st.number_input("average entry")
    QT2= st.number_input("Quantity")
    Pricet3 = st.number_input("average entry")
    QT3= st.number_input("Quantity")
    Pricet4 = st.number_input("average entry")
    QT4= st.number_input("Quantity")
    Pricet5 = st.number_input("average entry")
    QT5= st.number_input("Quantity")
    
    
    value = st.number_input("Portfolio value", min_value=1000, value=100_000, step=1000)
    period = st.selectbox("History window", ["1y", "2y", "3y", "5y","10"], index=1)
    horizon = st.slider("Horizon (days)", 1, 60, 10)
    conf = st.select_slider("Confidence", options=[0.90, 0.95, 0.975, 0.99], value=0.95)
    n_sims = st.select_slider("Simulations", options=[10_000, 50_000, 100_000, 500_000], value=100_000)
    seed = st.number_input("Random seed (0 = random)", min_value=0, value=42, step=1)
    zero_drift = st.checkbox(
        "Assume zero drift",
        value=True,
        help="The average return in a short history window is very noisy and can make VaR look "
        "smaller after a strong run. Zero drift is the usual convention for short horizons.",
    )


@st.cache_data(ttl=3600, show_spinner="Downloading prices...")
def load_prices(a, b, per):
    import yfinance as yf

    df = yf.download([a, b], period=per, auto_adjust=True, progress=False)["Close"]
    return df[[a, b]].dropna()


try:
    prices = load_prices(t1, t2, period)
except Exception as e:
    st.error(f"Could not download data: {e}")
    st.stop()

if prices.shape[0] < 60 or prices.shape[1] != 2:
    st.error("Not enough data. Check the tickers.")
    st.stop()

# ---------- Estimation ----------
log_ret = np.log(prices / prices.shift(1)).dropna().to_numpy()
mu, cov = estimate_params(log_ret)
if zero_drift:
    mu = np.zeros_like(mu)
vols = np.sqrt(np.diag(cov)) * np.sqrt(252)
corr = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
weights = np.array([w1, 1 - w1])

c1, c2, c3 = st.columns(3)
c1.metric(f"{t1} ann. vol", f"{vols[0]:.1%}")
c2.metric(f"{t2} ann. vol", f"{vols[1]:.1%}")
c3.metric("Correlation", f"{corr:.2f}")

# ---------- Simulation ----------
s = None if seed == 0 else int(seed)
terminal = simulate_terminal_returns(mu, cov, horizon, n_sims, seed=s)
pnl = portfolio_pnl(terminal, weights, value)
var, es = var_es(pnl, conf)
p_var = parametric_var(mu, cov, weights, horizon, conf, value)
h_var = historical_var(log_ret, weights, horizon, conf, value)

st.subheader(f"{horizon}-day risk at {conf:.1%} confidence")
m1, m2 = st.columns(2)
m1.metric("Monte Carlo VaR", f"{var:,.0f}", f"{var / value:.2%} of portfolio", delta_color="off")
m2.metric("Expected Shortfall", f"{es:,.0f}", f"{es / value:.2%} of portfolio", delta_color="off")

st.caption(
    f"There is a {1 - conf:.1%} chance of losing more than {var:,.0f} over {horizon} day(s). "
    f"When it does happen, the average loss is about {es:,.0f}."
)

comp = pd.DataFrame(
    {
        "Method": ["Monte Carlo", "Parametric (normal)", "Historical simulation"],
        "VaR": [var, p_var, h_var],
    }
)
comp["% of portfolio"] = comp["VaR"] / value
st.dataframe(
    comp.style.format({"VaR": "{:,.0f}", "% of portfolio": "{:.2%}"}),
    hide_index=True,
    width="stretch",
)

# ---------- Charts ----------
fig, ax = plt.subplots(figsize=(6, 3.6))
ax.hist(pnl, bins=100, color="#4C78A8", alpha=0.85)
ax.axvline(-var, color="red", lw=2, label=f"VaR {conf:.0%}")
ax.axvline(-es, color="darkred", lw=2, ls="--", label="ES")
ax.set_xlabel("Portfolio P&L")
ax.set_ylabel("Frequency")
ax.set_title("Simulated P&L distribution")
ax.legend()
st.pyplot(fig)

n_plot = 100
paths = simulate_paths(mu, cov, horizon, n_plot, seed=s)
port_paths = value * (np.exp(paths) @ weights)
fig2, ax2 = plt.subplots(figsize=(6, 3.6))
ax2.plot(port_paths.T, color="grey", alpha=0.25, lw=0.8)
ax2.axhline(value, color="black", lw=1)
ax2.axhline(value - var, color="red", ls="--", lw=1.5, label="Value at VaR")
ax2.set_xlabel("Day")
ax2.set_ylabel("Portfolio value")
ax2.set_title(f"{n_plot} sample paths")
ax2.legend()
st.pyplot(fig2)

with st.expander("Method and limits"):
    st.markdown(
        "- Daily log returns are estimated from the history window (covariance, and mean unless zero drift is on).\n"
        "- Returns are drawn from a multivariate normal using a Cholesky factor, so the correlation is preserved.\n"
        "- VaR is the loss at the (1 - confidence) percentile of simulated P&L; ES is the average loss beyond it.\n"
        "- Normal returns understate fat tails, so compare with the historical figure.\n"
        "- The results depend on the history window, which is a backward-looking estimate."
    )
