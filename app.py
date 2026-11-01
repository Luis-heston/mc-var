import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from risk import (
    estimate_params,
    corr_matrix,
    simulate_terminal_returns,
    simulate_paths,
    portfolio_pnl,
    var_es,
    parametric_var,
    historical_var,
)

st.set_page_config(page_title="Monte Carlo VaR", layout="centered")
st.title("Monte Carlo VaR: portfolio")

COLS = ["Ticker", "Avg entry price", "Invested"]
# Placeholder positions only. Do not put real positions here: the repo is public.
DEFAULT = pd.DataFrame(
    {"Ticker": ["AAPL", "MSFT"], "Avg entry price": [200.0, 400.0], "Invested": [50_000.0, 50_000.0]}
)
CMAP = LinearSegmentedColormap.from_list("div", ["#2a6fb0", "#f0efec", "#c8442f"])  # blue - neutral - red

if "positions" not in st.session_state:
    st.session_state.positions = DEFAULT.copy()
    st.session_state.ver = 0

# ---------- Settings ----------
with st.sidebar:
    st.header("Settings")
    period = st.selectbox("History window", ["1y", "2y", "3y", "5y"], index=1)
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

# ---------- Positions ----------
st.subheader("Positions")
st.caption("Edit the table, use the + row at the bottom or the button below to add a stock. "
           "Prices in the stock's own currency.")
edited = st.data_editor(
    st.session_state.positions,
    key=f"pos_{st.session_state.ver}",
    num_rows="dynamic",
    hide_index=True,
    width="stretch",
    column_config={
        "Ticker": st.column_config.TextColumn("Ticker", required=True),
        "Avg entry price": st.column_config.NumberColumn("Avg entry price", min_value=0.0, format="%.2f", step=0.01),
        "Invested": st.column_config.NumberColumn("Invested", min_value=0.0, format="%.0f", step=100.0),
    },
)

if st.button("➕ Add stock"):
    blank = pd.DataFrame([{"Ticker": "", "Avg entry price": np.nan, "Invested": np.nan}])
    st.session_state.positions = pd.concat([edited, blank], ignore_index=True)
    st.session_state.ver += 1
    st.rerun()

with st.expander("Save / load portfolio"):
    st.download_button("Download positions (CSV)", edited.to_csv(index=False), "positions.csv", "text/csv")
    up = st.file_uploader("Load positions CSV", type="csv")
    if up is not None and st.session_state.get("loaded_id") != up.file_id:
        try:
            loaded = pd.read_csv(up)
            if not set(COLS).issubset(loaded.columns):
                raise ValueError(f"CSV needs the columns: {', '.join(COLS)}")
            st.session_state.positions = loaded[COLS]
            st.session_state.loaded_id = up.file_id
            st.session_state.ver += 1
            st.rerun()
        except Exception as e:
            st.error(f"Could not load file: {e}")

# ---------- Clean positions ----------
pos = edited.copy()
pos["Ticker"] = pos["Ticker"].fillna("").astype(str).str.upper().str.strip()
pos = pos[pos["Ticker"] != ""]
valid = (pos["Avg entry price"] > 0) & (pos["Invested"] > 0)
if (~valid).any():
    st.warning("Ignoring rows with a missing or zero entry price or amount: " + ", ".join(pos.loc[~valid, "Ticker"]))
pos = pos[valid]
if pos["Ticker"].nunique() < 2:
    st.info("Add at least two stocks with an entry price and an amount invested.")
    st.stop()

# Same ticker on several rows: add them up (several purchases)
pos = pos.assign(Shares=pos["Invested"] / pos["Avg entry price"])
agg = pos.groupby("Ticker", sort=False).agg(Shares=("Shares", "sum"), Invested=("Invested", "sum"))
agg["Entry"] = agg["Invested"] / agg["Shares"]
tickers = list(agg.index)


@st.cache_data(ttl=3600, show_spinner="Downloading prices...")
def load_prices(tks, per):
    import yfinance as yf

    df = yf.download(list(tks), period=per, auto_adjust=True, progress=False)["Close"]
    if isinstance(df, pd.Series):
        df = df.to_frame(tks[0])
    return df.reindex(columns=list(tks))


try:
    prices = load_prices(tuple(tickers), period)
except Exception as e:
    st.error(f"Could not download data: {e}")
    st.stop()

bad = [t for t in tickers if prices[t].isna().all()]
if bad:
    st.error("No price data for: " + ", ".join(bad) + ". Check the ticker spelling (Yahoo format, e.g. SAN.MC).")
    st.stop()
prices = prices.dropna()
if prices.shape[0] < 60:
    st.error("Fewer than 60 common trading days. Use a longer history window or remove very recent listings.")
    st.stop()

# ---------- Portfolio and weights ----------
last = prices.iloc[-1]
agg["Last"] = last.reindex(agg.index)
agg["Value"] = agg["Shares"] * agg["Last"]
total_value = float(agg["Value"].sum())
total_invested = float(agg["Invested"].sum())
weights = (agg["Value"] / total_value).to_numpy()

# ---------- Estimation ----------
log_ret = np.log(prices / prices.shift(1)).dropna()[tickers].to_numpy()
mu, cov = estimate_params(log_ret)
if zero_drift:
    mu = np.zeros_like(mu)
vols = np.sqrt(np.diag(cov)) * np.sqrt(252)
corr = corr_matrix(cov)

a, b, c = st.columns(3)
a.metric("Invested", f"{total_invested:,.0f}")
b.metric("Current value", f"{total_value:,.0f}")
c.metric("Unrealised P&L", f"{total_value - total_invested:,.0f}", f"{total_value / total_invested - 1:.2%}")

detail = pd.DataFrame(
    {
        "Ticker": tickers,
        "Entry": agg["Entry"].to_numpy(),
        "Last": agg["Last"].to_numpy(),
        "Invested": agg["Invested"].to_numpy(),
        "Value": agg["Value"].to_numpy(),
        "P&L": agg["Value"].to_numpy() / agg["Invested"].to_numpy() - 1,
        "Weight": weights,
        "Ann. vol": vols,
    }
)
st.dataframe(
    detail.style.format(
        {"Entry": "{:,.2f}", "Last": "{:,.2f}", "Invested": "{:,.0f}", "Value": "{:,.0f}",
         "P&L": "{:+.1%}", "Weight": "{:.1%}", "Ann. vol": "{:.1%}"}
    ),
    hide_index=True,
    width="stretch",
)
st.caption(f"Risk estimated from {len(log_ret)} daily returns ({prices.index[0]:%d %b %Y} to {prices.index[-1]:%d %b %Y}). "
           "Weights use current market value.")

# ---------- Simulation ----------
s = None if seed == 0 else int(seed)
terminal = simulate_terminal_returns(mu, cov, horizon, n_sims, seed=s)
pnl = portfolio_pnl(terminal, weights, total_value)
var, es = var_es(pnl, conf)
p_var = parametric_var(mu, cov, weights, horizon, conf, total_value)
h_var = historical_var(log_ret, weights, horizon, conf, total_value)

st.subheader(f"{horizon}-day risk at {conf:.1%} confidence")
m1, m2 = st.columns(2)
m1.metric("Monte Carlo VaR", f"{var:,.0f}", f"{var / total_value:.2%} of portfolio", delta_color="off")
m2.metric("Expected Shortfall", f"{es:,.0f}", f"{es / total_value:.2%} of portfolio", delta_color="off")
st.caption(
    f"There is a {1 - conf:.1%} chance of losing more than {var:,.0f} over {horizon} day(s). "
    f"When it does happen, the average loss is about {es:,.0f}."
)

comp = pd.DataFrame(
    {"Method": ["Monte Carlo", "Parametric (normal)", "Historical simulation"], "VaR": [var, p_var, h_var]}
)
comp["% of portfolio"] = comp["VaR"] / total_value
st.dataframe(
    comp.style.format({"VaR": "{:,.0f}", "% of portfolio": "{:.2%}"}),
    hide_index=True,
    width="stretch",
)

# ---------- Correlation matrix ----------
st.subheader("Correlation matrix")
n = len(tickers)
size = min(max(3.6, 0.8 * n + 1.8), 7)
figc, axc = plt.subplots(figsize=(size, size * 0.9))
im = axc.imshow(corr, cmap=CMAP, vmin=-1, vmax=1)
axc.set_xticks(range(n), tickers, rotation=45, ha="right")
axc.set_yticks(range(n), tickers)
for i in range(n):
    for j in range(n):
        axc.text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center", fontsize=9,
                 color="white" if abs(corr[i, j]) > 0.65 else "#222222")
axc.set_xticks(np.arange(-0.5, n), minor=True)
axc.set_yticks(np.arange(-0.5, n), minor=True)
axc.grid(which="minor", color="white", lw=2)
axc.tick_params(which="minor", length=0)
for sp in axc.spines.values():
    sp.set_visible(False)
figc.colorbar(im, ax=axc, fraction=0.046, pad=0.04, label="Correlation")
axc.set_title("Correlation of daily returns")
figc.tight_layout()
st.pyplot(figc)
plt.close(figc)
st.caption("Values near 1 mean the stocks move together, so diversification is weak. "
           "Values near 0 or below diversify the portfolio.")

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
plt.close(fig)

n_plot = 100
paths = simulate_paths(mu, cov, horizon, n_plot, seed=s)
port_paths = total_value * (np.exp(paths) @ weights)
fig2, ax2 = plt.subplots(figsize=(6, 3.6))
ax2.plot(port_paths.T, color="grey", alpha=0.25, lw=0.8)
ax2.axhline(total_value, color="black", lw=1)
ax2.axhline(total_value - var, color="red", ls="--", lw=1.5, label="Value at VaR")
ax2.set_xlabel("Day")
ax2.set_ylabel("Portfolio value")
ax2.set_title(f"{n_plot} sample paths")
ax2.legend()
st.pyplot(fig2)
plt.close(fig2)

with st.expander("Method and limits"):
    st.markdown(
        "- Daily log returns are estimated from the history window (covariance, and mean unless zero drift is on).\n"
        "- Returns are drawn from a multivariate normal using a Cholesky factor, so the correlations are preserved.\n"
        "- Weights come from current market value (shares = invested / entry price, times the last price).\n"
        "- VaR is the loss at the (1 - confidence) percentile of simulated P&L; ES is the average loss beyond it.\n"
        "- Normal returns understate fat tails, so compare with the historical figure.\n"
        "- All values are added up as if in one currency: no FX conversion is applied.\n"
        "- The results depend on the history window, which is a backward-looking estimate."
    )
