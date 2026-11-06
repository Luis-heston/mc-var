import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import PercentFormatter

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
from signals import (
    MONTHS,
    monthly_returns,
    year_month_table,
    seasonality_stats,
    crossover_signals,
    backtest,
    perf_stats,
    latest_status,
)

st.set_page_config(page_title="Portfolio risk", layout="centered")
st.title("Portfolio: risk, seasonality, momentum")

COLS = ["Ticker", "Avg entry price", "Invested"]
# Placeholder positions only. Do not put real positions here: the repo is public.
DEFAULT = pd.DataFrame(
    {"Ticker": ["AAPL", "MSFT"], "Avg entry price": [200.0, 400.0], "Invested": [50_000.0, 50_000.0]}
)
CMAP = LinearSegmentedColormap.from_list("div", ["#2a6fb0", "#f0efec", "#c8442f"])  # blue - neutral - red
RET_CMAP = LinearSegmentedColormap.from_list("ret", ["#c8442f", "#f0efec", "#2a6fb0"])  # losses red, gains blue
RET_CMAP.set_bad("#f3f3f1")

if "positions" not in st.session_state:
    st.session_state.positions = DEFAULT.copy()
    st.session_state.ver = 0

# ---------- Settings ----------
with st.sidebar:
    st.header("Risk settings")
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


@st.cache_data(ttl=6 * 3600, show_spinner="Downloading full price history...")
def load_history(tks):
    """Full daily history, each stock on its own dates (no common-window trimming)."""
    import yfinance as yf

    df = yf.download(list(tks), period="max", auto_adjust=True, progress=False)["Close"]
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

tab_risk, tab_season, tab_mom = st.tabs(["Risk (VaR)", "Seasonality", "Momentum"])


# =====================================================================  RISK
with tab_risk:
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

# Long history for the two tabs below
try:
    hist = load_history(tuple(tickers))
except Exception as e:
    hist = None
    hist_error = str(e)


def draw_heatmap(df, vmax, title, fmt="{:+.1%}", sep_last_row=False, fontsize=7.5):
    """Annotated heatmap, losses red / gains blue, missing cells grey."""
    data = df.to_numpy(dtype=float)
    nr, nc = data.shape
    fig, ax = plt.subplots(figsize=(7, min(max(2.4, 0.3 * nr + 1.3), 10)))
    im = ax.imshow(np.ma.masked_invalid(data), cmap=RET_CMAP, vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(nc), list(df.columns))
    ax.set_yticks(range(nr), [str(i) for i in df.index])
    ax.xaxis.tick_top()
    for i in range(nr):
        for j in range(nc):
            v = data[i, j]
            if np.isfinite(v):
                ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=fontsize,
                        color="white" if abs(v) > 0.6 * vmax else "#222222")
    ax.set_xticks(np.arange(-0.5, nc), minor=True)
    ax.set_yticks(np.arange(-0.5, nr), minor=True)
    ax.grid(which="minor", color="white", lw=1.5)
    ax.tick_params(which="minor", length=0)
    ax.tick_params(length=0)
    if sep_last_row:
        ax.axhline(nr - 1.5, color="white", lw=5)
    for sp in ax.spines.values():
        sp.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cb.ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.set_title(title, pad=22)
    fig.tight_layout()
    return fig


# =============================================================  SEASONALITY
with tab_season:
    st.subheader("Seasonality: best month by stock")
    if hist is None:
        st.error(f"Could not download the full history: {hist_error}")
    else:
        yrs = st.select_slider("Years of history", options=["5", "10", "15", "20", "All"], value="10")
        tables = {}
        for t in tickers:
            r = monthly_returns(hist[t])
            if yrs != "All" and len(r):
                r = r[r.index.year > r.index.year.max() - int(yrs)]
            tbl = year_month_table(r)
            if tbl.shape[0] >= 3:
                tables[t] = tbl
        short = [t for t in tickers if t not in tables]
        if short:
            st.warning("Less than 3 years of data, skipped: " + ", ".join(short))

        if tables:
            stats = {t: seasonality_stats(tbl) for t, tbl in tables.items()}
            avg = pd.DataFrame({t: stats[t]["Average"] for t in tables}).T  # rows = stocks, cols = months

            vmax_all = max(float(np.nanpercentile(np.abs(avg.to_numpy()), 95)), 0.01)
            st.pyplot(draw_heatmap(avg, vmax_all, "Average monthly return", fmt="{:+.1%}", fontsize=8))
            plt.close("all")

            rows = []
            for t, sdf in stats.items():
                best, worst = sdf["Average"].idxmax(), sdf["Average"].idxmin()
                rows.append(
                    {
                        "Ticker": t,
                        "Best month": best,
                        "Best avg": sdf.loc[best, "Average"],
                        "Hit rate": sdf.loc[best, "Hit rate"],
                        "Worst month": worst,
                        "Worst avg": sdf.loc[worst, "Average"],
                        "Years": int(sdf.loc[best, "Years"]),
                    }
                )
            st.dataframe(
                pd.DataFrame(rows).style.format({"Best avg": "{:+.1%}", "Worst avg": "{:+.1%}", "Hit rate": "{:.0%}"}),
                hide_index=True,
                width="stretch",
            )
            st.caption("Hit rate = share of years in which that month was positive. Blue = gains, red = losses.")

            st.subheader("Year by month")
            pick = st.selectbox("Stock", list(tables), key="season_pick")
            tbl = tables[pick]
            plot_tbl = pd.concat([tbl, pd.DataFrame([tbl.mean()], index=["Avg"])])
            vmax = max(float(np.nanpercentile(np.abs(tbl.to_numpy()), 95)), 0.02)
            st.pyplot(draw_heatmap(plot_tbl, vmax, f"{pick}: monthly return by year", sep_last_row=True))
            plt.close("all")

            sd = stats[pick].T[MONTHS]
            st.dataframe(
                sd.style.format(
                    {m: "{:+.1%}" for m in MONTHS}, na_rep=""
                ).format({m: "{:.0%}" for m in MONTHS}, subset=pd.IndexSlice[["Hit rate"], :])
                .format({m: "{:.0f}" for m in MONTHS}, subset=pd.IndexSlice[["Years"], :]),
                width="stretch",
            )

            with st.expander("How to read this"):
                st.markdown(
                    "- Each cell is the return from the last trading day of the previous month to the last trading day "
                    "of that month. The current month is left out until it is complete.\n"
                    "- Greyed cells have no data (the stock did not exist yet, or the month has not happened).\n"
                    "- With 10 years, each month has only about 10 observations. A month that looks best can easily "
                    "be luck, so treat this as a pattern to check, not a rule. A hit rate near 50% or a wide gap between "
                    "the best and worst year means the signal is weak.\n"
                    "- Returns are price returns adjusted for splits and dividends (Yahoo adjusted close)."
                )


# ================================================================  MOMENTUM
with tab_mom:
    st.subheader("Momentum: moving-average crossover")
    if hist is None:
        st.error(f"Could not download the full history: {hist_error}")
    else:
        c1, c2, c3, c4 = st.columns(4)
        kind = c1.radio("Average", ["SMA", "EMA"], horizontal=True)
        fast = int(c2.number_input("Fast window", min_value=5, max_value=150, value=50, step=5))
        slow = int(c3.number_input("Slow window", min_value=20, max_value=400, value=200, step=10))
        cost = float(c4.number_input("Cost per trade (bps)", min_value=0, max_value=100, value=5, step=1))
        recent = st.slider("Flag crossovers from the last N trading days", 1, 30, 10)
        st.caption("Entry signal: the fast average crosses above the slow one (uptrend starts). "
                   "Exit: it crosses back below.")

        if fast >= slow:
            st.warning("The fast window must be shorter than the slow window.")
        else:
            sigs, bts, skipped = {}, {}, []
            for t in tickers:
                px = hist[t].dropna()
                if len(px) < slow + 60:
                    skipped.append(t)
                    continue
                sigs[t] = crossover_signals(px, fast, slow, kind)
                bts[t] = backtest(sigs[t], cost)
            if skipped:
                st.warning(f"Not enough history for a {slow}-day average, skipped: " + ", ".join(skipped))

            if sigs:
                rows = []
                for t, sg in sigs.items():
                    stt = latest_status(sg, recent)
                    icon = {"Bullish": "▲ Bullish", "Bearish": "▼ Bearish"}.get(stt["trend"], "n/a")
                    sig_txt = "-"
                    if stt["signal"] != "-":
                        sig_txt = f"{'● ENTRY' if stt['signal'] == 'ENTRY' else '○ EXIT'} ({stt['days_since']}d ago)"
                    rows.append(
                        {
                            "Ticker": t,
                            "Trend": icon,
                            "Signal": sig_txt,
                            "Last cross": (f"{stt['last_event']} {stt['last_date']:%d %b %Y}"
                                           if stt["last_event"] else "-"),
                            "Price vs slow MA": stt["price_vs_slow"],
                        }
                    )
                st.dataframe(
                    pd.DataFrame(rows).style.format({"Price vs slow MA": "{:+.1%}"}),
                    hide_index=True,
                    width="stretch",
                )

                brow = []
                for t, bt in bts.items():
                    ps, pb = perf_stats(bt["strategy"]), perf_stats(bt["buy_hold"])
                    brow.append(
                        {
                            "Ticker": t,
                            "CAGR strategy": ps["CAGR"],
                            "CAGR buy&hold": pb["CAGR"],
                            "Max DD strategy": ps["Max drawdown"],
                            "Max DD buy&hold": pb["Max drawdown"],
                            "Trades": int(sigs[t].loc[bt.index, "entry"].sum()),
                            "Time invested": float(bt["position"].mean()),
                        }
                    )
                st.dataframe(
                    pd.DataFrame(brow).style.format(
                        {"CAGR strategy": "{:+.1%}", "CAGR buy&hold": "{:+.1%}", "Max DD strategy": "{:.0%}",
                         "Max DD buy&hold": "{:.0%}", "Time invested": "{:.0%}"}
                    ),
                    hide_index=True,
                    width="stretch",
                )

                st.subheader("Chart and signals")
                d1, d2 = st.columns([1, 2])
                pick_m = d1.selectbox("Stock", list(sigs), key="mom_pick")
                win = d2.radio("Window", ["6M", "1Y", "2Y", "5Y", "All"], index=2, horizontal=True)
                ndays = {"6M": 126, "1Y": 252, "2Y": 504, "5Y": 1260, "All": None}[win]

                sg, bt = sigs[pick_m], bts[pick_m]
                sl = sg.iloc[-ndays:] if ndays else sg
                bl = bt.iloc[-ndays:] if ndays else bt

                figm, (axp, axe) = plt.subplots(2, 1, figsize=(7, 6.2), height_ratios=[2.2, 1], sharex=True)
                axp.plot(sl.index, sl["close"], color="#444444", lw=1.2, label="Price")
                axp.plot(sl.index, sl["fast"], color="#4C78A8", lw=1.6, label=f"{kind} {fast}")
                axp.plot(sl.index, sl["slow"], color="#F58518", lw=1.6, label=f"{kind} {slow}")
                axp.fill_between(sl.index, 0, 1, where=sl["state"].to_numpy(), transform=axp.get_xaxis_transform(),
                                 color="#4C78A8", alpha=0.08, lw=0, label="In uptrend")
                en, ex = sl[sl["entry"]], sl[sl["exit"]]
                axp.scatter(en.index, en["close"], marker="^", s=90, color="#2e9e4f", edgecolor="white", zorder=5,
                            label="Entry")
                axp.scatter(ex.index, ex["close"], marker="v", s=90, color="#d4442f", edgecolor="white", zorder=5,
                            label="Exit")
                axp.set_ylabel("Price")
                axp.set_title(f"{pick_m}: {kind} {fast}/{slow} crossover")
                axp.legend(loc="upper left", ncol=3, fontsize=8, frameon=False)
                axp.grid(alpha=0.2)

                eq_s = (1 + bl["strategy"]).cumprod()
                eq_b = (1 + bl["buy_hold"]).cumprod()
                axe.plot(eq_b.index, eq_b, color="#999999", lw=1.4, label="Buy & hold")
                axe.plot(eq_s.index, eq_s, color="#4C78A8", lw=1.6, label="Strategy")
                axe.set_ylabel("Growth of 1")
                axe.legend(loc="upper left", fontsize=8, frameon=False)
                axe.grid(alpha=0.2)
                figm.tight_layout()
                st.pyplot(figm)
                plt.close(figm)

                ps, pb = perf_stats(bl["strategy"]), perf_stats(bl["buy_hold"])
                cmp = pd.DataFrame({"Strategy": ps, "Buy & hold": pb})
                fm = {"Sharpe (rf=0)": "{:.2f}"}
                st.dataframe(
                    cmp.style.format(lambda v: f"{v:.1%}").format(fm["Sharpe (rf=0)"], subset=pd.IndexSlice[["Sharpe (rf=0)"], :]),
                    width="stretch",
                )
                st.caption(f"Statistics for the chart window ({win}).")

                ev = sg[sg["entry"] | sg["exit"]].tail(8).iloc[::-1]
                if len(ev):
                    evt = pd.DataFrame(
                        {
                            "Date": [f"{d:%d %b %Y}" for d in ev.index],
                            "Signal": np.where(ev["entry"], "Entry", "Exit"),
                            "Price": ev["close"].to_numpy(),
                        }
                    )
                    st.markdown("**Latest crossovers**")
                    st.dataframe(evt.style.format({"Price": "{:,.2f}"}), hide_index=True, width="stretch")

            with st.expander("Method and limits"):
                st.markdown(
                    "- The signal is read at the close of each day. The strategy is invested from the next day "
                    "(no look-ahead), assuming you can trade at that close, and charges the cost on every entry and exit.\n"
                    "- Long only: invested in an uptrend, in cash otherwise (cash earns 0%). Taxes, dividends beyond the "
                    "adjusted price and slippage are not modelled.\n"
                    "- Moving-average systems trade rarely, so these backtests rest on few trades. The windows can also be "
                    "tuned to the past, so a good fit on history says little about the future.\n"
                    "- Crossovers lag: they usually avoid part of a big fall but also enter after part of the rise, and "
                    "they give false signals in sideways markets."
                )
