import yfinance as yf
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
from scipy.stats import norm, t

st.write("# Construct the portfolio")

st.write("## Choose stocks comprising the portfolio")
ticker_list = st.multiselect(
    "Pick tickers",
    ["AAPL", "MSFT", "GOOG", "^GSPC", "AMZN", "TSLA", "META", "NVDA", "UBER", "^DJI", "^IXIC", "^OEX"],
    default=["AAPL"]
)

if len(ticker_list) == 0:
    st.stop()

if 'ticker_list' not in st.session_state:
    st.session_state.ticker_list = ticker_list

if 'data' not in st.session_state:
    st.session_state.data = yf.download(ticker_list, period="ytd", interval="1d")

new_tickers = [ticker for ticker in ticker_list if ticker not in st.session_state.ticker_list]

if len(new_tickers) > 0:
    for ticker in new_tickers:
        st.session_state.ticker_list.append(ticker)
    st.session_state.data = yf.download(ticker_list, period="ytd", interval="1d")

data = st.session_state.data

if len(ticker_list) > 1:
    st.write("## Choose portfolio weights")

ticker_weight = {}
for ticker in ticker_list:
    if len(ticker_weight) != len(ticker_list) - 1:
        ticker_weight[ticker] = st.number_input(f"Weight for {ticker}")
    else:
        ticker_weight[ticker] = 1 - sum(ticker_weight.values())

df = pd.DataFrame(sum([ticker_weight[ticker] * data['Close'][ticker] for ticker in ticker_list])).reset_index()

df['Date'] = pd.to_datetime(data.index.date)
df.columns = ['Date', 'Close']

df['Logret'] = np.log(df['Close'] / df['Close'].shift(1)).replace(to_replace=[np.inf], value=[np.nan]).values
df = df.dropna().reset_index().drop(columns=['index'])

def monte_carlo_gbm(S0, mu, sigma, T=1, n_sim=1000, n_steps=252):
    paths = np.zeros((n_sim, n_steps+1))
    paths[:, 0] = S0
    dt = T / n_steps
    for i in range(n_sim):
        dW = np.random.normal(0, np.sqrt(dt), n_steps)
        W = np.cumsum(dW)
        paths[i, 1:] = S0 * np.exp((mu - 0.5*sigma**2)*np.arange(1, n_steps+1)*dt + sigma*W)
    return paths

st.write("# Simulation Parameters")
st.write("On the basis of the chosen time frame $\\mu$ and $\\sigma$ are calculated that are then used for GBM and parametric VaR")
portf_timeseries = px.line(data_frame=df, x='Date', y='Close', title='Portfolio total value YTD')
st.plotly_chart(portf_timeseries)

t0 = pd.to_datetime(st.date_input(
    "Simulation start date",
    min_value=df['Date'].min(),
    max_value=df['Date'].max(),
    value=df['Date'].min()
))

T = pd.to_datetime(st.date_input(
    "Simulation end date",
    min_value=df['Date'].min(),
    max_value=df['Date'].max(),
    value=df['Date'].max()
))

df.set_index('Date', inplace=True)

mu = df.loc[t0:T, 'Logret'].mean() + df.loc[t0: T, 'Logret'].var() / 2
sigma = df.loc[t0:T, 'Logret'].std()

st.write(f"$\\mu$ = {mu:.2f}, $\\sigma$ = {sigma:.2f}")

total_time = (T - t0).days

gbm_paths = monte_carlo_gbm(S0=df.loc[t0, 'Close'], mu=mu, sigma=sigma, T=total_time)
time = np.linspace(0, 1, total_time)

var_days = st.number_input(
    "VaR estimation window (days)",
    min_value=1,
    max_value=total_time,
    value=7
)

var_q = st.number_input(
    "VaR percentile (in percentage points)",
    min_value=1,
    max_value=100,
    value=5
)

st.write("Historic VaR and ES")
var_historic = np.quantile(df['Logret'], q=var_q/100)
es_historic = df.loc[df['Logret'] <= var_historic, 'Logret'].mean()
st.write(f"{100 - var_q}% Historic {var_days} day(s) VaR = {-var_historic*100:.2f}%, historic ES = {-es_historic*100:.2f}%")

st.write("# Parametric VaR and ES")
z_score = norm.ppf(var_q / 100)

dfs = [4, 5, 6]
t_scores = [t.ppf(var_q / 100, df) for df in dfs]  # Output: negative value (left tail)

var_norm = mu + sigma * z_score * np.sqrt(var_days)
var_ts = [(mu + sigma * t_score * np.sqrt(var_days)) * -100 for t_score in t_scores]

norm_pdf = norm.pdf(z_score)
es_norm = mu + sigma * np.sqrt(var_days) * (norm_pdf / (var_q / 100))

t_pdfs = [t.pdf(t_score, df) for (t_score, df) in zip(t_scores, dfs)]
es_ts = [100 * (mu + sigma * np.sqrt(var_days) * (t_pdf / (var_q / 100)) * (df + t_score**2) / (df - 1))
         for (t_pdf, t_score, df) in zip(t_pdfs, t_scores, dfs)]

st.write(f"{100 - var_q}% Parametric {var_days} day(s) {100 - var_q}% VaR assuming normal distribution: {-var_norm * 100:.2f}%, ES: {es_norm * 100:.2f}%")
st.write(f"{100 - var_q}% Parametric {var_days} day(s) {100 - var_q}% VaR assuming t-distribution distribution")

var_table = pd.DataFrame({
    "Degrees of freedom": dfs,
    "VaR": var_ts,
    "ES": es_ts
}).set_index("Degrees of freedom")

config = {
    "_index": st.column_config.NumberColumn("Degrees of freedom"),
    "VaR": st.column_config.NumberColumn("VaR", format="%.2f%%"),
    "ES": st.column_config.NumberColumn("ES", format="%.2f%%"),
}

st.dataframe(var_table, column_config=config)

st.write("# MC simulation for VaR and ES")

fv_dist = ff.create_distplot(hist_data=[gbm_paths[:,-1]], group_labels=['Close prices'])
fv_dist.update_layout(
    title=dict(text="Distribution of portfolio final values")
)
st.plotly_chart(fv_dist)

nsims = 5_000

logret_dist = np.random.normal(
    loc=(mu - 0.5 * sigma**2) * var_days,
    scale=sigma * np.sqrt(var_days),
    size=nsims
)

arith_dist = np.exp(logret_dist) - 1

var = np.percentile(arith_dist, var_q)

st.write(f"{100 - var_q}% simulated {var_days} day(s) VaR is {-var * 100:.2f}%, Estimated Shortfall is {-arith_dist[arith_dist <= var].mean() * 100:.2f}%")

logret_plot = px.histogram(pd.DataFrame({"Logreturns": logret_dist}), title=f"Distribution of returns for {var_days} days")
st.plotly_chart(logret_plot)