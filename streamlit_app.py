import yfinance as yf
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff

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

st.write(f"mu = {mu:.2f}, sigma = {sigma:.2f}")

total_time = (T - t0).days

gbm_paths = monte_carlo_gbm(S0=df.loc[t0, 'Close'], mu=mu, sigma=sigma, T=total_time)
time = np.linspace(0, 1, total_time)

st.write("# Simulation results")

fv_dist = ff.create_distplot(hist_data=[gbm_paths[:,-1]], group_labels=['Close prices'])
fv_dist.update_layout(
    title=dict(text="Distribution of portfolio final values")
)
st.plotly_chart(fv_dist)

nsims = 5_000

var_days = st.number_input(
    "VaR estimation window (days)",
    min_value=1,
    max_value=total_time,
    value=7
)

logret_dist = np.random.normal(
    loc=(mu - 0.5 * sigma**2) * var_days,
    scale=sigma * np.sqrt(var_days),
    size=nsims
)

arith_dist = np.exp(logret_dist) - 1

var_q = st.number_input(
    "VaR percentile (in percentage points)",
    min_value=1,
    max_value=100,
    value=5
)

var = np.percentile(arith_dist, var_q)

st.write(f"{100 - var_q}% VaR is {var * 100:.2f}%, Estimated Shortfall is {arith_dist[arith_dist <= var].mean() * 100:.2f}%")

logret_plot = px.histogram(pd.DataFrame({"Logreturns": logret_dist}), title=f"Distribution of returns for {var_days} days")
st.plotly_chart(logret_plot)