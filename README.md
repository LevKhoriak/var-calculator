# Equity Portfolio VaR and ES Calculator
This is a simple VaR and ES calculator that allows the user to pick some equities and indices from Yahoo Finance and see how VaR changes:
- When the securities weights in portfolio change
- When the simulation time frame is changed
- When VaR percentile and time window is changed

[Live demo](https://levkhoriak-var-calculator.streamlit.app/) is available on Streamlit Community Cloud
# Methods Employed
The three main methods of VaR and ES estimation are employed in the project
## Historic method
Historic distribution of log-returns of the portfolio are constructed. Then a given quantile is sampled from the distribution to estimate VaR and the average log-return of this tail is calculated for ES
## Parametric method
Historic $\mu$ and $\sigma$ are estimated from realized log-returns, t-score and z-score are taken from corresponding Student t-distributions and normal distributions and then VaR is calculated as $\text{VaR} = \mu + \sigma \cdot \text{z-score (or t-score)}$
## Monte-Carlo simulation method
The portfolio value is simulated using Geometric Brownian Motion and then the distribution of simulated log-returns is used to sample VaR and ES
![Simulated portfolio path](https://github.com/user-attachments/assets/80d94a7f-3fd1-4bd6-a0e3-5884bf90b84e)
![Simulated portfolio final values](https://github.com/user-attachments/assets/21d9a26a-695e-4843-a3b8-43f281440749)
![Simulated log-returns](https://github.com/user-attachments/assets/3bdac69d-acef-4c9b-beac-6c226bbd9bf6)
