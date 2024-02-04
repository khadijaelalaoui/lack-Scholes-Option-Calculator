import streamlit as st
import numpy as np
import pandas as pd
from scipy.stats import norm
import time
import matplotlib.pyplot as plt

def black_scholes(option_type, spot_price, strike_price, time_to_maturity, volatility, interest_rate):
    if strike_price == 0:
        st.warning('Strike Price cannot be zero.')
        return None

    d1 = (np.log(spot_price / strike_price) + (interest_rate + 0.5 * volatility**2) * time_to_maturity) / (volatility * np.sqrt(time_to_maturity))
    d2 = d1 - volatility * np.sqrt(time_to_maturity)

    if option_type == 'Call':
        option_price = spot_price * norm.cdf(d1) - strike_price * np.exp(-interest_rate * time_to_maturity) * norm.cdf(d2)
    elif option_type == 'Put':
        option_price = strike_price * np.exp(-interest_rate * time_to_maturity) * norm.cdf(-d2) - spot_price * norm.cdf(-d1)
    else:
        option_price = None

    return option_price

def monte_carlo_simulation(option_type, spot_price, strike_price, time_to_maturity, volatility, interest_rate, num_simulations):
    simulation_results = []

    for _ in range(num_simulations):
        spot_prices = [spot_price]
        for _ in range(int(time_to_maturity)):
            spot_price_simulation = spot_prices[-1] * np.exp((interest_rate - 0.5 * volatility**2) +
                                                             volatility * np.sqrt(1) * np.random.normal())
            spot_prices.append(spot_price_simulation)
        simulation_results.append(spot_prices)

    return simulation_results

# Streamlit UI
st.title('📊 Black-Scholes Option Calculator')

option_type = st.selectbox('Select Option Type:', ['Call', 'Put'])
spot_price = st.number_input('Spot Price:', min_value=0.0)
strike_price = st.number_input('Strike Price:', min_value=0.0)
time_to_maturity = st.number_input('Time to Maturity:', min_value=0.0)
volatility = st.number_input('Volatility:', min_value=0.0)
interest_rate = st.number_input('Interest Rate:', min_value=0.0)

# Calculate option price
option_price = black_scholes(option_type, spot_price, strike_price, time_to_maturity, volatility, interest_rate)

# Display result
if option_price is not None:
    st.header('💰 Option Price Calculation')
    st.write(f'**{option_type} Option Price:** {option_price:.4f}')
else:
    st.warning('Please select a valid option type.')

# Monte Carlo Simulation with multiple paths in a single chart
st.header('📈 Monte Carlo Simulation')

# Number of simulations for Monte Carlo simulation
num_simulations = st.number_input('Number of Simulations:', min_value=1, value=5)

# Simulate multiple paths and plot
simulation_results = monte_carlo_simulation(option_type, spot_price, strike_price, time_to_maturity, volatility, interest_rate, num_simulations)

# Display simulation results in a table
st.subheader('📊 Simulation Results')
simulation_df = pd.DataFrame(simulation_results).transpose()
st.write(simulation_df)

# Plotting multiple paths in a single chart
fig, ax = plt.subplots()
for path in simulation_results:
    ax.plot(path)

ax.set_title('Multiple Simulated Paths')
ax.set_xlabel('Time Steps')
ax.set_ylabel('Spot Price')
st.pyplot(fig)

# Adding your name at the end with a larger font size and a separating line
st.markdown("***")
st.markdown('<p style="font-size:20px;">🖊 Réalisé par Khadija El Alaoui</p>', unsafe_allow_html=True)
