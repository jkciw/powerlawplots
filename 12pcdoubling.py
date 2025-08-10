import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv("coinmcap_consolidated.csv", parse_dates=["timeClose"], index_col="timeClose")
btc = df["close"].sort_index()

genesis = pd.Timestamp("2009-01-03", tz="UTC")
days_since_genesis = (btc.index - genesis).days.values
prices = btc.values

# Filter valid data
mask = (days_since_genesis > 0) & (prices > 0)
days_since_genesis = days_since_genesis[mask]
prices = prices[mask]
dates = btc.index[mask]

# Start at first available day
step_indices = []
current_day = days_since_genesis[0]
while current_day <= days_since_genesis[-1]:
    # Find the closest actual day in the data
    idx = np.abs(days_since_genesis - current_day).argmin()
    step_indices.append(idx)
    # Increase by 12.7%
    current_day *= 1.127

# Extract the actual days and prices at each 12.7% step
step_days = days_since_genesis[step_indices]
step_prices = prices[step_indices]
step_dates = dates[step_indices]


# Build ideal doubling curve for reference
ideal_prices = [step_prices[0]]
for _ in range(1, len(step_days)):
    ideal_prices.append(ideal_prices[-1] * 2)
# Plot all price data
plt.figure(figsize=(12, 7))
plt.plot(dates, prices, color='gray', alpha=0.5, label='Daily Close (all data)')
plt.plot(step_dates, step_prices, 'ro-', label='Actual (12.7% time increase)')
plt.plot(step_dates, ideal_prices, 'b--', label='Ideal Doubling Reference')

# Plot the 12.7% step points
plt.plot(step_dates, step_prices, 'ro-', label='Price at each 12.7% time increase')
for d, p in zip(step_dates, step_prices):
    plt.text(d, p, f"${p:,.0f}", fontsize=8, ha='left', va='bottom', color='red')

plt.yscale('log')
plt.title('Bitcoin Price: Doubling for Every 12.7% Increase in Network Time')
plt.xlabel('Date')
plt.ylabel('BTC Price (USD, log scale)')
plt.grid(True, which='both', ls='--', alpha=0.7)
plt.legend()
plt.tight_layout()
plt.show()
