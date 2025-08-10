import pandas as pd
import numpy as np

# Load your data
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

# Build the 12.7% step table
step_indices = []
current_day = days_since_genesis[0]
while current_day <= days_since_genesis[-1]:
    idx = np.abs(days_since_genesis - current_day).argmin()
    step_indices.append(idx)
    current_day *= 1.127

step_days = days_since_genesis[step_indices]
step_prices = prices[step_indices]
step_dates = dates[step_indices]

# Build the table
table = pd.DataFrame({
    "Date": step_dates,
    "Days Since Genesis": step_days,
    "BTC Price (USD)": step_prices
})

# Add price ratio to previous step
table["Price Ratio to Previous"] = table["BTC Price (USD)"].pct_change().add(1).round(2)
table["Price Ratio to Previous"] = table["Price Ratio to Previous"].replace(np.nan, "-")

# Print the table
print(table.to_string(index=False))
