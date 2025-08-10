import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# 1. Load data and compute days since genesis
df = pd.read_csv("coinmcap_consolidated.csv", 
                parse_dates=["timeClose"], 
                index_col="timeClose")
btc = df["close"].sort_index()

genesis = pd.Timestamp("2009-01-03", tz="UTC")
days = (btc.index - genesis).days.values
prices = btc.values

# Filter valid data
mask = (days > 0) & (prices > 0)
days = days[mask]
prices = prices[mask]

# 2. Fit POWER LAW regression (log-log space)
X_log = np.log(days).reshape(-1, 1)
Y_log = np.log(prices).reshape(-1, 1)
power_model = LinearRegression().fit(X_log, Y_log)

# Extract power law parameters: Price = A × Days^n
n_slope = power_model.coef_[0, 0]
A_intercept = np.exp(power_model.intercept_[0])
r_squared = power_model.score(X_log, Y_log)

# 3. Calculate residuals and percentiles
predicted_log_prices = power_model.predict(X_log).flatten()
residuals = Y_log.flatten() - predicted_log_prices

# Calculate percentiles of residuals
percentile_2_5 = np.percentile(residuals, 2.5)
percentile_97_5 = np.percentile(residuals, 97.5)

# 4. Generate regression lines for plotting (EXTENDED TO 2035)
target_date_2035 = pd.Timestamp("2035-12-31", tz="UTC")
days_to_2035 = (target_date_2035 - genesis).days

# Create extended regression array
days_reg = np.linspace(days.min(), days_to_2035, 2000)  # More points for smoother lines

# Generate all regression lines extended to 2035
prices_reg = A_intercept * (days_reg ** n_slope)
prices_2_5 = A_intercept * (days_reg ** n_slope) * np.exp(percentile_2_5)
prices_97_5 = A_intercept * (days_reg ** n_slope) * np.exp(percentile_97_5)

# 5. Create linear-log plot
plt.figure(figsize=(14, 8))

# Plot Bitcoin price data (historical only)
plt.semilogy(days, prices, color='green', linewidth=1, alpha=0.7, 
             label='Bitcoin price')

# Plot power law regression (extended to 2035)
plt.semilogy(days_reg, prices_reg, 'k-', linewidth=2, 
             label=f'Power regression (R²={r_squared:.3f})')

# Plot 2.5th percentile line (support) - extended to 2035
plt.semilogy(days_reg, prices_2_5, color='red', linewidth=2, 
             linestyle='--', label='2.5th percentile (support)')

# Plot 97.5th percentile line (resistance) - extended to 2035
plt.semilogy(days_reg, prices_97_5, color='purple', linewidth=2, 
             linestyle='--', label='97.5th percentile (resistance)')

# 6. Formatting
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.title('Bitcoin Power Law Model with Projections to 2035')
plt.ylim(0.1, 1e7)  # Extended Y-axis to accommodate 2035 projections
plt.grid(True, which='both', alpha=0.3)

# Format x-axis with dates (extended to 2035)
years = np.arange(2010, 2036, 2)  # Extended to 2035
year_days = [(pd.Timestamp(f"{y}-01-01", tz="UTC") - genesis).days for y in years]
plt.xticks(year_days, [f"{y}" for y in years], rotation=45)
plt.xlim(days.min(), days_to_2035)

plt.legend()
plt.tight_layout()
plt.show()

# Print model statistics and 2035 projections
print(f"\nPower Law Model: Price = {A_intercept:.2e} × Days^{n_slope:.3f}")
print(f"R² = {r_squared:.4f}")
print(f"\n--- 2035 PROJECTIONS ---")
print(f"Days to 2035: {days_to_2035}")
print(f"Power law trend (2035): ${prices_reg[-1]:,.0f}")
print(f"Support level (2035): ${prices_2_5[-1]:,.0f}")
print(f"Resistance level (2035): ${prices_97_5[-1]:,.0f}")
print(f"Support to Resistance range: ${prices_2_5[-1]:,.0f} - ${prices_97_5[-1]:,.0f}")
