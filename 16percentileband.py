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

# Calculate all percentiles
percentile_2_5 = np.percentile(residuals, 2.5)
percentile_16_5 = np.percentile(residuals, 16.5)
percentile_83_5 = np.percentile(residuals, 83.5)
percentile_97_5 = np.percentile(residuals, 97.5)

print(f"Percentile values:")
print(f"2.5th percentile: {percentile_2_5:.3f}")
print(f"16.5th percentile: {percentile_16_5:.3f}")
print(f"83.5th percentile: {percentile_83_5:.3f}")
print(f"97.5th percentile: {percentile_97_5:.3f}")

# 4. Calculate projection dates
target_date_2030 = pd.Timestamp("2030-12-31", tz="UTC")
target_date_2035 = pd.Timestamp("2035-12-31", tz="UTC")
days_to_2030 = (target_date_2030 - genesis).days
days_to_2035 = (target_date_2035 - genesis).days

# Create extended regression array
days_reg = np.linspace(days.min(), days_to_2035, 2000)

# Generate all regression lines extended to 2035
prices_reg = A_intercept * (days_reg ** n_slope)
prices_2_5 = A_intercept * (days_reg ** n_slope) * np.exp(percentile_2_5)
prices_16_5 = A_intercept * (days_reg ** n_slope) * np.exp(percentile_16_5)
prices_83_5 = A_intercept * (days_reg ** n_slope) * np.exp(percentile_83_5)
prices_97_5 = A_intercept * (days_reg ** n_slope) * np.exp(percentile_97_5)

# Calculate specific values for 2030
prices_2030_reg = A_intercept * (days_to_2030 ** n_slope)
prices_2030_2_5 = A_intercept * (days_to_2030 ** n_slope) * np.exp(percentile_2_5)
prices_2030_16_5 = A_intercept * (days_to_2030 ** n_slope) * np.exp(percentile_16_5)
prices_2030_83_5 = A_intercept * (days_to_2030 ** n_slope) * np.exp(percentile_83_5)
prices_2030_97_5 = A_intercept * (days_to_2030 ** n_slope) * np.exp(percentile_97_5)

# 5. Create linear-log plot
plt.figure(figsize=(14, 8))

# Plot Bitcoin price data (historical only)
plt.semilogy(days, prices, color='green', linewidth=1, alpha=0.7,
             label='Bitcoin price')

# Plot power law regression (central trend)
plt.semilogy(days_reg, prices_reg, 'k-', linewidth=3,
             label=f'Power regression (R²={r_squared:.3f})')

# Plot extreme percentile lines (2.5th and 97.5th)
plt.semilogy(days_reg, prices_2_5, color='red', linewidth=2,
             linestyle='--', label='2.5th percentile (extreme support)')
plt.semilogy(days_reg, prices_97_5, color='purple', linewidth=2,
             linestyle='--', label='97.5th percentile (extreme resistance)')

# Plot moderate percentile lines (16.5th and 83.5th)
plt.semilogy(days_reg, prices_16_5, color='orange', linewidth=2,
             linestyle=':', label='16.5th percentile (moderate support)')
plt.semilogy(days_reg, prices_83_5, color='blue', linewidth=2,
             linestyle=':', label='83.5th percentile (moderate resistance)')

# 6. Formatting
plt.xlabel('Days since Genesis (2009-01-03)')
plt.ylabel('BTC Price (USD, log scale)')
plt.title('Bitcoin Power Law Model with Multiple Confidence Bands')
plt.ylim(0.1, 1e7)
plt.grid(True, which='both', alpha=0.3)

# Format x-axis with dates (extended to 2035)
years = np.arange(2010, 2036, 2)
year_days = [(pd.Timestamp(f"{y}-01-01", tz="UTC") - genesis).days for y in years]
plt.xticks(year_days, [f"{y}" for y in years], rotation=45)
plt.xlim(days.min(), days_to_2035)

plt.legend(loc='lower right')
plt.tight_layout()
plt.show()

# 7. Print model statistics and projections for 2030 and 2035
print(f"\nPower Law Model: Price = {A_intercept:.2e} × Days^{n_slope:.3f}")
print(f"R² = {r_squared:.4f}")

print(f"\n--- 2030 PROJECTIONS ---")
print(f"Days to 2030: {days_to_2030}")
print(f"Extreme support (2.5th): ${prices_2030_2_5:,.0f}")
print(f"Moderate support (16.5th): ${prices_2030_16_5:,.0f}")
print(f"Power law trend: ${prices_2030_reg:,.0f}")
print(f"Moderate resistance (83.5th): ${prices_2030_83_5:,.0f}")
print(f"Extreme resistance (97.5th): ${prices_2030_97_5:,.0f}")

print(f"\n--- 2035 PROJECTIONS ---")
print(f"Days to 2035: {days_to_2035}")
print(f"Extreme support (2.5th): ${prices_2_5[-1]:,.0f}")
print(f"Moderate support (16.5th): ${prices_16_5[-1]:,.0f}")
print(f"Power law trend: ${prices_reg[-1]:,.0f}")
print(f"Moderate resistance (83.5th): ${prices_83_5[-1]:,.0f}")
print(f"Extreme resistance (97.5th): ${prices_97_5[-1]:,.0f}")

# 8. Print percentage of data in each band
total_points = len(residuals)
extreme_band = np.sum((residuals >= percentile_2_5) & (residuals <= percentile_97_5))
moderate_band = np.sum((residuals >= percentile_16_5) & (residuals <= percentile_83_5))

print(f"\n--- BAND COVERAGE ---")
print(f"Extreme band (2.5th-97.5th): {100*extreme_band/total_points:.1f}% of data")
print(f"Moderate band (16.5th-83.5th): {100*moderate_band/total_points:.1f}% of data")

# 9. Create summary table for key projection years
print(f"\n--- SUMMARY TABLE: KEY PROJECTION YEARS ---")
projection_years = [2030, 2035]
projection_days = [days_to_2030, days_to_2035]

print(f"{'Year':<6} {'Days':<6} {'2.5th %':<12} {'16.5th %':<12} {'Trend':<12} {'83.5th %':<12} {'97.5th %':<12}")
print("-" * 78)

for year, day_count in zip(projection_years, projection_days):
    trend = A_intercept * (day_count ** n_slope)
    support_2_5 = trend * np.exp(percentile_2_5)
    support_16_5 = trend * np.exp(percentile_16_5)
    resistance_83_5 = trend * np.exp(percentile_83_5)
    resistance_97_5 = trend * np.exp(percentile_97_5)
    
    print(f"{year:<6} {day_count:<6} ${support_2_5:>10,.0f} ${support_16_5:>10,.0f} ${trend:>10,.0f} ${resistance_83_5:>10,.0f} ${resistance_97_5:>10,.0f}")
