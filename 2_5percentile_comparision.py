import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# 1. Load the data
df = pd.read_csv("coinmcap_consolidated.csv",
                 parse_dates=["timeClose"],
                 index_col="timeClose")
btc = df["close"].sort_index()

# 2. Calculate days since genesis
genesis = pd.Timestamp("2009-01-03", tz="UTC")
days = (btc.index - genesis).days.values
prices = btc.values

# Filter valid data
mask = (days > 0) & (prices > 0)
days = days[mask]
prices = prices[mask]
dates = btc.index[mask]

# 3. Fit power law regression in log-log space
X_log = np.log(days).reshape(-1, 1)
Y_log = np.log(prices).reshape(-1, 1)

model = LinearRegression()
model.fit(X_log, Y_log)

# Extract power law parameters
n_slope = model.coef_[0, 0]
A_intercept = np.exp(model.intercept_[0])
r_squared = model.score(X_log, Y_log)

print(f"Power Law Model: Price = {A_intercept:.2e} × Days^{n_slope:.3f}")
print(f"R² = {r_squared:.4f}")

# 4. Calculate residuals and percentiles
predicted_log_prices = model.predict(X_log).flatten()
residuals = Y_log.flatten() - predicted_log_prices

# Calculate 2.5th percentile of residuals
percentile_2_5 = np.percentile(residuals, 2.5)
print(f"2.5th percentile residual: {percentile_2_5:.3f}")

# 5. Calculate 2.5th percentile support line for all data points
support_line = A_intercept * (days ** n_slope) * np.exp(percentile_2_5)

# 6. Check for breaches below 2.5th percentile line
breaches = prices < support_line
breach_dates = dates[breaches]
breach_prices = prices[breaches]
breach_support_values = support_line[breaches]

# 7. Display results
print(f"\n=== BREACH ANALYSIS ===")
print(f"Total data points: {len(prices)}")
print(f"Points below 2.5th percentile line: {np.sum(breaches)}")
print(f"Percentage of total: {100 * np.sum(breaches) / len(prices):.2f}%")

if np.sum(breaches) > 0:
    print(f"\n=== DATES BELOW 2.5TH PERCENTILE SUPPORT ===")
    for i, (date, price, support) in enumerate(zip(breach_dates, breach_prices, breach_support_values)):
        deviation = 100 * (price - support) / support
        print(
            f"{i+1:2d}. {date.date()}: ${price:7.2f} (support: ${support:7.2f}, {deviation:+5.1f}%)")

    # Find the most extreme breach
    deviations = 100 * \
        (breach_prices - breach_support_values) / breach_support_values
    most_extreme_idx = np.argmin(deviations)

    print(f"\n=== MOST EXTREME BREACH ===")
    print(f"Date: {breach_dates[most_extreme_idx].date()}")
    print(f"Price: ${breach_prices[most_extreme_idx]:.2f}")
    print(f"Support: ${breach_support_values[most_extreme_idx]:.2f}")
    print(f"Deviation: {deviations[most_extreme_idx]:+.1f}%")

    # Analyze by year
    print(f"\n=== BREACHES BY YEAR ===")
    breach_years = [date.year for date in breach_dates]
    unique_years = sorted(set(breach_years))

    for year in unique_years:
        year_breaches = sum(1 for y in breach_years if y == year)
        print(f"{year}: {year_breaches} breaches")

else:
    print("No daily closes fell below the 2.5th percentile support line!")

# 8. Create visualization
plot_viz = input("\nCreate visualization? (y/n): ").lower() == 'y'

if plot_viz:
    # Sample data for cleaner visualization (every 30th point)
    sample_idx = np.arange(0, len(days), 30)

    plt.figure(figsize=(12, 8))

    # Plot price data (sampled)
    plt.semilogy(days[sample_idx], prices[sample_idx], '.',
                 markersize=3, color='green', alpha=0.7, label='Bitcoin Price (sampled)')

    # Plot regression line
    days_line = np.linspace(days.min(), days.max(), 1000)
    prices_line = A_intercept * (days_line ** n_slope)
    plt.semilogy(days_line, prices_line, 'k-',
                 linewidth=2, label='Power Law Trend')

    # Plot 2.5th percentile support line
    support_line_full = A_intercept * \
        (days_line ** n_slope) * np.exp(percentile_2_5)
    plt.semilogy(days_line, support_line_full, 'r--',
                 linewidth=2, label='2.5th Percentile Support')

    # Highlight breach points
    if np.sum(breaches) > 0:
        breach_days = days[breaches]
        plt.semilogy(breach_days, breach_prices, 'ro', markersize=5,
                     label=f'Breaches ({np.sum(breaches)} points)')

    plt.xlabel('Days since Genesis (2009-01-03)')
    plt.ylabel('BTC Price (USD, log scale)')
    plt.title('Bitcoin Price vs 2.5th Percentile Support Line')
    plt.legend()
    plt.grid(True, which='both', alpha=0.3)
    plt.tight_layout()
    plt.show()

print("\nAnalysis complete!")
