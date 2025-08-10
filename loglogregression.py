import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import matplotlib.ticker as ticker

# 1. Load consolidated CSV with daily BTC closes
df = pd.read_csv(
    "coinmcap_consolidated.csv",
    parse_dates=["timeClose"],
    index_col="timeClose"
)
btc = df["close"].sort_index()

# 2. Define Bitcoin genesis date (first block mined)
genesis_date = pd.Timestamp("2009-01-03", tz="UTC")

# 3. Calculate days since genesis for all data
days_since_genesis = (btc.index - genesis_date).days.values
mask = days_since_genesis > 0  # Remove any data before genesis
days_plot = days_since_genesis[mask]
prices_plot = btc.values[mask]

# 4. Compute extension to 2030 for projection
target_date = pd.Timestamp("2030-01-01", tz="UTC")
days_to_2030 = (target_date - genesis_date).days

# 5. Perform log-log regression
X_log = np.log(days_plot).reshape(-1, 1)
Y_log = np.log(prices_plot).reshape(-1, 1)

model = LinearRegression()
model.fit(X_log, Y_log)

# 6. Extract regression parameters
n_slope = model.coef_[0, 0]  # power-law exponent
A_intercept = np.exp(model.intercept_[0])  # scale factor
r_squared = model.score(X_log, Y_log)

# 7. Generate regression line (extend to 2030)
days_reg = np.logspace(np.log10(days_plot.min()), np.log10(days_to_2030), 100)
prices_reg = A_intercept * days_reg**n_slope

# 8. Create log-log plot with regression
plt.figure(figsize=(10, 6))

# Plot daily prices
plt.loglog(days_plot, prices_plot, ".", markersize=2, color="orange", 
           label="Daily Price")

# Plot regression line
plt.loglog(days_reg, prices_reg, "-", color="red", linewidth=2,
           label=f"Power Law: Price = {A_intercept:.2e} × Days^{n_slope:.2f}")

# Add more meaningful x-axis ticks
major_ticks = [365, 1000, 1825, 3650, 5475, 7000]  # 1y, ~3y, 5y, 10y, 15y, 19y
plt.gca().set_xticks(major_ticks, minor=False)

# Custom formatter to show years
def days_to_years(x, pos):
    years = x / 365.25
    if years < 1:
        return f"{int(x)}d"
    else:
        return f"{years:.0f}y"

plt.gca().xaxis.set_major_formatter(ticker.FuncFormatter(days_to_years))
plt.gca().xaxis.set_minor_formatter(ticker.NullFormatter())

# 9. Formatting
plt.xlim(days_plot.min(), days_to_2030)
plt.ylim(bottom=0.01)
plt.xlabel("Days since Genesis (Jan 3, 2009) - log scale")
plt.ylabel("BTC Price (USD, log scale)")
plt.title("Bitcoin Price vs Days Since Genesis (Log-Log Scale)")
plt.grid(True, which="both", linestyle="-", linewidth=0.5, alpha=0.7)
plt.legend(loc="lower right")

# 10. Add R² text box
textstr = f'$R^2 = {r_squared:.3f}$'
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
plt.gca().text(0.05, 0.95, textstr, transform=plt.gca().transAxes, 
               fontsize=12, verticalalignment='top', bbox=props)


plt.tight_layout()
plt.show()

# 11. Print regression results
print(f"Power-Law Regression Results (Days Since Genesis):")
print(f"Genesis Date: {genesis_date.date()}")
print(f"Equation: Price = {A_intercept:.2e} × Days^{n_slope:.3f}")
print(f"R² = {r_squared:.4f}")
print(f"Current days since genesis: {days_plot[-1]}")
print(f"Days to 2030: {days_to_2030}")
print(f"Projected price in 2030: ${A_intercept * days_to_2030**n_slope:,.0f}")
