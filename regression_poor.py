import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# 1. Generate synthetic observed data following y = A * x^n
np.random.seed(0)
x = np.linspace(1, 10, 50)
A_true, n_true = 2.0, 1.8
y_clean = A_true * x**n_true

# 2. Add noise scaled to produce R² ≈ 0.025
noise = np.random.normal(scale=0.6 * y_clean)  # 60% noise
y_obs = y_clean + noise

# 3. Prepare data for log–log regression
X_log = np.log(x).reshape(-1, 1)
Y_log = np.log(np.clip(y_obs, a_min=1e-3, a_max=None)).reshape(-1, 1)

# 4. Fit linear model on log data
model = LinearRegression()
model.fit(X_log, Y_log)
n_est     = model.coef_[0, 0]
A_est     = np.exp(model.intercept_[0])
r_squared = model.score(X_log, Y_log)

# 5. Generate fitted curve
y_fit = A_est * x**n_est

# 6. Plot observed data, fitted model, and R²
plt.figure(figsize=(8, 6))
plt.scatter(x, y_obs, color='blue', label='Observed Data')
plt.plot(
    x, y_fit, color='red',
    label=f'Fitted Model: y = {A_est:.2f}·x^{n_est:.2f}\n$R^2$ = {r_squared:.3f}'
)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Power-Law Fit with $R^2\\approx 0.025$')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
