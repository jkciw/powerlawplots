import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# 1. Generate synthetic observed data following a power law y = A * x^n with noise
np.random.seed(0)
x = np.linspace(1, 10, 50)
A_true, n_true = 2.0, 1.8
y_clean = A_true * x**n_true
noise = np.random.normal(scale=0.2 * y_clean)  # 20% noise
y_obs = y_clean + noise

# 2. Prepare data for log–log regression
X_log = np.log(x).reshape(-1, 1)
Y_log = np.log(y_obs.clip(min=1e-3)).reshape(-1, 1)  # avoid log(0)

# 3. Fit linear model on log data
model = LinearRegression()
model.fit(X_log, Y_log)

# 4. Extract estimated parameters
n_est     = model.coef_[0, 0]
A_est     = np.exp(model.intercept_[0])
r_squared = model.score(X_log, Y_log)

# 5. Generate fitted curve
y_fit = A_est * x**n_est

# 6. Plot observed data, fitted model, and R²
plt.figure(figsize=(8, 6))
plt.scatter(x, y_obs, color='blue', label='Observed Data')
plt.plot(x, y_fit, color='red',
         label=f'Fitted Model: y = {A_est:.2f}·x^{n_est:.2f}\n$R^2$ = {r_squared:.3f}')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Power-Law Fit to Observed Data')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
