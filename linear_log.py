import numpy as np
import matplotlib.pyplot as plt

# Generate data
x = np.linspace(1, 10, 100)
y = x**2

# Create a 1×3 grid of subplots
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# 1. Linear–Linear plot
axes[0].plot(x, y, color='teal')
axes[0].set_title('Linear–Linear Plot')
axes[0].set_xlabel('x')
axes[0].set_ylabel('y')
axes[0].grid(True)

# 2. Log–Linear plot (linear x, log y)
axes[1].plot(x, y, color='teal')
axes[1].set_yscale('log')
axes[1].set_title('Log–Linear Plot')
axes[1].set_xlabel('x')
axes[1].set_ylabel('log y')
axes[1].grid(True, which='both', linestyle='--')

# 3. Log–Log plot (log x, log y)
axes[2].plot(x, y, color='teal')
axes[2].set_xscale('log')
axes[2].set_yscale('log')
axes[2].set_title('Log–Log Plot')
axes[2].set_xlabel('log x')
axes[2].set_ylabel('log y')
axes[2].grid(True, which='both', linestyle='--')

# Overall title and layout
plt.suptitle(r'Plotting $y = x^2$ on Different Scales', fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()
