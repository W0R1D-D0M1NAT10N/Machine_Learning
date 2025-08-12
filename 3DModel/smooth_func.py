import numpy as np
import matplotlib.pyplot as plt

def smooth_bump(x):
    result = np.zeros_like(x)
    mask = np.abs(x) < 2
    result[mask] = np.exp(-x[mask]**2 / (1 - (x[mask]/4)**2))
    return result

# Sample and plot
x_vals = np.linspace(-3, 3, 1000)
y_vals = smooth_bump(x_vals)

plt.plot(x_vals, y_vals)
plt.title("Smooth Even Bump Function with Compact Support in [-2, 2]")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.grid(True)
plt.show()

