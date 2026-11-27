import numpy as np
import matplotlib.pyplot as plt



def x1(t):
    return (np.sin(t)**2)/4

def x2(t):
    return -(np.sin(t)**2)/4


t=np.linspace(-10,0,1000)

t = np.linspace(-10, 10, 1000)
plt.figure(figsize=(8, 5))
plt.plot(t, x1(t), label="x1")
plt.plot(t, x2(t), label="x2")
plt.axhline(0, color="black", linewidth=1, linestyle="--", label="Trivial: 0")
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()