import numpy as np
import matplotlib.pyplot as plt

# Unit-step function
def u(n):
    return (n >= 0).astype(int)

# Unit-impulse function
def delta(n):
    return (n == 0).astype(int)

def plot_signal(n, x, title):
    plt.figure(figsize=(9, 4.5))
    plt.stem(n, x, basefmt="k-")
    plt.axhline(0, linewidth=0.8)
    plt.axvline(0, linewidth=0.8)
    plt.xlabel("n")
    plt.ylabel("x[n]")
    plt.title(title)
    plt.grid(True, alpha=0.25)
    plt.xticks(n)
    plt.tight_layout()
    plt.show()

# -------------------------------------------------
# 1. Given discrete-time sequence
# -------------------------------------------------
n1 = np.arange(-3, 7)
x1 = np.array([2, 3, 2, 4, 2, 3, 2, 3, 2, 3])
plot_signal(n1, x1, "Signal 1: Given Discrete-Time Sequence")

# -------------------------------------------------
# 2. Staircase sequence
# -------------------------------------------------
n2 = np.arange(0, 5)
x2 = np.array([1, 2, 3, 2, 1])
plot_signal(n2, x2, "Signal 2: Staircase Discrete-Time Signal")

# -------------------------------------------------
# 3. Piecewise sequence
#    Interpreted at integer samples from the graph
# -------------------------------------------------
n3 = np.arange(-2, 3)
x3 = np.array([-2, -2, 0, 2, 2])
plot_signal(n3, x3, "Signal 3: Piecewise Discrete-Time Signal")

# -------------------------------------------------
# 4. x[n] = 4u[n] - u[n-3] - 5u[n-7]
# -------------------------------------------------
n4 = np.arange(-4, 11)
x4 = 4*u(n4) - u(n4 - 3) - 5*u(n4 - 7)
plot_signal(n4, x4, r"Signal 4: x[n] = 4u[n] - u[n-3] - 5u[n-7]")

# -------------------------------------------------
# 5. x[n] = delta[n] + 3delta[n-1] + 5delta[n+1]
# -------------------------------------------------
n5 = np.arange(-5, 6)
x5 = delta(n5) + 3*delta(n5 - 1) + 5*delta(n5 + 1)
plot_signal(n5, x5, r"Signal 5: x[n] = δ[n] + 3δ[n-1] + 5δ[n+1]")
