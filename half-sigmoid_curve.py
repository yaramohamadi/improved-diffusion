import numpy as np
import matplotlib.pyplot as plt

def half_sigmoid_curve(span, k=10, c=0.5):
    """
    Generate the second half of a sigmoid curve, normalized to start at 0 and end at 1.

    Parameters:
        span (int): Number of points in the curve.
        k (float): Steepness parameter. Larger values mean faster ascent toward 1.
        c (float): Center point where the sigmoid starts rising.

    Returns:
        np.array: A curve starting at 0 and ending at 1.
    """
    x = np.linspace(c, 1, span)  # Take only the second half (x from c to 1)
    y = 1 / (1 + np.exp(-k * (x - c)))  # Sigmoid function
    y_start = 1 / (1 + np.exp(-k * (c - c)))  # Sigmoid value at x = c
    y_end = 1 / (1 + np.exp(-k * (1 - c)))  # Sigmoid value at x = 1
    y_normalized = (y - y_start) / (y_end - y_start)  # Normalize to [0, 1]
    return y_normalized

# Demonstrating different curves
span = 200
curves = {
    "(k=1000)": half_sigmoid_curve(span, k=1000),
    "(k=250)": half_sigmoid_curve(span, k=250),
    "(k=100)": half_sigmoid_curve(span, k=100),
    "(k=50)": half_sigmoid_curve(span, k=50),
    "(k=25)": half_sigmoid_curve(span, k=25),
    "(k=10)": half_sigmoid_curve(span, k=10),
    "(k=1)": half_sigmoid_curve(span, k=1),
}

# Plotting the curves
plt.figure(figsize=(10, 6))
for label, curve in curves.items():
    plt.plot(np.linspace(0, span, span), curve, label=label)
plt.title("Second Half of Sigmoid (Start at 0, End at 1)")
plt.xlabel("Normalized Input (x)")
plt.ylabel("Output (y)")
plt.legend()
plt.grid()
plt.savefig('/home/ens/AT74470/datasets/tmpfig.png')

print(half_sigmoid_curve(span, k=250))