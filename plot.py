import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("gp_output.csv")

plt.figure(figsize=(10, 5))
plt.plot(df.x, df["mean"], color="blue", label="GP mean")
plt.fill_between(df.x, df.lower, df.upper, alpha=0.2, color="blue", label="±2σ")

# plot the training points
train_x = [0.1, 0.3, 0.5, 0.7, 0.9]
train_y = [np.sin(2*np.pi*x) for x in train_x]
plt.scatter(train_x, train_y, color="red", zorder=5, label="Training data")

# plot true function
x = np.linspace(0, 1, 200)
plt.plot(x, np.sin(2*np.pi*x), "--", color="green", alpha=0.7, label="True function")

plt.legend()
plt.savefig("gp_plot.png")
plt.show()