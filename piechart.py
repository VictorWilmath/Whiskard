import matplotlib.pyplot as plt
import numpy as np

x = np.array([41, 37, 22])
labels = ["Beans-41%", "Pips-37%", "Sammy-22%"]
colors = ["Gray", "Orange", "Black"]
plt.pie(x, labels=labels, colors = colors)
plt.title("Data by each cat")
plt.show()