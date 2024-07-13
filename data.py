import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Parameters for the dataset
n_inner = 1000  # number of points in the inner group
n_outer = 1000  # number of points in the outer group
r1 = 6  # radius of the inner circle
r2 = 5  # inner radius of the outer ring
r3 = 15  # outer radius of the outer ring

# Generate inner group points
angles_inner = np.random.uniform(0, 2 * np.pi, n_inner)
radii_inner = np.random.uniform(0, r1, n_inner)
x_inner = radii_inner * np.cos(angles_inner)
y_inner = radii_inner * np.sin(angles_inner)

# Generate outer group points
angles_outer = np.random.uniform(0, 2 * np.pi, n_outer)
radii_outer = np.random.uniform(r2, r3, n_outer)
x_outer = radii_outer * np.cos(angles_outer)
y_outer = radii_outer * np.sin(angles_outer)

# Combine the datasets
x = np.concatenate((x_inner, x_outer))
y = np.concatenate((y_inner, y_outer))

# Labels: 0 for inner group, 1 for outer group
labels_inner = np.zeros(n_inner)
labels_outer = np.ones(n_outer)
labels = np.concatenate((labels_inner, labels_outer))

# Prepare the data for training
data = np.column_stack((x, y))

fig, ax = plt.subplots(figsize=(8, 8))
ax.scatter(x_inner, y_inner, color="blue", label="Inner Group")
ax.scatter(x_outer, y_outer, color="red", label="Outer Group")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.legend()
ax.grid(True)
ax.axis("equal")
plt.show()
