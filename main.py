import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from matplotlib.animation import FuncAnimation

# Parameters for the dataset
n_inner = 100  # number of points in the inner group
n_outer = 100  # number of points in the outer group
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

# Create the neural network model
model = Sequential(
    [
        Dense(16, input_dim=2, activation="relu"),
        Dense(16, activation="relu"),
        Dense(1, activation="sigmoid"),
    ]
)

# Compile the model
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])


# Define a callback to store the decision boundary during training
class BoundaryCallback(tf.keras.callbacks.Callback):
    def __init__(self):
        self.weights = []

    def on_epoch_end(self, epoch, logs=None):
        xx, yy = np.meshgrid(np.linspace(-20, 20, 100), np.linspace(-20, 20, 100))
        grid = np.c_[xx.ravel(), yy.ravel()]
        preds = model.predict(grid).reshape(xx.shape)
        self.weights.append((xx, yy, preds))


boundary_callback = BoundaryCallback()

# Train the model and save the decision boundary
history = model.fit(
    data, labels, epochs=100, batch_size=16, callbacks=[boundary_callback], verbose=0
)

# Function to update the plot
fig, ax = plt.subplots(figsize=(8, 8))


def update(frame):
    ax.clear()
    ax.scatter(x_inner, y_inner, color="blue", label="Inner Group")
    ax.scatter(x_outer, y_outer, color="red", label="Outer Group")
    ax.contourf(
        boundary_callback.weights[frame][0],
        boundary_callback.weights[frame][1],
        boundary_callback.weights[frame][2],
        levels=[0, 0.5, 1],
        alpha=0.2,
        colors=["blue", "red"],
    )
    ax.set_title(f"Epoch {frame + 1}")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.legend()
    ax.grid(True)
    ax.axis("equal")


# Create the animation
ani = FuncAnimation(fig, update, frames=len(boundary_callback.weights), repeat=False)
plt.show()
