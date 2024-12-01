import matplotlib.pyplot as plt
import numpy as np

# 1. **Basic Plotting: Line Plot**

# Generate sample data: x values and y = x^2
x = np.linspace(0, 10, 100)
y = x ** 2

# Create a basic line plot
plt.figure(figsize=(8, 6))
plt.plot(x, y, label='y = x^2', color='b')
plt.title("Basic Line Plot")
plt.xlabel("x values")
plt.ylabel("y values")
plt.legend()
plt.grid(True)
plt.show()


# 2. **Scatter Plot**

# Generate some random data
x = np.random.rand(50)
y = np.random.rand(50)

# Create a scatter plot
plt.figure(figsize=(8, 6))
plt.scatter(x, y, color='r', label='Random Data')
plt.title("Scatter Plot")
plt.xlabel("x values")
plt.ylabel("y values")
plt.legend()
plt.grid(True)
plt.show()


# 3. **Bar Plot**

# Data for bar plot
categories = ['A', 'B', 'C', 'D', 'E']
values = [5, 7, 3, 8, 4]

# Create a bar plot
plt.figure(figsize=(8, 6))
plt.bar(categories, values, color='g')
plt.title("Bar Plot")
plt.xlabel("Categories")
plt.ylabel("Values")
plt.grid(True)
plt.show()


# 4. **Histogram**

# Generate random data for histogram
data = np.random.randn(1000)

# Create a histogram
plt.figure(figsize=(8, 6))
plt.hist(data, bins=30, color='purple', edgecolor='black')
plt.title("Histogram")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.grid(True)
plt.show()


# 5. **Pie Chart**

# Data for pie chart
labels = ['A', 'B', 'C', 'D']
sizes = [15, 30, 45, 10]

# Create a pie chart
plt.figure(figsize=(8, 6))
plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=['orange', 'blue', 'green', 'red'])
plt.title("Pie Chart")
plt.show()


# 6. **Subplots**

# Create multiple subplots (2 rows, 2 columns)
x = np.linspace(0, 10, 100)
y1 = np.sin(x)
y2 = np.cos(x)
y3 = np.tan(x)

fig, axes = plt.subplots(2, 2, figsize=(10, 8))

# Plot in each subplot
axes[0, 0].plot(x, y1, 'r', label='sin(x)')
axes[0, 0].set_title("Sine Function")
axes[0, 0].legend()

axes[0, 1].plot(x, y2, 'b', label='cos(x)')
axes[0, 1].set_title("Cosine Function")
axes[0, 1].legend()

axes[1, 0].plot(x, y3, 'g', label='tan(x)')
axes[1, 0].set_ylim([-10, 10])  # Limit y-axis for tan(x) to avoid large values
axes[1, 0].set_title("Tangent Function")
axes[1, 0].legend()

axes[1, 1].axis('off')  # Turn off the 4th subplot

# Adjust layout for better spacing
plt.tight_layout()
plt.show()


# 7. **3D Plotting**

# Create a 3D plot (Surface plot)
from mpl_toolkits.mplot3d import Axes3D

x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)
x, y = np.meshgrid(x, y)
z = np.sin(np.sqrt(x**2 + y**2))

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x, y, z, cmap='viridis')

ax.set_title("3D Surface Plot")
ax.set_xlabel("X axis")
ax.set_ylabel("Y axis")
ax.set_zlabel("Z axis")

plt.show()


# 8. **Loss Curve Example (Deep Learning Visualization)**

# Simulating loss curve data for training
epochs = np.arange(1, 11)
train_loss = np.random.uniform(0.8, 1, size=10) - (epochs * 0.05)  # Simulated decreasing loss
val_loss = np.random.uniform(0.9, 1.2, size=10) - (epochs * 0.03)    # Simulated decreasing validation loss

# Plot the loss curve
plt.figure(figsize=(8, 6))
plt.plot(epochs, train_loss, label='Training Loss', color='blue', marker='o')
plt.plot(epochs, val_loss, label='Validation Loss', color='red', marker='x')
plt.title("Loss Curve During Training")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.show()

