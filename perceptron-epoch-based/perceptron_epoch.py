import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

# Generate dataset
X, y = datasets.make_blobs(n_samples=150, n_features=2,
                          centers=2, cluster_std=1.05,
                          random_state=2)

# Step function
def step_func(z):
    return 1.0 if z > 0 else 0.0

# Perceptron algorithm
def perceptron(X, y, lr, epochs):
    m, n = X.shape
    theta = np.zeros((n + 1, 1))  # weights + bias
    miss_list = []

    for epoch in range(epochs):
        miss = 0
        for idx, x_i in enumerate(X):
            x_i = np.insert(x_i, 0, 1).reshape(-1, 1)
            y_hat = step_func(np.dot(theta.T, x_i))

            if (y_hat - y[idx]) != 0:
                theta += lr * (y[idx] - y_hat) * x_i
                miss += 1

        miss_list.append(miss)

    return theta, miss_list

# Plot decision boundary
def plot_decision_boundary(X, y, theta):
    x1 = [min(X[:, 0]), max(X[:, 0])]
    m = -theta[1] / theta[2]
    c = -theta[0] / theta[2]
    x2 = m * np.array(x1) + c

    plt.figure(figsize=(10, 8))
    plt.plot(X[:, 0][y==0], X[:, 1][y==0], "r^")
    plt.plot(X[:, 0][y==1], X[:, 1][y==1], "bs")
    plt.xlabel("feature 1")
    plt.ylabel("feature 2")
    plt.title("Perceptron Decision Boundary")
    plt.plot(x1, x2, 'y-')
    plt.show()

# Train model
theta, miss_list = perceptron(X, y, lr=0.5, epochs=100)

# Plot result
plot_decision_boundary(X, y, theta)

## Output
Decision boundary visualization:

![Output](output1.png)
