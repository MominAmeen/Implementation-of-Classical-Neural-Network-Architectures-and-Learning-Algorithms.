import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification

# Generate dataset
X, y = make_classification(n_samples=100, n_features=2, n_informative=1,
                           n_redundant=0, n_classes=2,
                           n_clusters_per_class=1,
                           random_state=41, class_sep=10)

# Plot dataset
plt.figure(figsize=(10,6))
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='winter', s=100)
plt.title("Dataset")
plt.show()


# Step function
def step(z):
    return 1 if z > 0 else 0


# Stochastic Perceptron
def perceptron(X, y):
    X = np.insert(X, 0, 1, axis=1)  # add bias
    weights = np.ones(X.shape[1])
    lr = 0.1

    for i in range(1000):
        j = np.random.randint(0, 100)  # random sample
        y_hat = step(np.dot(X[j], weights))
        weights = weights + lr * (y[j] - y_hat) * X[j]

    return weights[0], weights[1:]


# Train model
intercept_, coef_ = perceptron(X, y)

print("Weights:", coef_)
print("Bias:", intercept_)


# Decision boundary
m = -(coef_[0] / coef_[1])
b = -(intercept_ / coef_[1])

x_input = np.linspace(-3, 3, 100)
y_input = m * x_input + b

# Plot decision boundary
plt.figure(figsize=(10,6))
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='winter', s=100)
plt.plot(x_input, y_input, color='red', linewidth=3)
plt.title("Stochastic Perceptron Decision Boundary")
plt.ylim(-3, 3)
plt.show()
