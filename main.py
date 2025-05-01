import numpy as np
import sys

def sigmoid(x):
    return 1 / (1 + np.exp(-x))
def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)
def load_data(file_path):
    features = []
    labels = []
    with open(file_path) as file:
        for line in file:
            values = list(map(float, line.strip().split()))
            features.append(values[:-1])
            labels.append(values[-1])
    return np.array(features), np.array(labels)
def evaluate(X, y, weights, bias, weight_vector = None):
    if weight_vector is None: predictions = sigmoid(np.dot(X, weights) + bias) >= 0.5
    else:
        y_hat, activation, preactivation = mlp_forward_pass(X, weights, bias, weight_vector)
        predictions = y_hat >= 0.5
    return np.mean(predictions == y)
def logistic_regression(X, y, learning_rate, epochs):
    np.random.seed(0)
    weights = np.random.randn(X.shape[1])
    bias = 0.0
    for epoch in range(epochs):
        y_hat = sigmoid(np.dot(X, weights) + bias)
        error = y_hat - y
        for j in range(len(weights)):
            weights[j] -= learning_rate * sum((y_hat[i] - y[i]) * X[i][j] for i in range(len(X))) / len(X)
        bias -= learning_rate * np.mean(error)
    return weights, bias
def mlp_initialize(input_dimension, hidden_dimension):
    np.random.seed(0)
    weights = np.random.randn(hidden_dimension, input_dimension) * 0.01
    bias = np.zeros(hidden_dimension)
    out_weights = np.random.randn(hidden_dimension) * 0.01
    return weights, bias, out_weights
def mlp_forward_pass(X, weights, bias, output_weights):
    z = np.dot(X, weights.T) + bias
    h = sigmoid(z)
    y_hat = sigmoid(np.dot(h, output_weights))
    return y_hat, h, z
def mlp_backward_pass(X, y, y_hat, h, z, out_weights):
    error = y_hat - y
    hidden_error = np.outer(error, out_weights) * sigmoid_derivative(z)
    gradient_output_weights = np.dot(error, h) / len(X)
    gradient_weights = np.dot(hidden_error.T, X) / len(X)
    gradient_bias = np.mean(hidden_error, axis = 0)
    return gradient_weights, gradient_bias, gradient_output_weights
def mlp_bgd(X, y, learning_rate, epochs, hidden_dimensions = 10):
    weights, bias, out_weights = mlp_initialize(X.shape[1], hidden_dimensions)
    for epoch in range(epochs):
        y_hat, h, z = mlp_forward_pass(X, weights, bias, out_weights)
        gradient_weights, gradient_bias, gradient_output_weights = mlp_backward_pass(X, y, y_hat, h, z, out_weights)
        weights -= learning_rate * gradient_weights
        bias -= learning_rate * gradient_bias
        out_weights -= learning_rate * gradient_output_weights
    return weights, bias, out_weights
def mlp_sgd(X, y, learning_rate, epochs, hidden_dimensions = 10):
    weights, bias, out_weights = mlp_initialize(X.shape[1], hidden_dimensions)
    for epoch in range(epochs):
        indices = np.random.permutation(len(X))
        for index in indices:
            x_i = X[index:index+1]
            y_i = y[index:index+1]
            y_hat, h, z = mlp_forward_pass(x_i, weights, bias, out_weights)
            gradient_weights, gradient_bias, gradient_output_weights = mlp_backward_pass(x_i, y_i, y_hat, h, z, out_weights)
            weights -= learning_rate * gradient_weights
            bias -= learning_rate * gradient_bias
            out_weights -= learning_rate * gradient_output_weights
    return weights, bias, out_weights
def print_model(weights, bias, out_weights = None):
    if out_weights is None: print(" ".join(map(str, list(weights) + [bias])))
    else:
        for row in weights:
            print(" ".join(map(str, row)))
        print(" ".join(map(str, bias)))
        print(" ".join(map(str, out_weights)))

train_file = sys.argv[1]
learning_rate = float(sys.argv[2])
epochs = int(sys.argv[3])
model_type = sys.argv[4] if len(sys.argv) > 4 and sys.argv[4] in {"mlp", "mlp_sgd"} else "logistic"
dev_file = sys.argv[5] if len(sys.argv) > 5 else None
X_train, y_train = load_data(train_file)

if model_type == "logistic":
    weights, bias = logistic_regression(X_train, y_train, learning_rate, epochs)
    print_model(weights, bias)
elif model_type == "mlp":
    weights, bias, out_weights = mlp_bgd(X_train, y_train, learning_rate, epochs)
    print_model(weights, bias, out_weights)
elif model_type == "mlp_sgd":
    weights, bias, out_weights = mlp_sgd(X_train, y_train, learning_rate, epochs)
    print_model(weights, bias, out_weights)