import numpy as np
import sys

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

def load_data(file_path):
    features, labels = [], []
    with open(file_path) as f:
        for line in f:
            values = list(map(float, line.strip().split()))
            features.append(values[:-1])
            labels.append(values[-1])
    return np.array(features), np.array(labels)

def evaluate(X, y, weights, bias, out_weights=None):
    if out_weights is None:
        predictions = sigmoid(np.dot(X, weights) + bias) >= 0.5
    else:
        y_hat, _, _ = mlp_forward_pass(X, weights, bias, out_weights)
        predictions = y_hat >= 0.5
    return np.mean(predictions == y)

def logistic_regression(X, y, lr_, epochs_):
    np.random.seed(42)
    weights = np.random.randn(X.shape[1])
    bias = 0.0
    for _ in range(epochs_):
        preds = sigmoid(np.dot(X, weights) + bias)
        error = preds - y
        weights -= lr_ * np.dot(X.T, error) / len(X)
        bias -= lr_ * np.mean(error)
    return weights, bias

def mlp_initialize(input_dim, hidden_dim):
    np.random.seed(42)
    weights = np.random.randn(hidden_dim, input_dim) * 0.01
    bias = np.zeros(hidden_dim)
    out_weights = np.random.randn(hidden_dim) * 0.01
    return weights, bias, out_weights

def mlp_forward_pass(X, weights, bias, out_weights):
    z = np.dot(X, weights.T) + bias
    h = sigmoid(z)
    out = sigmoid(np.dot(h, out_weights))
    return out, h, z

def mlp_backward_pass(X, y, out, h, z, out_weights):
    delta_out = out - y
    delta_h = np.outer(delta_out, out_weights) * sigmoid_derivative(z)
    grad_out_weights = np.dot(delta_out, h) / len(X)
    grad_weights = np.dot(delta_h.T, X) / len(X)
    grad_bias = np.mean(delta_h, axis=0)
    return grad_weights, grad_bias, grad_out_weights

def mlp_bgd(X, y, lr_, epochs_, hidden_dim=10):
    weights, bias, out_weights = mlp_initialize(X.shape[1], hidden_dim)
    for _ in range(epochs_):
        out, h, z = mlp_forward_pass(X, weights, bias, out_weights)
        grad_w, grad_b, grad_v = mlp_backward_pass(X, y, out, h, z, out_weights)
        weights -= lr_ * grad_w
        bias -= lr_ * grad_b
        out_weights -= lr_ * grad_v
    return weights, bias, out_weights

def mlp_sgd(X, y, lr_, epochs_, hidden_dim=10):
    weights, bias, out_weights = mlp_initialize(X.shape[1], hidden_dim)
    for _ in range(epochs_):
        indices = np.random.permutation(len(X))
        for i in indices:
            xi = X[i:i+1]
            yi = y[i:i+1]
            out, h, z = mlp_forward_pass(xi, weights, bias, out_weights)
            grad_w, grad_b, grad_v = mlp_backward_pass(xi, yi, out, h, z, out_weights)
            weights -= lr_ * grad_w
            bias -= lr_ * grad_b
            out_weights -= lr_ * grad_v
    return weights, bias, out_weights

def print_model(weights, bias, out_weights=None):
    if out_weights is None:
        print(" ".join(map(str, list(weights) + [bias])))
    else:
        for row in weights:
            print(" ".join(map(str, row)))
        print(" ".join(map(str, bias)))
        print(" ".join(map(str, out_weights)))

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python main.py TRAIN_FILE LEARNING_RATE NUM_EPOCHS [mlp|mlp_sgd] [DEV_FILE]")
        sys.exit(1)

    train_file = sys.argv[1]
    lr = float(sys.argv[2])
    epochs = int(sys.argv[3])
    model_type = sys.argv[4] if len(sys.argv) > 4 and sys.argv[4] in {"mlp", "mlp_sgd"} else "logistic"
    dev_file = sys.argv[5] if len(sys.argv) == 6 else None

    X_train, y_train = load_data(train_file)

    if model_type == "logistic":
        weights, bias = logistic_regression(X_train, y_train, lr, epochs)
        print_model(weights, bias)
    elif model_type == "mlp":
        weights, bias, out_weights = mlp_bgd(X_train, y_train, lr, epochs)
        print_model(weights, bias, out_weights)
    elif model_type == "mlp_sgd":
        weights, bias, out_weights = mlp_sgd(X_train, y_train, lr, epochs)
        print_model(weights, bias, out_weights)

    if dev_file:
        X_dev, y_dev = load_data(dev_file)
        acc = evaluate(X_dev, y_dev, weights, bias, out_weights if model_type != "logistic" else None)
        print("Dev Accuracy:", acc)