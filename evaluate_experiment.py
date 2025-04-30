import numpy as np
import matplotlib.pyplot as plt
from main import load_data, logistic_regression, evaluate

learning_rates = [1.0, 0.1, 0.01, 0.001, 3.0]
epochs = 100
runs_per_lr = 5

X_train, y_train = load_data("train.txt")
X_dev, y_dev = load_data("dev.txt")

def run_experiment(lr):
    train_accs_all = []
    dev_accs_all = []
    for seed in range(runs_per_lr):
        np.random.seed(seed)
        acc_train, acc_dev = [], []

        weights = np.random.randn(X_train.shape[1])
        bias = 0.0

        for epoch in range(epochs):
            z = np.dot(X_train, weights) + bias
            y_pred = 1 / (1 + np.exp(-z))
            error = y_pred - y_train
            weights -= lr * np.dot(X_train.T, error) / X_train.shape[0]
            bias -= lr * np.mean(error)

            acc_train.append(evaluate(X_train, y_train, weights, bias))
            acc_dev.append(evaluate(X_dev, y_dev, weights, bias))

        train_accs_all.append(acc_train)
        dev_accs_all.append(acc_dev)

    return np.array(train_accs_all), np.array(dev_accs_all)

# Plotting
for lr in learning_rates:
    train_accs, dev_accs = run_experiment(lr)
    mean_train = train_accs.mean(axis=0)
    mean_dev = dev_accs.mean(axis=0)
    std_train = train_accs.std(axis=0)
    std_dev = dev_accs.std(axis=0)

    epochs_range = np.arange(1, epochs + 1)

    plt.figure(figsize=(10, 5))
    plt.plot(epochs_range, mean_train, label="Training Data")
    plt.plot(epochs_range, mean_dev, label="Development Data")
    plt.fill_between(epochs_range, mean_train - std_train, mean_train + std_train, alpha=0.2)
    plt.fill_between(epochs_range, mean_dev - std_dev, mean_dev + std_dev, alpha=0.2)
    plt.title(f"Learning Rate = {lr}")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"accuracy_lr_{lr}.png")
    plt.close()
