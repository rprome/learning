import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def load_data(file_path):
    features, labels = [], []
    with open(file_path) as file:
        for line in file:
            values = list(map(float, line.strip().split()))
            features.append(values[:-1])
            labels.append(values[-1])
    return np.array(features), np.array(labels)

def evaluate(X, y, weights, bias):
    predictions = sigmoid(np.dot(X, weights) + bias) >= 0.5
    return np.mean(predictions == y)

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
        weights = np.random.randn(X_train.shape[1])
        bias = 0.0
        acc_train, acc_dev = [], []

        for _ in range(epochs):
            z = np.dot(X_train, weights) + bias
            y_pred = sigmoid(z)
            error = y_pred - y_train
            weights -= lr * np.dot(X_train.T, error) / len(X_train)
            bias -= lr * np.mean(error)

            acc_train.append(evaluate(X_train, y_train, weights, bias))
            acc_dev.append(evaluate(X_dev, y_dev, weights, bias))

        train_accs_all.append(acc_train)
        dev_accs_all.append(acc_dev)

    return np.array(train_accs_all), np.array(dev_accs_all)

# Main execution
if __name__ == "__main__":
    for learning_rate in learning_rates:
        train_accs, dev_accs = run_experiment(learning_rate)

        mean_train = train_accs.mean(axis=0)
        mean_dev = dev_accs.mean(axis=0)
        min_train = train_accs.min(axis=0)
        max_train = train_accs.max(axis=0)
        min_dev = dev_accs.min(axis=0)
        max_dev = dev_accs.max(axis=0)

        epochs_range = np.arange(1, epochs + 1)

        plt.figure(figsize=(12, 6))
        plt.plot(epochs_range, mean_train, label="TS Mean Accuracy", color="#00205B")

        plt.plot(epochs_range, mean_dev, label="DS Mean Accuracy", color="#FFC70A")

        summary_text = \
        (
            f"Training Set (TS) — Mean Accuracy: {mean_train[-1]:.3f}, Minimum: {min_train[-1]:.3f}, Maximum: {max_train[-1]:.3f}\n"
            f"Development Set (DS) — Mean Accuracy: {mean_dev[-1]:.3f}, Minimum: {min_dev[-1]:.3f}, Maximum: {max_dev[-1]:.3f}"
        )
        plt.gca().text(0.5, 0.02, summary_text, transform=plt.gca().transAxes,
                      fontsize=10, ha='center', bbox=dict(facecolor='white', edgecolor='black', boxstyle='round'))

        plt.title(f"Mean Accuracy over Epochs (Learning Rate = {learning_rate})")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        filename = f"annotated_accuracy_lr_{learning_rate}.png"
        plt.savefig(filename)
        plt.close()
