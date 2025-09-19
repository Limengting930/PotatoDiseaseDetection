import matplotlib.pyplot as plt

def plot_history(history, metric="accuracy"):
    plt.plot(history.history[metric], label=metric)
    plt.plot(history.history["val_" + metric], label="val_" + metric)
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.title(f"Training and Validation {metric}")
    plt.legend()
    plt.show()
