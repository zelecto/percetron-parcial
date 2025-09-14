import numpy as np
import matplotlib.pyplot as plt
from typing import List

class Trainer:
    def __init__(self, perceptron, max_iter=100, max_error=0.01):
        self.perceptron = perceptron
        self.max_iter = max_iter
        self.max_error = max_error
        self.errors: List[float] = []

    def fit(self, X: np.ndarray, y: np.ndarray, show_plot: bool = True):
        for epoch in range(self.max_iter):
            total_error = 0
            for xi, yi in zip(X, y):
                total_error += abs(self.perceptron.update_weights(xi, yi))

            avg_error = total_error / len(y)
            self.errors.append(avg_error)

            if show_plot:
                plt.cla()
                plt.plot(self.errors, label="Error por época")
                plt.xlabel("Iteraciones")
                plt.ylabel("Error promedio")
                plt.legend()
                plt.pause(0.05)

            if avg_error <= self.max_error:
                break

        if show_plot:
            plt.show()

        return self.errors
