import numpy as np

class Perceptron:
    def __init__(self, input_size: int, learning_rate: float = 0.1, threshold: float = 0.0):
        self.weights = np.random.uniform(-1, 1, input_size)
        self.bias = np.random.uniform(-1, 1)
        self.learning_rate = learning_rate
        self.threshold = threshold

    def activation(self, x: float) -> int:
        """Función de activación escalón"""
        return 1 if x >= self.threshold else 0

    def predict(self, inputs: np.ndarray) -> int:
        """Predice la salida para una entrada"""
        summation = np.dot(inputs, self.weights) + self.bias
        return self.activation(summation)

    def update_weights(self, x: np.ndarray, y_true: int):
        """Ajusta los pesos con la Regla Delta"""
        y_pred = self.predict(x)
        error = y_true - y_pred
        self.weights += self.learning_rate * error * x
        self.bias += self.learning_rate * error
        return error
