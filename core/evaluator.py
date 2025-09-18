import numpy as np

class Evaluator:
    @staticmethod
    def evaluate(perceptron, X: np.ndarray, y: np.ndarray):
        correct = 0
        # PARA CADA MUESTRA, COMPARA PREDICCIÓN vs REAL
        for xi, yi in zip(X, y):
            y_pred = perceptron.predict(xi)
            if y_pred == yi:
                correct += 1
        accuracy = correct / len(y)
        return accuracy
