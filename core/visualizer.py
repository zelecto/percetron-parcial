import matplotlib.pyplot as plt

class Visualizer:
    @staticmethod
    def plot_errors(errors):
        plt.plot(errors, label="Error por época")
        plt.xlabel("Iteraciones")
        plt.ylabel("Error promedio")
        plt.legend()
        plt.show()
