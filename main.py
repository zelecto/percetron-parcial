import numpy as np
from core.perceptron import Perceptron
from core.trainer import Trainer
from core.dataset_loader import DatasetLoader
from core.evaluator import Evaluator
from utils.logger import Logger
import config.settings as settings

if __name__ == "__main__":
    Logger.log("Cargando dataset...")
    X, y = DatasetLoader.load_dataset("datasets/dataset1.csv")

    Logger.log("Inicializando perceptrón...")
    p = Perceptron(
        input_size=X.shape[1],
        learning_rate=settings.LEARNING_RATE,
        threshold=settings.THRESHOLD
    )

    Logger.log("Entrenando perceptrón...")
    trainer = Trainer(p, max_iter=settings.MAX_ITER, max_error=settings.MAX_ERROR)
    errors = trainer.fit(X, y, show_plot=True)

    Logger.log("Evaluando modelo...")
    acc = Evaluator.evaluate(p, X, y)
    Logger.log(f"Precisión en dataset: {acc*100:.2f}%")

    Logger.log("Prueba con nuevo patrón [1, 1, 1]:")
    print("Predicción:", p.predict(np.array([1, 1, 1])))
