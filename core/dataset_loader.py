import pandas as pd

class DatasetLoader:
    @staticmethod
    def load_dataset(file_path: str):
        """Carga dataset CSV, JSON o Excel y separa X, y"""
        if file_path.endswith(".csv"):
            df = pd.read_csv(file_path)
        elif file_path.endswith(".json"):
            df = pd.read_json(file_path)
        elif file_path.endswith(".xls") or file_path.endswith(".xlsx"):
            df = pd.read_excel(file_path)
        else:
            raise ValueError("Formato de archivo no soportado")

        X = df.iloc[:, :-1].values  # Todas las columnas menos la última
        y = df.iloc[:, -1].values   # Última columna como target
        return X, y
