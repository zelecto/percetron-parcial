import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext
import os
import pandas as pd
import numpy as np
import datetime
from core.dataset_loader import DatasetLoader
from core.perceptron import Perceptron
from core.trainer import Trainer
from core.evaluator import Evaluator
import config.settings as settings

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Perceptron GUI")
        self.geometry("800x900") # Increased height for logs

        # Data variables
        self.X = None
        self.y = None
        self.perceptron = None

        # Main container
        main_container = ttk.Frame(self)
        main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # --- Create Frames for each section ---

        # 1. Dataset Loading
        dataset_frame = ttk.LabelFrame(main_container, text="1. Cargar Dataset")
        dataset_frame.pack(fill=tk.X, padx=5, pady=5)
        self._create_dataset_widgets(dataset_frame)

        # 2. Parameter Configuration
        params_frame = ttk.LabelFrame(main_container, text="2. Configurar Parámetros")
        params_frame.pack(fill=tk.X, padx=5, pady=5)
        self._create_params_widgets(params_frame)

        # 3. Training & Visualization
        train_frame = ttk.LabelFrame(main_container, text="3. Entrenamiento y Visualización")
        train_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self._create_training_widgets(train_frame)

        # 4. Simulation & Testing
        sim_frame = ttk.LabelFrame(main_container, text="4. Simulación y Pruebas")
        sim_frame.pack(fill=tk.X, padx=5, pady=5)
        self._create_simulation_widgets(sim_frame)

        # 5. Logs
        logs_frame = ttk.LabelFrame(main_container, text="Logs")
        logs_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self._create_logs_widgets(logs_frame)


    def _create_dataset_widgets(self, parent_frame):
        """Creates and places widgets for the dataset loading section."""
        # Frame for quick selection
        quick_select_frame = ttk.Frame(parent_frame)
        quick_select_frame.pack(fill=tk.X, padx=10, pady=5)

        ttk.Label(quick_select_frame, text="Selección rápida:").pack(side=tk.LEFT)
        self.dataset_combobox = ttk.Combobox(
            quick_select_frame,
            values=["datasets/dataset1.csv", "datasets/dataset2.csv", "datasets/dataset3.csv"]
        )
        self.dataset_combobox.pack(side=tk.LEFT, padx=5)
        self.dataset_combobox.bind("<<ComboboxSelected>>", self._on_quick_select)

        # Frame for custom file selection
        custom_select_frame = ttk.Frame(parent_frame)
        custom_select_frame.pack(fill=tk.X, padx=10, pady=5)

        ttk.Button(
            custom_select_frame,
            text="Cargar desde archivo...",
            command=self._browse_file
        ).pack(side=tk.LEFT)

        # Frame for dataset info
        info_frame = ttk.Frame(parent_frame)
        info_frame.pack(fill=tk.X, padx=10, pady=5)

        self.info_label = ttk.Label(info_frame, text="No se ha cargado ningún dataset.")
        self.info_label.pack(side=tk.LEFT)

    def _browse_file(self):
        """Opens a file dialog to select a dataset."""
        filepath = filedialog.askopenfilename(
            title="Seleccionar un dataset",
            filetypes=[("CSV files", "*.csv"), ("JSON files", "*.json"), ("Excel files", "*.xlsx *.xls")]
        )
        if filepath:
            self.load_dataset_path(filepath)

    def _on_quick_select(self, event):
        """Handles quick dataset selection from the combobox."""
        filepath = self.dataset_combobox.get()
        if filepath and os.path.exists(filepath):
            self.load_dataset_path(filepath)
        else:
            self.info_label.config(text=f"Error: No se encontró el archivo {filepath}")


    def load_dataset_path(self, filepath: str):
        """Loads a dataset from a given file path and updates the UI."""
        try:
            self.X, self.y = DatasetLoader.load_dataset(filepath)
            self._update_dataset_info()
            self._update_sim_patterns_dropdown()
            self.log(f"Dataset '{os.path.basename(filepath)}' cargado correctamente.")
        except Exception as e:
            self.log(f"Error al cargar el dataset: {e}")
            self.info_label.config(text=f"Error al cargar el dataset: {e}")

    def _update_dataset_info(self):
        """Updates the info label with details of the loaded dataset."""
        if self.X is not None and self.y is not None:
            num_patterns, num_inputs = self.X.shape
            num_outputs = self.y.shape[0] if len(self.y.shape) > 1 else 1
            info_text = f"Dataset cargado: {num_patterns} patrones, {num_inputs} entradas, {num_outputs} salida(s)."
            self.info_label.config(text=info_text)


    def _create_params_widgets(self, parent_frame):
        """Creates and places widgets for the parameter configuration section."""
        # Using a grid layout for alignment
        params_grid = ttk.Frame(parent_frame)
        params_grid.pack(fill=tk.X, padx=10, pady=5)

        # Labels
        ttk.Label(params_grid, text="Tasa de aprendizaje (η):").grid(row=0, column=0, sticky=tk.W, pady=2)
        ttk.Label(params_grid, text="Nro. max de iteraciones:").grid(row=1, column=0, sticky=tk.W, pady=2)
        ttk.Label(params_grid, text="Error máximo permitido (ε):").grid(row=2, column=0, sticky=tk.W, pady=2)
        ttk.Label(params_grid, text="Umbral (θ):").grid(row=3, column=0, sticky=tk.W, pady=2)

        # Entry fields
        self.lr_var = tk.StringVar(value=str(settings.LEARNING_RATE))
        self.iter_var = tk.StringVar(value=str(settings.MAX_ITER))
        self.error_var = tk.StringVar(value=str(settings.MAX_ERROR))
        self.threshold_var = tk.StringVar(value=str(settings.THRESHOLD))

        ttk.Entry(params_grid, textvariable=self.lr_var).grid(row=0, column=1, sticky=tk.W)
        ttk.Entry(params_grid, textvariable=self.iter_var).grid(row=1, column=1, sticky=tk.W)
        ttk.Entry(params_grid, textvariable=self.error_var).grid(row=2, column=1, sticky=tk.W)
        ttk.Entry(params_grid, textvariable=self.threshold_var).grid(row=3, column=1, sticky=tk.W)

        # Initialization button
        ttk.Button(
            params_grid,
            text="Inicializar Modelo",
            command=self._initialize_perceptron
        ).grid(row=4, column=0, columnspan=2, pady=10)

    def _initialize_perceptron(self):
        """Initializes the perceptron model based on UI parameters."""
        if self.X is None:
            self.log("Error: Cargue un dataset antes de inicializar el modelo.")
            return

        try:
            lr = float(self.lr_var.get())
            threshold = float(self.threshold_var.get())
            input_size = self.X.shape[1]

            self.perceptron = Perceptron(
                input_size=input_size,
                learning_rate=lr,
                threshold=threshold
            )
            self.log("Perceptrón inicializado correctamente.")
        except ValueError as e:
            self.log(f"Error en los parámetros: {e}")
        except Exception as e:
            self.log(f"Error inesperado: {e}")


    def _create_training_widgets(self, parent_frame):
        """Creates and places widgets for the training and visualization section."""
        # Matplotlib Figure and Canvas
        fig = Figure(figsize=(5, 3), dpi=100)
        self.ax = fig.add_subplot(111)
        self.ax.set_title("Error por Iteración")
        self.ax.set_xlabel("Iteraciones")
        self.ax.set_ylabel("Error Promedio")

        self.canvas = FigureCanvasTkAgg(fig, master=parent_frame)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.canvas.draw()

        # Training controls frame
        control_frame = ttk.Frame(parent_frame)
        control_frame.pack(fill=tk.X, pady=5)

        ttk.Button(
            control_frame,
            text="Entrenar",
            command=self._start_training
        ).pack(side=tk.LEFT, padx=10)

        self.status_label = ttk.Label(control_frame, text="Estado: Esperando para entrenar.")
        self.status_label.pack(side=tk.LEFT)

    def _start_training(self):
        """Handles the training process and updates the plot."""
        if self.X is None or self.y is None or self.perceptron is None:
            msg = "Error: Cargar dataset e inicializar modelo primero."
            self.status_label.config(text=msg)
            self.log(msg)
            return

        try:
            max_iter = int(self.iter_var.get())
            max_error = float(self.error_var.get())
        except ValueError:
            msg = "Error: Parámetros de iteración/error inválidos."
            self.status_label.config(text=msg)
            self.log(msg)
            return

        # Clear previous plot
        self.ax.clear()
        self.ax.set_title("Error por Iteración")
        self.ax.set_xlabel("Iteraciones")
        self.ax.set_ylabel("Error Promedio")
        errors = []

        msg = "Iniciando entrenamiento..."
        self.status_label.config(text=msg)
        self.log(msg)
        self.update_idletasks() # Force UI update

        # Re-implementation of Trainer.fit to allow real-time plotting
        for epoch in range(max_iter):
            total_error = 0
            for xi, yi in zip(self.X, self.y):
                # We use the existing update_weights method which returns the error
                total_error += abs(self.perceptron.update_weights(xi, yi))

            avg_error = total_error / len(self.y)
            errors.append(avg_error)

            # Update plot
            self.ax.plot(errors, color='b')
            self.canvas.draw()
            self.update_idletasks() # Allow UI to process events

            if avg_error <= max_error:
                msg = f"Entrenamiento finalizado: Error objetivo alcanzado en la iteración {epoch + 1}."
                self.status_label.config(text=msg)
                self.log(msg)
                self._calculate_accuracy()
                return # Stop training

        msg = f"Entrenamiento finalizado: Máximo de iteraciones ({max_iter}) alcanzado."
        self.status_label.config(text=msg)
        self.log(msg)
        self._calculate_accuracy()


    def _create_simulation_widgets(self, parent_frame):
        """Creates widgets for the simulation and testing section."""
        sim_grid = ttk.Frame(parent_frame)
        sim_grid.pack(fill=tk.X, padx=10, pady=5)

        # Test pattern from dataset
        ttk.Label(sim_grid, text="Probar patrón del dataset:").grid(row=0, column=0, sticky=tk.W)
        self.sim_pattern_combobox = ttk.Combobox(sim_grid, width=40)
        self.sim_pattern_combobox.grid(row=0, column=1, pady=2, sticky=tk.W)
        ttk.Button(sim_grid, text="Probar", command=self._test_selected_pattern).grid(row=0, column=2, padx=5)

        # Test manual pattern
        ttk.Label(sim_grid, text="Probar patrón manual (e.g., 1,0,1):").grid(row=1, column=0, sticky=tk.W)
        self.manual_pattern_var = tk.StringVar()
        ttk.Entry(sim_grid, textvariable=self.manual_pattern_var, width=42).grid(row=1, column=1, pady=2, sticky=tk.W)
        ttk.Button(sim_grid, text="Probar", command=self._test_manual_pattern).grid(row=1, column=2, padx=5)

        # Result display
        self.sim_result_label = ttk.Label(sim_grid, text="Resultado:")
        self.sim_result_label.grid(row=2, column=0, columnspan=3, sticky=tk.W, pady=5)

        # Accuracy display
        self.accuracy_label = ttk.Label(sim_grid, text="Precisión en dataset de entrenamiento: --")
        self.accuracy_label.grid(row=3, column=0, columnspan=3, sticky=tk.W, pady=5)

    def _update_sim_patterns_dropdown(self):
        """Populates the simulation dropdown with patterns from the loaded dataset."""
        if self.X is None:
            self.sim_pattern_combobox['values'] = []
            return

        patterns_str = [f"Entradas: {row.tolist()}, Salida esperada: {self.y[i]}" for i, row in enumerate(self.X)]
        self.sim_pattern_combobox['values'] = patterns_str
        if patterns_str:
            self.sim_pattern_combobox.current(0)

    def _test_selected_pattern(self):
        """Tests the pattern selected in the combobox."""
        if self.perceptron is None:
            msg = "Resultado: El modelo no ha sido entrenado."
            self.sim_result_label.config(text=msg)
            self.log("Intento de simulación sin modelo entrenado.")
            return

        try:
            selected_index = self.sim_pattern_combobox.current()
            if selected_index < 0:
                return

            pattern = self.X[selected_index]
            expected_y = self.y[selected_index]
            predicted_y = self.perceptron.predict(pattern)

            result_text = f"Resultado: Para la entrada {pattern.tolist()}, la predicción es {predicted_y} (esperada: {expected_y})."
            self.sim_result_label.config(text=result_text)
            self.log(f"Simulación de patrón conocido: {result_text}")
        except Exception as e:
            msg = f"Error en la simulación: {e}"
            self.sim_result_label.config(text=msg)
            self.log(msg)

    def _test_manual_pattern(self):
        """Tests a manually entered pattern."""
        if self.perceptron is None:
            msg = "Resultado: El modelo no ha sido entrenado."
            self.sim_result_label.config(text=msg)
            self.log("Intento de simulación sin modelo entrenado.")
            return

        try:
            pattern_str = self.manual_pattern_var.get()
            # Convert comma-separated string to numpy array of floats/ints
            pattern = np.array([float(x.strip()) for x in pattern_str.split(',')])

            if len(pattern) != self.X.shape[1]:
                msg = f"Error: El patrón debe tener {self.X.shape[1]} entradas."
                self.sim_result_label.config(text=msg)
                self.log(f"Error de simulación manual: {msg}")
                return

            predicted_y = self.perceptron.predict(pattern)
            result_text = f"Resultado: Para la entrada manual {pattern.tolist()}, la predicción es {predicted_y}."
            self.sim_result_label.config(text=result_text)
            self.log(f"Simulación de patrón manual: {result_text}")
        except Exception as e:
            msg = f"Error: Patrón manual inválido. Use números separados por comas. ({e})"
            self.sim_result_label.config(text=msg)
            self.log(msg)

    def _calculate_accuracy(self):
        """Calculates and displays the model's accuracy on the training set."""
        if self.perceptron and self.X is not None and self.y is not None:
            acc = Evaluator.evaluate(self.perceptron, self.X, self.y)
            self.accuracy_label.config(text=f"Precisión en dataset de entrenamiento: {acc*100:.2f}%")


    def _create_logs_widgets(self, parent_frame):
        """Creates the text area for logging."""
        self.log_text = scrolledtext.ScrolledText(parent_frame, state='disabled', height=5, wrap=tk.WORD)
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.log("Bienvenido a la GUI del Perceptrón.")

    def log(self, message: str):
        """Adds a message to the log text area."""
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        full_message = f"[{timestamp}] {message}\n"

        self.log_text.config(state='normal')
        self.log_text.insert(tk.END, full_message)
        self.log_text.config(state='disabled')
        self.log_text.see(tk.END) # Scroll to the end


if __name__ == "__main__":
    app = App()
    app.mainloop()
