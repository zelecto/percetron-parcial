# Aplicación de Perceptrón Simple con GUI

Esta es una aplicación de escritorio educativa diseñada para demostrar el funcionamiento de un perceptrón simple, el modelo más fundamental de una neurona artificial. La interfaz gráfica de usuario (GUI) permite a los usuarios cargar datasets, configurar parámetros, entrenar el modelo y visualizar el proceso de aprendizaje en tiempo real.

## Características

La aplicación cuenta con las siguientes funcionalidades:

*   **Carga de Datasets**: Permite cargar datasets en formatos `.csv`, `.json`, y `.xlsx` a través de un explorador de archivos. También incluye una selección rápida de datasets de ejemplo.
*   **Información del Dataset**: Muestra automáticamente el número de patrones, entradas y salidas del dataset cargado.
*   **Configuración de Parámetros**: Interfaz para ajustar los hiperparámetros del perceptrón:
    *   Tasa de aprendizaje (η)
    *   Número máximo de iteraciones
    *   Error máximo permitido (ε)
    *   Umbral de activación (θ)
*   **Visualización en Tiempo Real**: Una gráfica de Matplotlib integrada muestra la evolución del error promedio del modelo en cada iteración durante el entrenamiento.
*   **Simulación y Pruebas**:
    *   Permite seleccionar un patrón del dataset cargado para ver la predicción del modelo frente al valor esperado.
    *   Permite ingresar un nuevo patrón de entrada manualmente para probar la capacidad de generalización del modelo.
*   **Logging**: Un panel de logs registra todas las acciones importantes realizadas en la aplicación, como la carga de datos, la inicialización del modelo, el estado del entrenamiento y los resultados de las simulaciones.

## Capturas de Pantalla

*(Aquí puedes añadir capturas de pantalla de la interfaz en funcionamiento)*

![Placeholder para la GUI](https://via.placeholder.com/800x600.png?text=Interfaz+de+la+Aplicación)

## Estructura del Proyecto

El código está organizado de forma modular para separar las responsabilidades:

```
perceptron_project/
├── main.py             # Punto de entrada principal que lanza la GUI.
├── core/               # Lógica principal del perceptrón.
│   ├── perceptron.py   # Clase del Perceptrón.
│   ├── trainer.py      # Lógica de entrenamiento.
│   ├── dataset_loader.py # Utilidades para cargar datos.
│   └── evaluator.py    # Lógica para evaluar la precisión.
├── ui/                 # Módulo de la interfaz gráfica.
│   └── app.py          # Código principal de la aplicación Tkinter.
├── datasets/           # Datasets de ejemplo.
├── config/             # Archivos de configuración (parámetros por defecto).
└── utils/              # Módulos de utilidad (ej. logger).
```

## Instalación

Para ejecutar este proyecto, necesitas tener Python 3 instalado. Sigue estos pasos:

1.  **Clona el repositorio:**
    ```bash
    git clone <url-del-repositorio>
    cd perceptron_project
    ```

2.  **Instala las dependencias:**
    La aplicación requiere `numpy`, `pandas`, y `matplotlib`. Puedes instalarlas usando pip:
    ```bash
    pip install numpy pandas matplotlib
    ```

## Uso

Una vez que las dependencias estén instaladas, puedes ejecutar la aplicación desde la raíz del proyecto con el siguiente comando:

```bash
python main.py
```

Esto abrirá la ventana de la interfaz gráfica, desde donde podrás interactuar con todas las funcionalidades del perceptrón.

## Conceptos Implementados

*   **Perceptrón Simple**: Implementación del modelo de neurona de McCulloch-Pitts.
*   **Regla Delta**: El algoritmo de entrenamiento se basa en la regla delta para ajustar los pesos sinápticos en función del error.
*   **Función de Activación Escalón**: Se utiliza una función de activación de tipo escalón (step function) para producir una salida binaria.
