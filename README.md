# Regresión Lineal Simple con TensorFlow y Flask

Este proyecto implementa un ejercicio de regresión lineal simple utilizando TensorFlow con las siguientes características:

- Backend en Flask para servir la aplicación web
- Entrenamiento de un modelo de regresión lineal con TensorFlow
- Predicción de múltiples valores a la vez
- Visualización de la pérdida durante el entrenamiento
- Gráfico interactivo de los datos y la línea de regresión

## Requisitos

Para ejecutar este proyecto necesitarás:

```
Flask
TensorFlow
NumPy
```

Puedes instalar todas las dependencias con:

```bash
pip install flask tensorflow numpy
```

## Instrucciones de uso

1. Ejecuta el servidor Flask:

```bash
python app.py
```

2. Abre tu navegador en `http://localhost:5000`

3. La interfaz te permitirá:
   - Ingresar datos de entrenamiento (pares x,y)
   - Configurar el número de épocas para entrenar
   - Entrenar el modelo
   - Ver la línea de regresión y el historial de pérdida
   - Predecir múltiples valores a la vez

## Estructura del proyecto

- `app.py`: Servidor Flask y lógica del backend
- `templates/index.html`: Interfaz de usuario con gráficos interactivos

## Detalles de implementación

### Backend (Flask + TensorFlow)

El servidor Flask proporciona varias rutas:

- `/`: Muestra la interfaz de usuario
- `/train`: Recibe los datos de entrenamiento y entrena el modelo
- `/predict`: Recibe valores y devuelve predicciones
- `/loss_history`: Devuelve el historial de pérdida durante el entrenamiento

El modelo de regresión lineal se implementa utilizando TensorFlow con una sola capa densa (Dense) y se entrena utilizando el optimizador SGD (Descenso de Gradiente Estocástico) con la función de pérdida MSE (Error Cuadrático Medio).

### Frontend (HTML + JavaScript + Chart.js)

La interfaz de usuario permite:

- Ingresar datos de entrenamiento en formato texto
- Visualizar los datos de entrenamiento como puntos en un gráfico
- Visualizar la línea de regresión
- Ver el historial de pérdida en un gráfico
- Predecir múltiples valores y visualizar los resultados

Se utiliza Chart.js para crear visualizaciones interactivas de los datos y el historial de pérdida.
