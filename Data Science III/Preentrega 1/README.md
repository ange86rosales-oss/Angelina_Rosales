# Pipeline Base Deep Learning - PyTorch

## Alumna
Angelina Rosales

## Objetivo
Implementar un pipeline completo de entrenamiento y validación utilizando PyTorch para un problema de clasificación.

## Dataset
Iris Dataset (Scikit-Learn)

## Arquitectura
MLP con dos capas ocultas:

- Linear(4,16)
- ReLU
- Linear(16,8)
- ReLU
- Linear(8,3)

## Optimizador

Adam

Learning Rate: 0.001

## Función de pérdida

CrossEntropyLoss

## Métrica

Accuracy

## Resultados

La pérdida disminuye progresivamente durante las épocas de entrenamiento, mostrando que la red aprende patrones útiles del conjunto de datos.

## Reproducibilidad

Se fija una semilla aleatoria (42) para garantizar resultados reproducibles.
