# Pipeline Base Deep Learning - PyTorch

## Alumna

Angelina Rosales

## Curso

Data Science III - NLP & Deep Learning

---

## Objetivo

Implementar un pipeline completo de entrenamiento y validación utilizando PyTorch para un problema de clasificación multiclase.

El proyecto incluye:

- Configuración automática de dispositivo (CPU, CUDA o MPS)
- Reproducibilidad mediante semillas aleatorias
- Arquitectura neuronal MLP implementada con nn.Module
- Entrenamiento utilizando PyTorch y optimizador Adam
- Validación sobre datos no vistos
- Monitoreo de métricas de desempeño
- Visualización de resultados
- Matriz de confusión

---

## Dataset

Se utilizó el dataset Iris provisto por Scikit-Learn.

Características:

- 150 observaciones
- 4 variables predictoras
- 3 clases objetivo

---

## Arquitectura del Modelo

### Multi Layer Perceptron (MLP)

Entrada:
- 4 neuronas

Capa Oculta 1:
- 16 neuronas
- ReLU

Capa Oculta 2:
- 8 neuronas
- ReLU

Salida:
- 3 neuronas

La función ReLU fue seleccionada por su eficiencia computacional y por reducir problemas asociados al desvanecimiento del gradiente.

---

## Configuración de Entrenamiento

| Parámetro | Valor |
|-----------|--------|
| Optimizer | Adam |
| Learning Rate | 0.001 |
| Epochs | 50 |
| Batch Size | 16 |
| Loss Function | CrossEntropyLoss |

---

## Métricas Utilizadas

- Training Loss
- Validation Loss
- Training Accuracy
- Validation Accuracy

Además se calculó una matriz de confusión para evaluar el desempeño final del modelo.

---

## Versión de PyTorch

La versión utilizada puede verificarse ejecutando:

```python
print(torch.__version__)
```

---

## Resultados

Durante el entrenamiento se observó:

- Disminución sostenida de la función de pérdida.
- Incremento progresivo de la accuracy.
- Comportamiento estable entre entrenamiento y validación.

No se observaron evidencias significativas de overfitting.

La mejor época se registró automáticamente utilizando la menor pérdida de validación obtenida durante el entrenamiento.

---

## Estructura del Proyecto

```text
Proyecto_NLP_Rosales/

├── data/
├── notebooks/
│   └── PipelineBase_Rosales.ipynb
├── README.md
└── requirements.txt
```

---

## Ejecución

Instalar dependencias:

```bash
pip install -r requirements.txt
```

Luego ejecutar el notebook en:

- Google Colab
- Jupyter Notebook
- VS Code

---

## Conclusiones

Se logró construir exitosamente un pipeline base de Deep Learning utilizando PyTorch.

La arquitectura implementada permitió clasificar correctamente las especies del dataset Iris y sirvió para validar el funcionamiento de:

- Forward Pass
- Backward Pass
- Optimización con Adam
- Ciclo de Validación
- Monitoreo de métricas

Este pipeline servirá como base para proyectos posteriores de NLP y Deep Learning.
