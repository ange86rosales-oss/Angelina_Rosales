# Pipeline Base de Entrenamiento y Validación — Deep Learning

Checkpoint 1 del Proyecto Final (Data Science III). Infraestructura base
para entrenar y validar un clasificador con PyTorch, siguiendo el ciclo
completo: **forward → loss → zero_grad → backward → step**, con
tracking de métricas y validación en datos no vistos.

## Estructura del repositorio

```
.
├── data/                    # outputs del experimento (se generan al correr train.py)
│   ├── training_history.csv
│   ├── run_summary.json
│   └── loss_curve.png
├── src/
│   ├── dataset.py           # carga y preparación de datos (Iris)
│   ├── model.py              # arquitectura del clasificador (MLP)
│   ├── utils.py               # semilla y detección de dispositivo
│   └── train.py                # training loop + validation loop
├── requirements.txt
└── README.md
```

## Dataset

Se usa **Iris** (clásico, 4 features numéricas, 3 clases), cargado
directamente desde `scikit-learn` sin descargas externas, lo que
mantiene el pipeline 100% reproducible offline. Se separa en
train/validation (80/20, estratificado) y las features se estandarizan
(media 0, desvío 1) ajustando el `scaler` solo con el set de train.

## Arquitectura

`MLPClassifier`: `Linear(4 → hidden_dim) → ReLU → Linear(hidden_dim → 3)`,
implementada con `nn.Sequential`. Devuelve logits crudos, ya que la
pérdida usada (`nn.CrossEntropyLoss`) aplica `log-softmax` internamente.

## Cómo correrlo

```bash
pip install -r requirements.txt
python src/train.py --epochs 100 --lr 0.01 --hidden-dim 16
```

Al finalizar, se generan en `data/`:
- `training_history.csv`: loss y accuracy (train y validation) por época.
- `run_summary.json`: resumen de hiperparámetros y métricas finales.
- `loss_curve.png`: gráfico de la curva de pérdida (se genera aparte, ver nota abajo).

## Documentación del experimento

- **Versión de PyTorch utilizada:** `2.14.0` (CPU).
- **Dispositivo:** detectado automáticamente vía `get_device()`
  (prioridad CUDA → MPS → CPU). En esta corrida: `cpu`.
- **Learning rate elegido:** `0.01`, con el optimizador `Adam`. Se probó
  por ser un valor estándar de partida para Adam en problemas de
  clasificación pequeños; con este valor el modelo converge en pocas
  épocas sin oscilaciones.
- **Semilla:** `42`, fijada en Python, NumPy y PyTorch para reproducibilidad.

### Interpretación de la curva de pérdida

![Curva de pérdida](data/loss_curve.png)

Sí, la pérdida baja de forma clara y estable durante el entrenamiento:
arranca en ~0.91 (train) / ~0.74 (validation) en la época 1 y cae por
debajo de 0.10 hacia la época 20, estabilizándose cerca de 0.04
(train) y 0.06 (validation) hacia el final de las 100 épocas. El
accuracy de validación termina en **~96.7%**.

Las curvas de train y validation bajan juntas y quedan muy cerca una
de la otra durante todo el entrenamiento, sin que la pérdida de
validación vuelva a subir mientras la de train sigue bajando — eso
seria la señal típica de overfitting. Con este dataset pequeño y una
arquitectura simple, el modelo generaliza bien dentro del rango de
épocas usado.

## Errores comunes evitados en esta implementación

- **`zero_grad()` en cada iteración:** se llama antes de cada
  `backward()` en el training loop (`run_epoch`), evitando que los
  gradientes se acumulen entre iteraciones.
- **Consistencia de dispositivos:** tanto el modelo (`.to(device)`)
  como cada batch de datos (`X_batch.to(device)`, `y_batch.to(device)`)
  se mueven explícitamente al mismo dispositivo antes de operar.
- **Validación sin gradientes:** el loop de validación corre dentro de
  `torch.no_grad()` y con `model.eval()` (vía `model.train(mode=False)`),
  para no gastar memoria en gradientes que no se van a usar.
