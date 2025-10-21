 # Entrega proyecto final - Modelado Predictivo - Un Enfoque de Machine Learning en Python

## 📘 Abstract

Objetivo de Investigación: Este estudio desarrolla un framework predictivo integral para analizar los determinantes resultados de partidos internacionales de handball femenino utilizando un dataset que contiene información sobre equipos, fechas, sedes, torneos y resultados numéricos cuyo objetivo principal es explorar patrones de rendimiento entre equipos, identificar tendencias en los resultados y evaluar la competitividad de los encuentros. Para ello nos propusimos varios pasos que detallaremos precedentemente.


## ❓ Preguntas e hipótesis de interés

- ¿Qué equipos tienen mayor cantidad de victorias en torneos específicos?
- ¿Cuál fue el ranking de goles por país entre 2020 y 2023?
- ¿Existe una relación entre la diferencia de goles y el tipo de torneo?
- ¿Hay equipos que consistentemente ganan por márgenes amplios?

## 🎯 Objetivo del Proyecto

El objetivo de este proyecto es realizar un análisis predictivo del rendimiento de los equipos en competiciones internacionales de handball femenino en el periodo de 2020 y 2023, utilizando visualizaciones en Python para identificar patrones, tendencias y relaciones entre variables clave. A través del procesamiento del dataset Handball_W_InternationalResults.csv, se busca responder preguntas relevantes sobre la competitividad de los equipos, la evolución de los resultados a lo largo del tiempo, y las características de las victorias en distintos torneos. 

## 📊 Visualizaciones y análisis

Se han generado visualizaciones que incluyen:

- Gráficos de barras con ranking de goles totales por equipo entre 2020 y 2023.

```python
# Calcular goles totales por equipo
total_goals = goals_by_team.groupby("Team")["Goals"].sum().reset_index()

# Ordenar por goles descendente
total_goals = total_goals.sort_values(by="Goals", ascending=False)

# Crear gráfico de barras
fig = px.bar(
    total_goals,
    x="Team",
    y="Goals",
    title="Ranking de Goles Totales por Equipo 2020-2023",
    labels={"Team": "Equipo", "Goals": "Goles Totales"}


fig.show()
```


- Gráficos de cajas con el top 1o de diferencia de goles por equipo.

```python
# Limpiar los nombres de las columnas
df.columns = df.columns.str.strip()

# Determinar el equipo ganador por partido
df["WinningTeam"] = df.apply(
    lambda row: row["TeamA"] if row["ScoreA"] > row["ScoreB"]
    else row["TeamB"] if row["ScoreB"] > row["ScoreA"]
    else "Draw", axis=1
)

# Calcular la diferencia de goles
df["GoalDifference"] = abs(df["ScoreA"] - df["ScoreB"])

# Filtrar partidos ganados (excluir empates)
df_wins = df[df["WinningTeam"] != "Draw"]

# Calcular el número de victorias por equipo
victory_counts = df_wins["WinningTeam"].value_counts().nlargest(10).index.tolist()

# Filtrar los partidos ganados por los 10 equipos con más victorias
df_top_wins = df_wins[df_wins["WinningTeam"].isin(victory_counts)]

# Crear gráfico de cajas
fig = px.box(
    df_top_wins,
    x="WinningTeam",
    y="GoalDifference",
    title="Distribución de la Diferencia de Goles por Equipo (Top 10 en Victorias)",
    labels={"WinningTeam": "Equipo Ganador", "GoalDifference": "Diferencia de Goles"}
)

fig.show()
```

- Visualizaciones multivariadas que combinan equipo, goles y tipo de victoria.

```python
# Limpiar los nombres de las columnas
df.columns = df.columns.str.strip()

# Determinar el equipo ganador por partido
df["WinningTeam"] = df.apply(
    lambda row: row["TeamA"] if row["ScoreA"] > row["ScoreB"]
    else row["TeamB"] if row["ScoreB"] > row["ScoreA"]
    else "Draw", axis=1
)

# Filtrar partidos con ganador (excluir empates)
df_wins = df[df["WinningTeam"] != "Draw"]

# Contar victorias por equipo y torneo
victory_counts = df_wins.groupby(["TournamentName", "WinningTeam"]).size().reset_index(name="Victories")

# Crear gráfico tipo treemap
fig = px.treemap(
    victory_counts,
    path=["TournamentName", "WinningTeam"],
    values="Victories",
    title="Cantidad de Victorias por Equipo según Torneo"
)

fig.show()
```
- Visualizaciones en un treemap.

```python
import pandas as pd
import plotly.express as px

# Cargar el archivo CSV
df = pd.read_csv (r"/content/drive/MyDrive/Handball_W_InternationalResults.csv")

# Limpiar los nombres de las columnas
df.columns = df.columns.str.strip()

# Determinar el equipo ganador por partido
df["WinningTeam"] = df.apply(
    lambda row: row["TeamA"] if row["ScoreA"] > row["ScoreB"]
    else row["TeamB"] if row["ScoreB"] > row["ScoreA"]
    else "Draw", axis=1
)

# Filtrar partidos con ganador (excluir empates)
df_wins = df[df["WinningTeam"] != "Draw"]

# Contar victorias por equipo y torneo
victory_counts = df_wins.groupby(["TournamentName", "WinningTeam"]).size().reset_index(name="Victories")

# Crear gráfico tipo treemap
fig = px.treemap(
    victory_counts,
    path=["TournamentName", "WinningTeam"],
    values="Victories",
    title="Cantidad de Victorias por Equipo según Torneo"
)

fig.show()
```



Además, se han creado nuevas columnas en el dataset:

- `Resultado Partido`: nombre del equipo ganador o empate.
- `Diferencia de Goles`: diferencia absoluta entre los goles anotados por cada equipo.
- `Resultado de la cantidad de victorias por equipo`: Calcular el número de victorias por equip



1. Definición del topico a chequear
• Objetivo claro: Predecir qué equipo (TeamA o TeamB) será el ganador (WinningTeam) de un partido de balonmano internacional femenino.
• Métricas de Éxito: Dado que es un problema de clasificación, las métricas clave son la Precisión (Accuracy), la Sensibilidad (Recall), el F1-Score, o el AUC-ROC.

3. Preparación de Datos
El dataset Handball_W_InternationalResults_with_Winner.csv contiene columnas como Date, TeamA, TeamB, ScoreA, ScoreB, Sex, TournamentName, year, Venue y WinningTeam, por lo que se procedió a realizar una limpieza de Datos con lo que el Data Cleaning es esencial para remover nulos, outliers e inconsistencias.
Para la carga de datos se utilizó la librería pandas (importada como pd) para cargar el archivo, dado que Pandas es fundamental para el manejo de datos en Python.
Para los valores Faltantes (NaN)se realizó una limpieza de datos.
Para los outliers: Se utilizo el Z-score para eliminar outliers si superan un umbral de 3 desviaciones estándar.

4. Transformación de datos
La transformación de datos es necesaria para que las variables categóricas o numéricas sean adecuadas para el modelado

En cuanto a Variables Categóricas (Codificación) como TeamA, TeamB, TournamentName y Venue se convirtieron a formato numérico, ya que los modelos de ML generalmente requieren entradas numéricas.


    
    ◦ Codificación One Hot Encoding: Recomendada para variables con pocas categorías. Se podría usar para Venue o TournamentName1:
    ◦ Codificación LabelEncoder/OrdinalEncoder: Podría usarse para las variables de equipo o si se considera que existe un orden inherente2022.
2. Normalización/Estandarización de Variables Numéricas: Las variables numéricas, como ScoreA y ScoreB, deben ser escaladas, especialmente si el algoritmo elegido es sensible a la magnitud de los datos (como SVM o Regresión Lineal)2....
    ◦ La Estandarización (Z-Score Scaling) es recomendada para algoritmos que asumen distribución gaussiana (ej. SVM, Regresión Lineal) y cuando existen outliers2.
    ◦ La Normalización (Min-Max Scaling) es más adecuada si los datos tienen límites conocidos (ej. imágenes RGB23) o para algoritmos como Redes Neuronales o K-NN224.




  ###   Missing values
No se encontraron valores perdidos en ninguna de las columnas:

 - Date              0
 - TeamA             0
 - TeamB             0
 - ScoreA            0
 - ScoreB            0
 - Sex               0
 - TournamentName    0
 - year              0
 - Venue             0


## 🛠️ Herramientas utilizadas

- Python
- Pandas
- NumPy
- Plotly
- Google Colab / Jupyter Notebook

## 📁 Estructura del proyecto

- `Rosales_Internationalresults_handball.ipynb`: notebook principal con análisis y visualizaciones https://colab.research.google.com/drive/1HCWJG1xg51Bk8JSu1iLII-78TraJrQ1t?usp=sharing 
- `README.md`: descripción del proyecto.
- `Handball_W_InternationalResults.csv`: dataset utilizado.

