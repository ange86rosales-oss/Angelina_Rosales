 # Modelado Predictivo de la performance de las selecciones internacionales de handball femenino: Un Enfoque de Machine Learning para analizar la predictibilidad de las próximas selecciones triunfadoras.

## 📘 Contexto

Desde el año 2010, el handball femenino mundial ha vivido una transformación marcada por la intensidad competitiva, el surgimiento de nuevas potencias y la consolidación de selecciones históricas. Europa mantuvo su hegemonía, pero también hubo sorpresas que rompieron el molde.
Noruega, con su estilo veloz y técnico, se consagró como una de las selecciones más dominantes, alzando el trofeo en varias ocasiones y manteniéndose siempre cerca del podio. Francia, por su parte, fue construyendo una generación dorada que alcanzó la gloria tanto en mundiales como en los Juegos Olímpicos, donde logró el oro en Tokio 2020. Países Bajos también dejó su huella, conquistando el mundo en 2019 con una actuación memorable en Japón.
Pero no todo fue Europa. En 2013, Brasil sorprendió al mundo entero al coronarse campeona en su propia tierra, demostrando que el talento sudamericano podía competir al más alto nivel. Esa victoria fue histórica, no solo por el título, sino por lo que representó para el desarrollo del deporte en América Latina.
En los Juegos Olímpicos, el handball femenino también vivió momentos intensos. Rusia, compitiendo bajo la bandera del Comité Olímpico Ruso, se llevó el oro en Río 2016, mientras que Noruega y Francia se mantuvieron como protagonistas constantes. La edición de Tokio, celebrada en 2021 por la pandemia, fue testigo de la consagración francesa, que venció a Rusia en una final cargada de emoción.
A lo largo de estos años, el deporte se volvió más global. Cuba, por ejemplo, logró en 2025 una clasificación histórica al Mundial tras ganar el campeonato regional de América del Norte y el Caribe, mostrando que el crecimiento del handball femenino no se limita a Europa.
Cada torneo, cada medalla, cada partido disputado en estos quince años ha sido parte de una narrativa que habla de esfuerzo, evolución y pasión. El handball femenino mundial se ha convertido en un espectáculo de alto nivel, donde la técnica, la táctica y el corazón se combinan para ofrecer historias inolvidables. 


## 🎯 Objetivo e hipótesis del proyecto

Este estudio desarrolla un framework predictivo integral para analizar los proximos resultados de partidos internacionales de handball femenino, para el cual analizaremos los partidos disputados desde 2010 al 2023 en los que se han disputado más de 2800 partidos oficiales lo que muestra una actividad constante y creciente lo que nos permite ver cómo el handball femenino ha crecido en volumen, diversidad y competitividad. Europa sigue siendo el núcleo, pero otras regiones están ganando terreno tanto en  en volumen, diversidad y competitividad. 
Dicho todo lo anterior, analizaremos las tendencias observadas en los torneos más importantes

En este sentido, analizaremos con datos respaldatorios, las tendencias observadas en los torneos más importantes a partir del 2024 y veremos si Europa seguirá  manteniendo su supremacia hegemónica.


## ❓ Preguntas de interés

- ¿Qué equipos tienen el mejor promedio de victorias en torneos específicos?
- ¿Cuál fue el ranking de goles por país entre 2020 y 2023?
- ¿Existe una relación entre la diferencia de goles y el tipo de torneo?
- ¿Hay equipos que consistentemente ganan por márgenes amplios?


## 📊 Visualizaciones y análisis

 - Análisis exploratorio (EDA) de variables clave, informacion general, filas, columnas, valores nulos, variables numericas, estadisticas clave.

```python
   import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff

# Cargar dataset
df = pd.read_csv("Handball_W_InternationalResults_with_Winner.txt")

# Información general
print(df.info())
print(df.describe())

# Valores nulos
print(df.isnull().sum())

# Histogramas
px.histogram(df, x="ScoreA", nbins=20, title="Distribución ScoreA").show()
px.histogram(df, x="ScoreB", nbins=20, title="Distribución ScoreB").show()
px.histogram(df, x="year", nbins=20, title="Distribución Year").show()

# Boxplot
px.box(df.melt(value_vars=['ScoreA','ScoreB'], var_name='Equipo', value_name='Goles'),
       x='Equipo', y='Goles', title='Boxplot ScoreA vs ScoreB').show()

# Conteo categorías
print(df['TournamentName'].value_counts().head(10))
print(df['WinningTeam'].value_counts().head(10))

# Heatmap correlación
corr_matrix = df[['ScoreA','ScoreB']].corr()
fig = ff.create_annotated_heatmap(z=corr_matrix.values,
                                  x=corr_matrix.columns.tolist(),
                                  y=corr_matrix.columns.tolist(),
                                  annotation_text=corr_matrix.round(2).values)
fig.show()

```


Se han generado visualizaciones que incluyen:

- ¿Qué equipos tienen el mejor promedio de victorias en torneos específicos entre 2020 y 2023 con su respectivo top 15 con mejor promedio de victorias por torneo

```python
import pandas as pd

# Cargar el dataset
file_path = "Handball_W_InternationalResults_with_Winner.txt"
df = pd.read_csv(file_path)

# Filtrar empates
df = df[df['WinningTeam'] != 'Draw']

# Agrupar por torneo y equipo ganador
grouped = df.groupby(['TournamentName', 'WinningTeam']).size().reset_index(name='Wins')

# Total de partidos por torneo
total_matches = df.groupby('TournamentName').size().reset_index(name='TotalMatches')

# Calcular tasa de victorias
data = pd.merge(grouped, total_matches, on='TournamentName')
data['WinRate'] = data['Wins'] / data['TotalMatches']

# Obtener el mejor equipo por torneo
best_teams = data.sort_values(['TournamentName', 'WinRate'], ascending=[True, False])
best_per_tournament = best_teams.groupby('TournamentName').first().reset_index()

print(best_per_tournament[['TournamentName', 'WinningTeam', 'WinRate']])

TOP 15
import pandas as pd
import plotly.express as px
import plotly.io as pio

# Cargar el archivo
file_path = (r"/content/drive/MyDrive/Handball_W_InternationalResults.csv")
df = pd.read_csv(file_path)

# Limpiar los nombres de las columnas
df.columns = df.columns.str.strip()

# Determinar el equipo ganador por partido
df["WinningTeam"] = df.apply(
    lambda row: row["TeamA"] if row["ScoreA"] > row["ScoreB"]
    else row["TeamB"] if row["ScoreB"] > row["ScoreA"]
    else "Draw", axis=1
)

# Filtrar partidos con ganador
df = df[df['WinningTeam'] != 'Draw']

# --- 1) Promedio de victorias por torneo ---
wins_per_tournament = df.groupby(['TournamentName', 'WinningTeam']).size().reset_index(name='Wins')
total_matches_per_tournament = df.groupby('TournamentName').size().reset_index(name='TotalMatches')

wins_per_tournament = wins_per_tournament.merge(total_matches_per_tournament, on='TournamentName')
wins_per_tournament['WinRate'] = wins_per_tournament['Wins'] / wins_per_tournament['TotalMatches']

wins_sorted = wins_per_tournament.sort_values('WinRate', ascending=False).head(15)

fig1 = px.bar(wins_sorted, x='WinningTeam', y='WinRate', color='TournamentName',
              title='Top 15 equipos con mejor promedio de victorias por torneo',
              labels={'WinningTeam': 'Equipo', 'WinRate': 'Promedio de victorias'})
fig1.write_html('top_winrate_per_tournament.html')

# --- 2) Ranking global ---
global_wins = df.groupby('WinningTeam').size().reset_index(name='TotalWins')
global_wins_sorted = global_wins.sort_values('TotalWins', ascending=False).head(20)

fig2 = px.bar(global_wins_sorted, x='WinningTeam', y='TotalWins',
              title='Ranking global de equipos más dominantes (total de victorias)',
              labels={'WinningTeam': 'Equipo', 'TotalWins': 'Total de victorias'})
fig2.write_html('global_dominant_teams.html')

```


- Eliminación de duplicados y outliers.
```python
import pandas as pd

# Cargar dataset
df = pd.read_csv("Handball_W_InternationalResults (1).csv")

# Eliminar duplicados
df = df.drop_duplicates()

# Eliminar outliers usando IQR
for col in ['ScoreA', 'ScoreB']:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]

# Guardar dataset limpio
df.to_csv("Handball_W_InternationalResults_clean.csv", index=False)

```



-  Imputacion por medio de simple imputerss, valores de moda y otros por valores 0

```python
import pandas as pd
from sklearn.impute import SimpleImputer

# Cargar dataset
df = pd.read_csv(r"/content/drive/MyDrive/Handball_W_InternationalResults.csv")

# Identificar columnas
numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
categorical_cols = df.select_dtypes(include=['object']).columns

# Imputación numérica (media)
imputer_mean = SimpleImputer(strategy='mean')
df[numeric_cols] = imputer_mean.fit_transform(df[numeric_cols])

# Imputación categórica (moda)
imputer_mode = SimpleImputer(strategy='most_frequent')
df[categorical_cols] = imputer_mode.fit_transform(df[categorical_cols])

# Guardar dataset imputado
df.to_csv("Handball_W_InternationalResults_imputed.csv", index=False)
```

- Analisis de dispersión para analizar la correlación entre goles y torneos disputados

```python
import plotly.express as px

# Sample data
df = px.data.iris()

# Create a simple scatter plot
fig = px.scatter(df, x="sepal_width", y="sepal_length", color="species",
                 title="Iris Sepal Length vs. Width")

# Export the plot as an HTML file
fig.write_html("iris_scatter_plot.html")

print("Plot exported as iris_scatter_plot.html")


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
- Regresión lineal simple completada para predecir goles totales según torneo y año:
```python
R² (poder explicativo): 0.0176 → El modelo explica muy poco la variabilidad (los goles dependen de más factores).
Coeficientes principales:

year: -0.0038 (impacto mínimo del año)
AsianChampionship: +4.94
WorldChampionship: +3.12
OlympicAfricanQualification: -5.66 (torneos con menos goles)
Otros torneos varían entre ±4 goles.


Predicción
Para WorldChampionship en 2023:
≈ 50.66 goles totales por partido.

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Cargar dataset
df = pd.read_csv(r"/content/drive/MyDrive/Handball_W_InternationalResults.csv")
df['TotalGoals'] = df['ScoreA'] + df['ScoreB']

# Variables
X = df[['TournamentName', 'year']]
y = df['TotalGoals']

# Dummies para torneos
X = pd.get_dummies(X, columns=['TournamentName'], drop_first=True)

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modelo
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluación
r2 = r2_score(y_test, model.predict(X_test))
print("R²:", r2)

# Coeficientes
coef_df = pd.DataFrame({'Feature': X.columns, 'Coefficient': model.coef_})
print(coef_df)

# Predicción ejemplo
example = pd.DataFrame([[2023] + [0]*(len(X.columns)-1)], columns=X.columns)
pred_example = model.predict(example)[0]
print("Predicción WorldChampionship 2023:", pred_example)


```

Además, se han creado nuevas columnas en el dataset:

- `Resultado Partido`: nombre del equipo ganador o empate.
- `Diferencia de Goles`: diferencia absoluta entre los goles anotados por cada equipo.
- `Resultado de la cantidad de victorias por equipo`: Calcular el número de victorias por equip

Modelo Predictivo
- OLS (Regresión Lineal Simple y Múltiple)
```python
Regresión Simple (OLS):

Modelo 1: Regresión Simple (TotalGoals ~ year)

R²: 0.000 → El año no explica la variabilidad en los goles.
Coeficiente year: 0.0063 (p-valor = 0.754, no significativo).
Interpretación: El año prácticamente no influye en la cantidad de goles.


Modelo 2: Regresión Múltiple (TotalGoals ~ year + TournamentName)

R²: 0.030 → Explica solo el 3% de la variabilidad.
Variables significativas (p < 0.05):

AsianChampionship: +4.50 goles
CaribbeanCup: +3.49 goles
CentralAmericanAndCaribbeanGames: +3.23 goles
CentralAmericanChampionship: +3.91 goles


El año sigue sin ser significativo (coeficiente ≈ 0.0004).


✅ Predicción
Para año 2025 y torneo WorldChampionship:
≈ 50.16 goles totales por partido.

import pandas as pd
import statsmodels.api as sm

# Cargar dataset
df = pd.read_csv(r"/content/drive/MyDrive/Handball_W_InternationalResults.csv")
df['TotalGoals'] = df['ScoreA'] + df['ScoreB']

# Modelo simple
X_simple = sm.add_constant(df['year'].astype(float))
y = df['TotalGoals'].astype(float)
model_simple = sm.OLS(y, X_simple).fit()

# Modelo múltiple
dummies = pd.get_dummies(df['TournamentName'], drop_first=True)
X_multiple = pd.concat([df['year'], dummies], axis=1).astype(float)
X_multiple = sm.add_constant(X_multiple)
model_multiple = sm.OLS(y, X_multiple).fit()

# Predicción
pred_cols = model_multiple.model.exog_names
pred_data = pd.DataFrame([[0]*len(pred_cols)], columns=pred_cols)
pred_data['const'] = 1
pred_data['year'] = 2025
if 'WorldChampionship' in pred_data.columns:
    pred_data['WorldChampionship'] = 1
prediction = model_multiple.predict(pred_data)[0]
```

- Análisis de residuos
```python
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns

# Cargar datos
file_path = (r"/content/drive/MyDrive/Handball_W_InternationalResults.csv")
df = pd.read_csv(file_path)

# Crear variable dependiente y predictoras
# Usamos ScoreA como dependiente y ScoreB + year como predictoras
X = df[["ScoreB", "year"]]
y = df["ScoreA"]

# Agregar constante
X = sm.add_constant(X)

# Ajustar modelo OLS
model = sm.OLS(y, X).fit()

# Obtener residuos y valores ajustados
residuals = model.resid
fitted = model.fittedvalues

# Crear figura con 3 subplots
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# 1. Residuos vs Ajustados
sns.scatterplot(x=fitted, y=residuals, ax=axes[0])
axes[0].axhline(0, color='red', linestyle='--')
axes[0].set_title('Residuos vs Ajustados')
axes[0].set_xlabel('Valores Ajustados')
axes[0].set_ylabel('Residuos')

# 2. Histograma de residuos
sns.histplot(residuals, bins=30, kde=True, ax=axes[1])
axes[1].set_title('Histograma de Residuos')

# 3. QQ Plot
sm.qqplot(residuals, line='45', ax=axes[2])
axes[2].set_title('QQ Plot')

plt.tight_layout()
plt.show()

# Interpretación
print("""
Interpretación:
- Normalidad: El QQ Plot muestra si los residuos siguen una distribución normal.
- Homocedasticidad: El gráfico de residuos vs ajustados debe mostrar dispersión aleatoria alrededor de cero.
- Histograma: Si es simétrico y con forma de campana, respalda la normalidad.
""")

Interpretación adaptada

Normalidad: El QQ Plot muestra que los residuos del modelo OLS (ScoreA ~ ScoreB + year) siguen aproximadamente la línea de referencia, lo que indica que la distribución es cercana a normal lo cual es importante para la validez de los intervalos de confianza y pruebas de hipótesis.
Homocedasticidad: El gráfico de residuos vs valores ajustados no presenta un patrón claro, esto sugiere que la varianza de los errores es relativamente constante. 
Histograma: La forma del histograma es simétrica y similar a una campana, reforzando la idea de normalidad en los residuos.

En resumen, el modelo cumple razonablemente los supuestos básicos de OLS para este dataset de resultados internacionales de handball.

```

## Feature Selection & Machine Learning

Se realizo la selección de variables con SelectKBest usando la función estadística f_regression para predecir la variable objetivo ScoreA. Aquí está el ranking de importancia:



- Selección de variables mediante métodos estadísticos (SelectKBest, f_regression)

```python
VariableScore (f_regression)ScoreB826.18TeamA15.84year3.58TeamB1.78Venue0.87TournamentName0.04Sex0.00
Interpretación

ScoreB (puntaje del equipo contrario) es, como era de esperar, la variable más influyente para predecir ScoreA.
TeamA y year también aportan algo de información, aunque mucho menor.
Variables como Venue, TournamentName y Sex prácticamente no tienen relevancia estadística para esta predicción.

Se realizó la selección de variables con SelectKBest usando la función estadística f_regression para predecir la variable objetivo ScoreA

ScoreB (puntaje del equipo contrario) es, como era de esperar, la variable más influyente para predecir ScoreA.
TeamA y year también aportan algo de información, aunque mucho menor.
Variables como Venue, TournamentName y Sex prácticamente no tienen relevancia estadística para esta predicción.


```

- Prediccion de que pais es el mas determinante para las victorias basado en las mismas como en la cantidad de goles convertiodos.
```python

import pandas as pd
import plotly.express as px

# Cargar el dataset
file_path = (r"/content/drive/MyDrive/Handball_W_InternationalResults.csv")
df = pd.read_csv(file_path)

# Crear variable objetivo: 1 si TeamA gana, 0 si no
df['WinA'] = (df['ScoreA'] > df['ScoreB']).astype(int)

# Calcular tasa de victorias por país (TeamA)
victory_rate = df.groupby('TeamA')['WinA'].mean().sort_values(ascending=False)
victory_rate_df = victory_rate.reset_index()
victory_rate_df.columns = ['Pais', 'Tasa_Victoria']

# Mostrar top 10 países más determinantes
print("Top 10 países con mayor tasa de victoria:")
print(victory_rate_df.head(10))

# Crear gráfico de barras
fig = px.bar(victory_rate_df.head(10), x='Pais', y='Tasa_Victoria',
             title='Top 10 Países con Mayor Tasa de Victoria', text='Tasa_Victoria')
fig.show()

```

## Feature Selection & Machine Learning

```python

import pandas as pd
from sklearn.model_selection import train_test_split

# Cargar el dataset
file_path = (r"/content/drive/MyDrive/Handball_W_InternationalResults.csv")
df = pd.read_csv(file_path)

# Crear variable objetivo: 1 si TeamA gana, 0 si no
df['WinA'] = (df['ScoreA'] > df['ScoreB']).astype(int)

# Eliminar columnas irrelevantes y manejar valores nulos
df = df.drop(columns=['Date'])
df = df.dropna()

# Definir X (predictoras) y y (objetivo)
X = df.drop(columns=['WinA'])
y = df['WinA']

# División en Train (70%), Validation (15%), Test (15%) con estratificación
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, stratify=y, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42)

# Mostrar tamaños de cada conjunto
print("Tamaños de los conjuntos:")
print(f"Train: {X_train.shape[0]} filas")
print(f"Validation: {X_val.shape[0]} filas")
print(f"Test: {X_test.shape[0]} filas")ython
import pandas as pd
from sklearn.model_selection import train_test_split

# Cargar el dataset
file_path = (r"/content/drive/MyDrive/Handball_W_InternationalResults.csv")
df = pd.read_csv(file_path)

# Crear variable objetivo: 1 si TeamA gana, 0 si no
df['WinA'] = (df['ScoreA'] > df['ScoreB']).astype(int)

# Eliminar columnas irrelevantes y manejar valores nulos
df = df.drop(columns=['Date'])
df = df.dropna()

# Definir X (predictoras) y y (objetivo)
X = df.drop(columns=['WinA'])
y = df['WinA']

# División en Train (70%), Validation (15%), Test (15%) con estratificación
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, stratify=y, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42)

# Mostrar tamaños de cada conjunto
print("Tamaños de los conjuntos:")
print(f"Train: {X_train.shape[0]} filas")
print(f"Validation: {X_val.shape[0]} filas")
print(f"Test: {X_test.shape[0]} filas")

 División realizada con estratificación para mantener la proporción de victorias en cada conjunto:

Train: 2,712 filas (70%)
Validation: 581 filas (15%)
Test: 582 filas (15%)


```
- Evaluacion con metricas R2, RMSE


```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import LabelEncoder
import numpy as np
# Cargar el dataset
file_path = (r"/content/drive/MyDrive/Handball_W_InternationalResults.csv")
df = pd.read_csv(file_path)

# Eliminar columnas irrelevantes y manejar valores nulos
df = df.drop(columns=['Date'])
df = df.dropna()

# Definir variable objetivo y predictoras
y = df['ScoreA']
X = df.drop(columns=['ScoreA'])

# Codificar variables categóricas
label_encoders = {}
for col in X.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    label_encoders[col] = le

# División en Train (70%), Validation (15%), Test (15%)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42)

# Entrenar modelo RandomForestRegressor
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predicciones en Validation y Test
y_val_pred = model.predict(X_val)
y_test_pred = model.predict(X_test)

# Calcular métricas R² y RMSE
r2_val = r2_score(y_val, y_val_pred)
rmse_val = np.sqrt(mean_squared_error(y_val, y_val_pred))

r2_test = r2_score(y_test, y_test_pred)
rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))

# Mostrar resultados
print("Evaluación del modelo RandomForestRegressor:")
print(f"Validation -> R²: {r2_val:.4f}, RMSE: {rmse_val:.4f}")
print(f"Test -> R²: {r2_test:.4f}, RMSE: {rmse_test:.4f}")

Resultados de la evaluación del modelo RandomForestRegressor:
R²: 0.3864
RMSE: 6.32

Test
R²: 0.2832
RMSE: 6.21

El modelo explica aproximadamente 38.6% de la varianza en el conjunto de validación y 28.3% en el test, lo que indica un desempeño moderado.
El RMSE (~6 goles) sugiere que el error promedio en la predicción del marcador es significativo, pero aceptable para datos deportivos donde la variabilidad es alta.
Podría mejorar con:

Feature engineering (por ejemplo, diferencia de goles, historial de equipos).
Modelos más complejos o ajuste de hiperparámetros.

```

- Visualizacion de resultados y predicciones

```python

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np
import plotly.express as px

# Cargar el dataset
file_path = (r"/content/drive/MyDrive/Handball_W_InternationalResults.csv")
df = pd.read_csv(file_path)

# Preprocesamiento: eliminar columnas irrelevantes y manejar valores nulos
df = df.drop(columns=['Date'])
df = df.dropna()

# Definir variable objetivo y predictoras
y = df['ScoreA']
X = df.drop(columns=['ScoreA'])

# Codificar variables categóricas
for col in X.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])

# División en Train (70%), Validation (15%), Test (15%)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42)


Interpretación

El modelo captura cierta tendencia, pero hay variabilidad alta.
R²: 0.386 (Validación), 0.283 (Test) → desempeño moderado.
RMSE: ~6 goles → error promedio significativo.
```

 

## 🛠️ Herramientas utilizadas

- Python 3.x → Base para todo el procesamiento y análisis.
- Pandas → Lectura, limpieza, transformación y análisis de datos.
- NumPy → Operaciones numéricas y manejo de arrays.
- Matplotlib → Gráficos básicos.
- Seaborn → Visualizaciones estadísticas (boxplots, distribuciones).
- Plotly → Gráficos interactivos (histogramas, scatter plots, boxplots).
- Statsmodels → Modelos OLS (Regresión Lineal Simple y Múltiple) con análisis detallado (p-valores, R²).
- Scikit-learn → Modelos predictivos (Regresión Lineal, imputación con SimpleImputer).
- Google Colab / Jupyter Notebook

## 📁 Estructura del proyecto

- `Rosales_Internationalresults_handball.ipynb`: notebook principal con análisis y visualizaciones https://colab.research.google.com/drive/1HCWJG1xg51Bk8JSu1iLII-78TraJrQ1t?usp=sharing 
- `README.md`: descripción del proyecto.
- `Handball_W_InternationalResults.csv`: dataset utilizado.

