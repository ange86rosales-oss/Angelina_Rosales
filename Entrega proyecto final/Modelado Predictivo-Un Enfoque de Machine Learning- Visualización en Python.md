 # Modelado Predictivo de la performance de las selecciones internacionales de handball femenino: Un Enfoque de Machine Learning para analizar la predictibilidad de las próximas selecciones triunfadoras.

## 📘 Contexto

Desde el año 2010, el handball femenino mundial ha vivido una transformación marcada por la intensidad competitiva, el surgimiento de nuevas potencias y la consolidación de selecciones históricas. Europa mantuvo su hegemonía, pero también hubo sorpresas que rompieron el molde.
Noruega, con su estilo veloz y técnico, se consagró como una de las selecciones más dominantes, alzando el trofeo en varias ocasiones y manteniéndose siempre cerca del podio. Francia, por su parte, fue construyendo una generación dorada que alcanzó la gloria tanto en mundiales como en los Juegos Olímpicos, donde logró el oro en Tokio 2020. Países Bajos también dejó su huella, conquistando el mundo en 2019 con una actuación memorable en Japón.
Pero no todo fue Europa. En 2013, Brasil sorprendió al mundo entero al coronarse campeona en su propia tierra, demostrando que el talento sudamericano podía competir al más alto nivel. Esa victoria fue histórica, no solo por el título, sino por lo que representó para el desarrollo del deporte en América Latina.
En los Juegos Olímpicos, el handball femenino también vivió momentos intensos. Rusia, compitiendo bajo la bandera del Comité Olímpico Ruso, se llevó el oro en Río 2016, mientras que Noruega y Francia se mantuvieron como protagonistas constantes. La edición de Tokio, celebrada en 2021 por la pandemia, fue testigo de la consagración francesa, que venció a Rusia en una final cargada de emoción.
A lo largo de estos años, el deporte se volvió más global. Cuba, por ejemplo, logró en 2025 una clasificación histórica al Mundial tras ganar el campeonato regional de América del Norte y el Caribe, mostrando que el crecimiento del handball femenino no se limita a Europa.
Cada torneo, cada medalla, cada partido disputado en estos quince años ha sido parte de una narrativa que habla de esfuerzo, evolución y pasión. El handball femenino mundial se ha convertido en un espectáculo de alto nivel, donde la técnica, la táctica y el corazón se combinan para ofrecer historias inolvidables. 


## 🎯 Objetivo e hipótesis del proyecto

Este estudio desarrolla un framework predictivo integral para analizar los proximos resultados de partidos internacionales de handball femenino, para el cual analizaremos los partidos disputados desde 2010 al 2023 en los que se han disputado más de 2800 partidos oficiales lo que muestra una actividad constante y creciente lo que nos permite ver cómo el handball femenino ha crecido en volumen, diversidad y competitividad. Europa sigue siendo el núcleo, pero otras regiones están pisando fuerte y ganando terreno. 
Dicho todo lo anterior, analizaremos las tendencias observadas en los torneos más importantes

En este sentido, analizaremos con datos respaldatorios, las tendencias observadas en los torneos más importantes a partir del 2024 y veremos si Europa seguirá  manteniendo su supremacia hegemónica de cara al mundial a disputarse en noviembre 2025.


## ❓ Preguntas de interés

- ¿Qué equipos tienen el mejor promedio de victorias en torneos específicos?
- ¿Cuál fue el ranking de goles por país entre 2020 y 2023?
- ¿Existe una relación entre la diferencia de goles y el tipo de torneo?
- ¿Hay equipos que consistentemente ganan por márgenes amplios?


## 📊 Visualizaciones y análisis

#### Este código realiza tres tareas principales sobre el dataset de partidos de handball femenino:

1.	Carga del dataset

2.	Limpieza de nombres de columnas: Se eliminan espacios en blanco al inicio o final de los nombres de las columnas, lo que evita errores al acceder a ellas

3.	Cálculo del equipo ganador: Se crea una nueva columna llamada Resultado Partido que indica el nombre del equipo que ganó el partido (según los goles) y si los goles fueron iguales se asigna "Empate".

```Phyton
import pandas as pd
import plotly.express as px
import numpy as np


df = pd.read_csv ("https://raw.githubusercontent.com/ange86rosales-oss/Angelina_Rosales/refs/heads/Reentrega-proyecto-final/Entrega%20proyecto%20final/Handball_W_InternationalResults_with_Winner.csv")

df.columns = df.columns.str.strip()

# Agregar la columna 'Resultado Partido' con el nombre del equipo ganador
df["Resultado Partido"] = np.where(df["ScoreA"] > df["ScoreB"], df["TeamA"],np.where(df["ScoreA"] < df["ScoreB"], df["TeamB"], "Empate"))


print (df.head())
```


#### Este código realiza un análisis para mostrar qué equipos anotaron más goles en partidos internacionales de handball femenino entre 2010 y 2023

```Phyton
import pandas as pd
import plotly.express as px

# Cargar el archivo CSV
df = pd.read_csv ("https://raw.githubusercontent.com/ange86rosales-oss/Angelina_Rosales/refs/heads/Reentrega-proyecto-final/Entrega%20proyecto%20final/Handball_W_InternationalResults_with_Winner.csv")

# Limpiar los nombres de las columnas
df.columns = df.columns.str.strip()

# Crear un DataFrame con goles por equipo
goals_by_team = pd.DataFrame({
    "Team": df["TeamA"].tolist() + df["TeamB"].tolist(),
    "Goals": df["ScoreA"].tolist() + df["ScoreB"].tolist()
})

# Calcular goles totales por equipo
total_goals = goals_by_team.groupby("Team")["Goals"].sum().reset_index()

# Ordenar por goles descendente
total_goals = total_goals.sort_values(by="Goals", ascending=False)

# Crear gráfico de barras
fig = px.bar(
    total_goals,
    x="Team",
    y="Goals",
    title="Ranking de Goles Totales por Equipo 2010-2023",
    labels={"Team": "Selección", "Goals": "Goles Totales"}
)

fig.show()
```


#### Este código genera un boxplot que muestra la distribución de la diferencia de goles en los partidos ganados por los 10 equipos con más victorias del 2010 al 2023
```Phyton
import pandas as pd
import plotly.express as px

# Cargar el archivo CSV
df = pd.read_csv ("https://raw.githubusercontent.com/ange86rosales-oss/Angelina_Rosales/refs/heads/Reentrega-proyecto-final/Entrega%20proyecto%20final/Handball_W_InternationalResults_with_Winner.csv")


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


#### Este código genera un gráfico tipo treemap que muestra la cantidad de victorias por equipo organizadas por torneo.

Esto permite ver qué equipos dominan cada torneo
```Phyton
import pandas as pd
import plotly.express as px

# Cargar el archivo CSV
df = pd.read_csv ("https://raw.githubusercontent.com/ange86rosales-oss/Angelina_Rosales/refs/heads/Reentrega-proyecto-final/Entrega%20proyecto%20final/Handball_W_InternationalResults_with_Winner.csv")


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


#### Este código realiza un diagnóstico de valores faltantes (nulos) en cada columna del dataset
```Phyton
import pandas as pd

# Cargar el archivo CSV
df = pd.read_csv ("https://raw.githubusercontent.com/ange86rosales-oss/Angelina_Rosales/refs/heads/Reentrega-proyecto-final/Entrega%20proyecto%20final/Handball_W_InternationalResults_with_Winner.csv")

# Diagnóstico de valores perdidos por columna
missing_values = df.isnull().sum()

# Mostrar el resultado
print("Diagnóstico de valores perdidos por columna:")
print(missing_values)
```

#### Este código calcula qué equipo tiene la mejor tasa de victorias en cada torneo y muestra los resultados en una tabla.
```Phyton
import pandas as pd

# Cargar el dataset
df = pd.read_csv ("https://raw.githubusercontent.com/ange86rosales-oss/Angelina_Rosales/refs/heads/Reentrega-proyecto-final/Entrega%20proyecto%20final/Handball_W_InternationalResults_with_Winner.csv")

# Limpiar los nombres de las columnas
df.columns = df.columns.str.strip()

# Determinar el equipo ganador por partido
df["WinningTeam"] = df.apply(
    lambda row: row["TeamA"] if row["ScoreA"] > row["ScoreB"]
    else row["TeamB"] if row["ScoreB"] > row["ScoreA"]
    else "Draw", axis=1
)


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

```

#### Este codigo expone dos puntos importantes para nuestro analisis:

- El primer gráfico muestra qué equipos son más efectivos en cada torneo.

- El segundo gráfico muestra qué equipos son globalmente más dominantes en la historia del dataset.
```Phyton
import pandas as pd
import plotly.express as px

# Cargar el archivo CSV
df = pd.read_csv("https://raw.githubusercontent.com/ange86rosales-oss/Angelina_Rosales/refs/heads/Reentrega-proyecto-final/Entrega%20proyecto%20final/Handball_W_InternationalResults_with_Winner.csv")

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
fig1.show()

# --- 2) Ranking global ---
global_wins = df.groupby('WinningTeam').size().reset_index(name='TotalWins')
global_wins_sorted = global_wins.sort_values('TotalWins', ascending=False).head(20)

fig2 = px.bar(global_wins_sorted, x='WinningTeam', y='TotalWins',
              title='Ranking global de equipos más dominantes (total de victorias)',
              labels={'WinningTeam': 'Equipo', 'TotalWins': 'Total de victorias'})
fig2.show()
```


#### Este código realiza un EDA con los resultados de partidos de handball femenino.

- Aqui vemos la distribución de goles, es decir los equipos que más anotan.
- Cuales son los torneos más frecuentes, es decir, ¿Dónde se juega más?
- Y por útimo la correlación entre goles, es decir, los partidos son palo y palo o desparejos?

```Phyton
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff

# Cargar dataset
df = pd.read_csv ("https://raw.githubusercontent.com/ange86rosales-oss/Angelina_Rosales/refs/heads/Reentrega-proyecto-final/Entrega%20proyecto%20final/Handball_W_InternationalResults_with_Winner.csv")

# Limpiar los nombres de las columnas
df.columns = df.columns.str.strip()

# Determinar el equipo ganador por partido
df["WinningTeam"] = df.apply(
    lambda row: row["TeamA"] if row["ScoreA"] > row["ScoreB"]
    else row["TeamB"] if row["ScoreB"] > row["ScoreA"]
    else "Draw", axis=1
)


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



#### Aqui eliminamos datos que pueden distorsionar análisis estadísticos y modelos predictivos, la finalidad es mejorar la calidad del dataset para obtener resultados más confiables
Tambien evita errores en modelos predictivos que no aceptan valores nulos y mejora la calidad del dataset para análisis estadísticos.
```Phyton
import pandas as pd

# Cargar dataset
df = pd.read_csv ("https://raw.githubusercontent.com/ange86rosales-oss/Angelina_Rosales/refs/heads/Reentrega-proyecto-final/Entrega%20proyecto%20final/Handball_W_InternationalResults_with_Winner.csv")

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

import pandas as pd
from sklearn.impute import SimpleImputer

# Cargar dataset
df = pd.read_csv("https://raw.githubusercontent.com/ange86rosales-oss/Angelina_Rosales/refs/heads/Reentrega-proyecto-final/Entrega%20proyecto%20final/Handball_W_InternationalResults_with_Winner.csv")

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


#### Aqui observamos la distribucion de la dispersión de los equipos, es decir, si hay patrones o agrupamientos y correlaciones entre las variables.

El gráfico muestra la relación entre los goles anotados por el Equipo A y el Equipo B en cada partido. La línea de tendencia (OLS) indica cuando un equipo anota , el otro equipo tiende a anotar más goles. Esto sugiere que los partidos suelen ser competitivos, con ambos equipos anotando en rangos similares.

```Phyton
import pandas as pd
import plotly.express as px

# Cargar el dataset
df = pd.read_csv("https://raw.githubusercontent.com/ange86rosales-oss/Angelina_Rosales/refs/heads/Reentrega-proyecto-final/Entrega%20proyecto%20final/Handball_W_InternationalResults_with_Winner.csv")

# Limpiar nombres de columnas
df.columns = df.columns.str.strip()

# Crear gráfico de dispersión con línea de tendencia entre ScoreA y ScoreB
fig = px.scatter(
    df,
    x="ScoreB",
    y="ScoreA",
    trendline="ols",  # Agrega línea de regresión lineal
    title="Relación entre goles del equipo B (ScoreB) y del equipo A (ScoreA)",
    labels={"ScoreB": "Goles del Equipo B", "ScoreA": "Goles del Equipo A"}
)

fig.show()
```


#### Aqui realizamos un análisis predictivo utilizando regresión lineal para estimar la cantidad total de goles anotados basándose en el año y el nombre del torneo.**
```Phyton
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Cargar dataset
df = pd.read_csv("https://raw.githubusercontent.com/ange86rosales-oss/Angelina_Rosales/refs/heads/Reentrega-proyecto-final/Entrega%20proyecto%20final/Handball_W_InternationalResults_with_Winner.csv")
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
example = pd.DataFrame([[2025] + [0]*(len(X.columns)-1)], columns=X.columns)
pred_example = model.predict(example)[0]
print("Predicción WorldChampionship 2025:", pred_example)

```



#### Aqui realizamos un análisis de regresión lineal utilizando la librería statsmodels para entender cómo el año y el tipo de torneo influyen en la cantidad total de goles anotados.**
```Phyton
import pandas as pd
import statsmodels.api as sm

# Cargar dataset
df = pd.read_csv("https://raw.githubusercontent.com/ange86rosales-oss/Angelina_Rosales/refs/heads/Reentrega-proyecto-final/Entrega%20proyecto%20final/Handball_W_InternationalResults_with_Winner.csv")
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


#### Aqui intentamos predecir los goles anotados por el equipo A (ScoreA) en función de los goles recibidos (ScoreB) y el año del partido (year) se hace mediante una regresión lineal ordinaria (OLS)
```Phyton
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns

# Cargar datos
file_path = ("https://raw.githubusercontent.com/ange86rosales-oss/Angelina_Rosales/refs/heads/Reentrega-proyecto-final/Entrega%20proyecto%20final/Handball_W_InternationalResults_with_Winner.csv")
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
```



#### Aqui queremos determinar qué variables tienen mayor influencia en la cantidad de goles anotados por el equipo A y visualizar la importancia relativa de cada variable.
```Phyton
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import plotly.express as px

# Cargar el dataset
file_path = ("https://raw.githubusercontent.com/ange86rosales-oss/Angelina_Rosales/refs/heads/Reentrega-proyecto-final/Entrega%20proyecto%20final/Handball_W_InternationalResults_with_Winner.csv")
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

# Entrenar modelo RandomForestRegressor
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# Obtener importancia de variables
importances = model.feature_importances_
feature_names = X.columns
importance_df = pd.DataFrame({'Variable': feature_names, 'Importancia': importances})
importance_df = importance_df.sort_values(by='Importancia', ascending=False)

# Mostrar ranking
print("Ranking de importancia de variables (Random Forest):")
print(importance_df)

# Crear gráfico de barras
fig = px.bar(importance_df, x='Variable', y='Importancia',
             title='Importancia de Variables (Random Forest)', text='Importancia')
fig.show()
```


#### Aqui se pretende identificar selecciones dominantes y evaluar rendimientos históricos, preparación de partidos y análisis estratégico.
```Phyton
import pandas as pd
import plotly.express as px

# Cargar el dataset
file_path = ("https://raw.githubusercontent.com/ange86rosales-oss/Angelina_Rosales/refs/heads/Reentrega-proyecto-final/Entrega%20proyecto%20final/Handball_W_InternationalResults_with_Winner.csv")
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



#### Aqui verificamos el entrenamiento de modelos supervisados ya que se analiza:

Entrenamiento (Train): Se usa para ajustar el modelo.

Validación (Validation): Se usa para ajustar hiperparámetros y evitar sobreajuste.

Prueba (Test): Se usa para evaluar el rendimiento final del modelo en datos no vistos.

```Phyton
import pandas as pd
from sklearn.model_selection import train_test_split

# Cargar el dataset
file_path = ("https://raw.githubusercontent.com/ange86rosales-oss/Angelina_Rosales/refs/heads/Reentrega-proyecto-final/Entrega%20proyecto%20final/Handball_W_InternationalResults_with_Winner.csv")
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
```



#### Aqui buscamos predecir los goles anotados por el equipo A (ScoreA), y evalúa su rendimiento en conjuntos de validación y prueba con Métricas de evaluación (R²) y Error promedio en la predicción (RMSE)
```Phyton
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import LabelEncoder
import numpy as np
# Cargar el dataset
file_path = ("https://raw.githubusercontent.com/ange86rosales-oss/Angelina_Rosales/refs/heads/Reentrega-proyecto-final/Entrega%20proyecto%20final/Handball_W_InternationalResults_with_Winner.csv")
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

```




#### Aqui buscamos predecir los goles del equipo A (ScoreA) y evalúa el modelo con tres métricas:

R² : variabilidad de los datos.

RMSE: indica cuánto se desvía la predicción en la misma escala que los goles.

MAE : error absoluto promedio.

```Phyton
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Cargar el dataset
df = pd.read_csv("https://raw.githubusercontent.com/ange86rosales-oss/Angelina_Rosales/refs/heads/Reentrega-proyecto-final/Entrega%20proyecto%20final/Handball_W_InternationalResults_with_Winner.csv")

# Preprocesamiento: eliminar columnas irrelevantes y manejar valores nulos
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

# Dividir en conjuntos de entrenamiento, validación y prueba
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42)

# Entrenar modelo RandomForestRegressor
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predicciones
y_val_pred = model.predict(X_val)
y_test_pred = model.predict(X_test)

# Calcular métricas
r2_val = r2_score(y_val, y_val_pred)
rmse_val = np.sqrt(mean_squared_error(y_val, y_val_pred))
mae_val = mean_absolute_error(y_val, y_val_pred)

r2_test = r2_score(y_test, y_test_pred)
rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))
mae_test = mean_absolute_error(y_test, y_test_pred)

# Mostrar resultados
print("Evaluación del modelo RandomForestRegressor:")
print(f"Validation -> R²: {r2_val:.4f}, RMSE: {rmse_val:.4f}, MAE: {mae_val:.4f}")
print(f"Test -> R²: {r2_test:.4f}, RMSE: {rmse_test:.4f}, MAE: {mae_test:.4f}")

```



#### Aqui dividimos el dataset en tres conjuntos para entrenar el modelo en datos conocidos, validar para ajustar hiperparámetros y evitar sobreajuste y para medir el rendimiento en datos no vistos.
```Phyton
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np
import plotly.express as px

# Cargar el dataset
file_path = ("https://raw.githubusercontent.com/ange86rosales-oss/Angelina_Rosales/refs/heads/Reentrega-proyecto-final/Entrega%20proyecto%20final/Handball_W_InternationalResults_with_Winner.csv")
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
```


#### Aqui lo que se pretende estimar es la tendencia histórica y proyectar qué equipos podrían dominar en el futuro.
```Phyton
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error

# Cargar el dataset
df = pd.read_csv("https://raw.githubusercontent.com/ange86rosales-oss/Angelina_Rosales/refs/heads/Reentrega-proyecto-final/Entrega%20proyecto%20final/Handball_W_InternationalResults_with_Winner.csv")

# Filtrar solo partidos del torneo World Championship
df_wc = df[df['TournamentName'] == 'WorldChampionship']

# Eliminar empates
df_wc = df_wc[df_wc['WinningTeam'] != 'Draw']

# Contar victorias por equipo y año
victories = df_wc.groupby(['WinningTeam', 'year']).size().reset_index(name='Victories')

# Crear dataset para modelado
X = victories[['WinningTeam', 'year']]
y = victories['Victories']

# Codificar variable categórica (WinningTeam)
le_team = LabelEncoder()
X['WinningTeam'] = le_team.fit_transform(X['WinningTeam'])

# Dividir en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenar modelo Random Forest
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluar con MAE
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
print(f"MAE: {mae:.2f}")

# Predecir para el año 2025 para todos los equipos históricos
teams = victories['WinningTeam'].unique()
future_data = pd.DataFrame({'WinningTeam': teams, 'year': 2025})
future_data['WinningTeam'] = le_team.transform(future_data['WinningTeam'])

# Predecir victorias
future_predictions = model.predict(future_data)
future_data['PredictedVictories'] = future_predictions

# Obtener el equipo con más victorias proyectadas
future_data['TeamName'] = le_team.inverse_transform(future_data['WinningTeam'])
best_team = future_data.sort_values('PredictedVictories', ascending=False).iloc[0]

print("\nEquipo con mayor cantidad de victorias proyectadas en el World Championship 2025:")
print(f"Equipo: {best_team['TeamName']}, Victorias proyectadas: {best_team['PredictedVictories']:.2f}")

```


#### Aqui lo que se intenta es estimar goles futuros en función de características históricas, analizar y saber qué factores influyen más (torneo, año, rival), ajustar variables y parámetros para mejorar la precisión y facilitar la interpretación de resultados para informes o presentaciones.

```Phyton
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.feature_selection import SelectKBest, f_regression
import plotly.express as px

# 1. Cargar el dataset
df = pd.read_csv("https://raw.githubusercontent.com/ange86rosales-oss/Angelina_Rosales/refs/heads/Reentrega-proyecto-final/Entrega%20proyecto%20final/Handball_W_InternationalResults_with_Winner.csv")

# 2. Preprocesamiento: eliminar columnas irrelevantes y manejar valores nulos
df = df.drop(columns=['Date', 'Sex'])  # Excluimos 'Sex'
df = df.dropna()

# 3. Definir variable objetivo y predictoras
y = df['ScoreA']
X = df.drop(columns=['ScoreA'])

# 4. Codificar variables categóricas
for col in X.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])

# 5. Selección de características con SelectKBest
selector = SelectKBest(score_func=f_regression, k='all')  # puedes cambiar 'all' por un número
X_selected = selector.fit_transform(X, y)
selected_features = X.columns[selector.get_support(indices=True)]

# 6. División en Train, Validation y Test
X_train, X_temp, y_train, y_temp = train_test_split(X_selected, y, test_size=0.30, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42)

# 7. Entrenar modelo RandomForestRegressor
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 8. Predicciones
y_val_pred = model.predict(X_val)
y_test_pred = model.predict(X_test)

# 9. Métricas
r2_val = r2_score(y_val, y_val_pred)
rmse_val = np.sqrt(mean_squared_error(y_val, y_val_pred))
mae_val = mean_absolute_error(y_val, y_val_pred)

r2_test = r2_score(y_test, y_test_pred)
rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))
mae_test = mean_absolute_error(y_test, y_test_pred)

print("Evaluación del modelo RandomForestRegressor con SelectKBest (sin 'Sex'):")
print(f"Validation -> R²: {r2_val:.4f}, RMSE: {rmse_val:.4f}, MAE: {mae_val:.4f}")
print(f"Test -> R²: {r2_test:.4f}, RMSE: {rmse_test:.4f}, MAE: {mae_test:.4f}")

print("\nCaracterísticas seleccionadas por SelectKBest:")
for feature in selected_features:
    print(f"- {feature}")

# 10. Importancia de variables
importances = model.feature_importances_
importance_df = pd.DataFrame({
    'Variable': selected_features,
    'Importancia': importances
}).sort_values(by='Importancia', ascending=False)

# 11. Gráfico de importancia
fig = px.bar(importance_df, x='Variable', y='Importancia',
             title='Random Forest', text='Importancia')
fig.show()

```

##  ✅ Conclusiones y recomendaciones 

#### Supremacía europea confirmada en los datos

El análisis realizado sobre el dataset histórico de resultados internacionales de handball femenino confirma que Europa mantiene su supremacía en este deporte. Los equipos europeos como Noruega, Francia, Dinamarca, Rusia y Hungría aparecen de manera consistente entre los más exitosos en torneos globales, especialmente en el World Championship y el European Championship.
En los rankings y proyecciones realizadas con Random Forest, Noruega principalmente aparecen entre los equipos con mayor número de victorias proyectadas para 2025.

#### Modelos predictivos y métricas

El modelo Random Forest Regressor aplicado sobre las variables del dataset (equipos, torneo, año, goles del rival) logró un R² entre 0.43 y 0.53, con MAE ≈ 4 goles. Esto indica que el modelo captura tendencias históricas, aunque no predice con alta precisión debido a la falta de variables contextuales (ranking, localía, fase del torneo).

#### Selección de características (SelectKBest)

Las variables más relevantes para explicar los goles fueron:
TeamA, TeamB, ScoreB, TournamentName, year, Venue, WinningTeam.
La columna Sex no aportó valor predictivo, lo que confirma que el género no influye en este contexto porque todos los partidos son femeninos.



#### Importancia de variables en Random Forest

ScoreB (goles del equipo rival) es el predictor más influyente. Factores como torneo y año también tienen peso, lo que refleja que el contexto histórico y competitivo importa.

#### 📌Mi hipótesis “Europa seguirá con su supremacía” se sostiene porque:

Los datos históricos muestran dominio europeo en títulos y victorias, las proyecciones del modelo para 2025 ubican a equipos europeos en la cima.
La tendencia se mantiene estable en los últimos 20 años según el dataset. No obstante, se observa crecimiento en otras regiones, lo que sugiere que, aunque la supremacía europea se mantiene, la competencia global podría intensificarse en el futuro ya que se evidencia un marcado crecimiento en otras regiones por ejemplo Angola en África y  Brasil en América


#### Se podría agregar:

- Evidencia gráfica: evolución histórica de victorias por continente.
- Mención de equipos europeos más dominantes (Noruega, Francia y por fuera Rusia).
- Limitaciones: aunque Europa domina, hay crecimiento en otras regiones.
 

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

- `ProyectoParteIII+ROSALES.ipynb`: notebook principal con análisis y visualizaciones [(https://colab.research.google.com/github/ange86rosales-oss/Angelina_Rosales/blob/main/Entrega%20proyecto%20final/ProyectoParteIII%2BROSALES.ipynb)]
- `Modelado Predictivo-Un Enfoque de Machine Learning- Visualización en Python.md`: descripción del proyecto.
- `Handball_W_InternationalResults_with_Winner.csv`: dataset utilizado.

