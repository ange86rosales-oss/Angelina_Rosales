# Primer entrega - Visualizaciones en Python

## 📘 Abstract

Este proyecto analiza los resultados de partidos internacionales de handball femenino utilizando un dataset que contiene información sobre equipos, fechas, sedes, torneos y resultados numéricos. El objetivo principal es explorar patrones de rendimiento entre equipos, identificar tendencias en los resultados y evaluar la competitividad de los encuentros. A través de visualizaciones univariadas, bivariadas y multivariadas, se busca responder preguntas clave sobre el comportamiento de los equipos en distintos torneos y años. Además, se realiza un diagnóstico de valores perdidos y se enriquecen los datos con nuevas variables como el equipo ganador, la diferencia de goles y el tipo de victoria. Este análisis permite comprender mejor la dinámica del handball femenino internacional y ofrece una base para futuras investigaciones deportivas.

## ❓ Preguntas e hipótesis de interés

- ¿Qué equipos tienen mayor cantidad de victorias en torneos específicos?
- ¿Cuál fue el ranking de goles por país entre 2020 y 2023?
- ¿Existe una relación entre la diferencia de goles y el tipo de torneo?
- ¿Hay equipos que consistentemente ganan por márgenes amplios?

## 🎯 Objetivo del Proyecto

El objetivo de este proyecto es realizar un análisis exploratorio del rendimiento de los equipos en competiciones internacionales de handball femenino en el periodo de 2020 y 2023, utilizando visualizaciones en Python para identificar patrones, tendencias y relaciones entre variables clave. A través del procesamiento del dataset Handball_W_InternationalResults.csv, se busca responder preguntas relevantes sobre la competitividad de los equipos, la evolución de los resultados a lo largo del tiempo, y las características de las victorias en distintos torneos. 

## 📊 Visualizaciones y análisis

Se han generado visualizaciones que incluyen:

- Gráficos de barras con ranking de goles totales por equipo entre 2020 y 2023.

### Calcular goles totales por equipo
total_goals = goals_by_team.groupby("Team")["Goals"].sum().reset_index()

### Ordenar por goles descendente
total_goals = total_goals.sort_values(by="Goals", ascending=False)

### Crear gráfico de barras
fig = px.bar(
    total_goals,
    x="Team",
    y="Goals",
    title="Ranking de Goles Totales por Equipo 2020-2023",
    labels={"Team": "Equipo", "Goals": "Goles Totales"}
)

fig.show()
  
- Gráficos de cajas con el top 1o de diferencia de goles por equipo.
- Visualizaciones multivariadas que combinan equipo, goles y tipo de victoria.

Además, se han creado nuevas columnas en el dataset:

- `Resultado Partido`: nombre del equipo ganador o empate.
- `Diferencia de Goles`: diferencia absoluta entre los goles anotados por cada equipo.
- `Resultado de la cantidad de victorias por equipo`: Calcular el número de victorias por equipo

  ###  Missing values
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

- `Rosales_Internationalresults_handball.ipynb`: notebook principal con análisis y visualizaciones.
- `README.md`: descripción del proyecto.
- `Handball_W_InternationalResults.csv`: dataset utilizado.

