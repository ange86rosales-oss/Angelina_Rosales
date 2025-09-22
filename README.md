# Primer entrega - Visualizaciones en Python

## 📘 Abstract

Este proyecto analiza los resultados de partidos internacionales de handball femenino utilizando un dataset que contiene información sobre equipos, fechas, sedes, torneos y resultados numéricos. El objetivo principal es explorar patrones de rendimiento entre equipos, identificar tendencias en los resultados y evaluar la competitividad de los encuentros. A través de visualizaciones univariadas, bivariadas y multivariadas, se busca responder preguntas clave sobre el comportamiento de los equipos en distintos torneos y años. Además, se realiza un diagnóstico de valores perdidos y se enriquecen los datos con nuevas variables como el equipo ganador, la diferencia de goles y el tipo de victoria. Este análisis permite comprender mejor la dinámica del handball femenino internacional y ofrece una base para futuras investigaciones deportivas.

## ❓ Preguntas e hipótesis de interés

- ¿Qué equipos tienen mayor cantidad de victorias en torneos específicos?
- ¿Existe una relación entre la diferencia de goles y el tipo de torneo o sede?
- ¿Los partidos con mayor diferencia de goles son más frecuentes en ciertos años o regiones?
- ¿Hay equipos que consistentemente ganan por márgenes amplios?
- ¿El sexo del equipo (en caso de incluir masculino en el futuro) influye en la distribución de goles?

## 📊 Visualizaciones y análisis

Se han generado visualizaciones que incluyen:

- Gráficos de barras con goles totales por equipo.
- Gráficos de líneas con evolución de goles por fecha.
- Gráficos de cajas (boxplot) para analizar la distribución de goles por equipo.
- Visualizaciones multivariadas que combinan equipo, goles y tipo de victoria.

Además, se han creado nuevas columnas en el dataset:

- `Resultado Partido`: nombre del equipo ganador o empate.
- `Diferencia de Goles`: diferencia absoluta entre los goles anotados por cada equipo.
- `Tipo de Victoria`: clasificada como "Ajustada", "Amplia" o "Empate".

## 🧼 Diagnóstico de valores perdidos

Se ha realizado un análisis de valores nulos en el dataset para garantizar la calidad de los datos. Las columnas con valores faltantes han sido identificadas y tratadas según corresponda.

## 🛠️ Herramientas utilizadas

- Python
- Pandas
- NumPy
- Plotly
- Google Colab / Jupyter Notebook

## 📁 Estructura del proyecto

- `notebook.ipynb`: notebook principal con análisis y visualizaciones.
- `README.md`: descripción del proyecto.
- `Handball_W_InternationalResults.csv`: dataset utilizado.

---

Este proyecto forma parte del desafío “Visualizaciones en Python” y busca aplicar técnicas de análisis exploratorio de datos para responder preguntas relevantes en el contexto deportivo.

