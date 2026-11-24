# -*- coding: utf-8 -*-
"""
Aplicación web interactiva de siniestralidad vial en Cartagena de Indias.

Ejecución local recomendada:
    python main.py dashboard

También se puede ejecutar directamente con:
    streamlit run ui/app.py

La aplicación consume data/accidentes_cartagena_limpio.csv, generado por
main.py. Usa pandas, plotly.express y scikit-learn.
"""

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split


RUTA_DATOS = Path(__file__).resolve().parent.parent / "data" / "accidentes_cartagena_limpio.csv"

FUENTE = (
    "Alcaldía Mayor de Cartagena - Datos Abiertos Colombia, código "
    "5z62-bn99, actualizado el 14 de octubre de 2026."
)

OBJETIVO = "gravedad"
VARIABLES_PREDICTORAS = [
    "clase_accidente",
    "choque_con",
    "objeto_fijo",
    "hipotesis_1",
]

COLORES_GRAVEDAD = {
    "Con Heridos": "#2E86AB",
    "Con Muertos": "#D1495B",
}

DIAS_ORDEN = [
    "Lunes",
    "Martes",
    "Miércoles",
    "Jueves",
    "Viernes",
    "Sábado",
    "Domingo",
]


# ---------------------------------------------------------------------------
# Carga y modelo
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def cargar_datos(ruta: Path) -> pd.DataFrame:
    """Carga la base limpia y normaliza la fecha."""
    df = pd.read_csv(ruta, encoding="utf-8-sig")
    df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")
    return df


@st.cache_resource(show_spinner="Entrenando el modelo...")
def entrenar_modelo(df: pd.DataFrame) -> dict:
    """Entrena un RandomForest y devuelve modelo, métricas y codificación."""
    datos = df.dropna(subset=VARIABLES_PREDICTORAS + [OBJETIVO]).copy()

    X = datos[VARIABLES_PREDICTORAS].astype(str)
    y = datos[OBJETIVO].astype(str)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    # Codificación one-hot. Se ajusta con train y se alinea con test.
    X_train_encoded = pd.get_dummies(X_train, columns=VARIABLES_PREDICTORAS).astype(int)
    X_test_encoded = (
        pd.get_dummies(X_test, columns=VARIABLES_PREDICTORAS)
        .reindex(columns=X_train_encoded.columns, fill_value=0)
        .astype(int)
    )

    modelo = RandomForestClassifier(
        n_estimators=50,
        max_depth=5,
        random_state=42,
    )
    modelo.fit(X_train_encoded, y_train)

    y_pred = modelo.predict(X_test_encoded)
    clases = [str(clase) for clase in modelo.classes_]

    # Importancia agregada por variable original (suma de sus dummies).
    importancias = pd.Series(
        modelo.feature_importances_,
        index=X_train_encoded.columns,
    )
    importancia_por_variable = {}
    for variable in VARIABLES_PREDICTORAS:
        columnas_variable = [
            columna
            for columna in X_train_encoded.columns
            if columna.startswith(f"{variable}_")
        ]
        importancia_por_variable[variable] = float(
            importancias[columnas_variable].sum()
        )
    importancias_variables = pd.Series(
        importancia_por_variable
    ).sort_values(ascending=False)

    return {
        "modelo": modelo,
        "columnas": list(X_train_encoded.columns),
        "clases": clases,
        "matriz_confusion": confusion_matrix(y_test, y_pred, labels=clases),
        "accuracy": accuracy_score(y_test, y_pred),
        "f1_weighted": f1_score(y_test, y_pred, average="weighted"),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "importancias": importancias_variables,
    }


# ---------------------------------------------------------------------------
# Utilidades del dashboard
# ---------------------------------------------------------------------------
def _porcentaje(parte: float, total: float) -> float:
    """Calcula un porcentaje seguro frente a denominadores en cero."""
    return 100.0 * parte / total if total else 0.0


def _con_hora(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega la hora numérica (0-23) a partir de la columna 'hora'."""
    return df.assign(
        hora_num=pd.to_datetime(
            df["hora"], format="%H:%M", errors="coerce"
        ).dt.hour
    )


def _agregar_letalidad(df: pd.DataFrame, columna: str) -> pd.DataFrame:
    """Agrupa por una columna y calcula casos, muertos y letalidad."""
    base = df.assign(_es_muerto=df[OBJETIVO].eq("Con Muertos"))
    resumen = (
        base.groupby(columna, dropna=False)
        .agg(casos=(OBJETIVO, "size"), muertos=("_es_muerto", "sum"))
        .reset_index()
    )
    resumen["letalidad"] = 100 * resumen["muertos"] / resumen["casos"]
    return resumen


def _mostrar(fig, caption: str = "") -> None:
    """Muestra una figura de Plotly con un pie de gráfico opcional."""
    st.plotly_chart(fig, width="stretch")
    if caption:
        st.caption(caption)


def _indice_opcion(opciones: list[str], valor: str) -> int:
    """Devuelve el índice de un valor en una lista de opciones."""
    return opciones.index(valor) if valor in opciones else 0


def filtrar_datos(
    df: pd.DataFrame,
    gravedades: list[str],
    clases: list[str],
    localidades: list[str],
    barrios: list[str]
) -> pd.DataFrame:
    """Aplica los filtros de la barra lateral. Lista vacía = sin filtro."""
    filtrado = df.copy()
    if gravedades:
        filtrado = filtrado[filtrado[OBJETIVO].isin(gravedades)]
    if clases:
        filtrado = filtrado[filtrado["clase_accidente"].isin(clases)]
    if localidades:
        filtrado = filtrado[
            filtrado["localidad"].astype(str).isin(localidades)
        ]
    if barrios:
        filtrado = filtrado[
            filtrado["BARRIO"].astype(str).isin(barrios)
        ]
    return filtrado


def mostrar_pie_fuente() -> None:
    """Pie de página con fuente y advertencia sobre hipotesis_1."""
    st.markdown("---")
    st.caption(f"**Fuente:** {FUENTE}")
    st.caption(
        "**Nota:** `hipotesis_1` corresponde a apreciaciones iniciales del "
        "agente de tránsito y no a causas comprobadas del siniestro."
    )


# ---------------------------------------------------------------------------
# Gráficos del dashboard
# ---------------------------------------------------------------------------
def _grafico_gravedad(df: pd.DataFrame) -> None:
    distribucion = (
        df[OBJETIVO]
        .value_counts()
        .rename_axis(OBJETIVO)
        .reset_index(name="casos")
    )
    fig = px.pie(
        distribucion,
        names=OBJETIVO,
        values="casos",
        hole=0.45,
        title="Distribución de siniestros por gravedad",
        color=OBJETIVO,
        color_discrete_map=COLORES_GRAVEDAD,
    )
    fig.update_traces(textinfo="label+value+percent", textposition="inside")
    fig.update_layout(
        legend_title_text="Gravedad",
        height=380,
        margin=dict(t=60, b=20, l=20, r=20),
    )
    _mostrar(fig, "Valores absolutos y porcentaje sobre el total filtrado.")


def _grafico_clase_letalidad(df: pd.DataFrame) -> None:
    clase = _agregar_letalidad(df, "clase_accidente").sort_values(
        "letalidad", ascending=False
    )
    fig = px.bar(
        clase,
        x="clase_accidente",
        y="letalidad",
        text=clase["letalidad"].map(lambda valor: f"{valor:.1f}%"),
        custom_data=["casos", "muertos"],
        labels={
            "clase_accidente": "Clase de accidente",
            "letalidad": "Letalidad (%)",
        },
        title="Letalidad por clase de accidente",
        color="clase_accidente",
        color_discrete_sequence=px.colors.qualitative.Safe,
    )
    fig.update_traces(
        textposition="outside",
        hovertemplate=(
            "<b>%{x}</b><br>"
            "Letalidad: %{y:.1f}%<br>"
            "Casos: %{customdata[0]}<br>"
            "Con muertos: %{customdata[1]}<extra></extra>"
        ),
    )
    fig.update_yaxes(ticksuffix="%", title="Letalidad (%)")
    fig.update_layout(
        showlegend=False,
        height=380,
        margin=dict(t=60, b=20, l=20, r=20),
    )

    atropello = clase.loc[clase["clase_accidente"].eq("Atropello"), "letalidad"]
    choque = clase.loc[clase["clase_accidente"].eq("Choque"), "letalidad"]
    atropello_txt = f"{atropello.iloc[0]:.1f}%" if not atropello.empty else "sin datos"
    choque_txt = f"{choque.iloc[0]:.1f}%" if not choque.empty else "sin datos"
    _mostrar(
        fig,
        f"Atropello: {atropello_txt}; Choque: {choque_txt}. "
        "Las clases con pocos casos deben interpretarse con cautela.",
    )


def _grafico_clase_gravedad(df: pd.DataFrame) -> None:
    cruce = (
        df.groupby(["clase_accidente", OBJETIVO])
        .size()
        .reset_index(name="casos")
    )
    cruce["porcentaje"] = cruce.groupby("clase_accidente")["casos"].transform(
        lambda serie: 100 * serie / serie.sum()
    )
    cruce["porcentaje_texto"] = cruce["porcentaje"].map(lambda valor: f"{valor:.1f}%")
    cruce["etiqueta"] = cruce.apply(
        lambda fila: f"{int(fila['casos']):,} ({fila['porcentaje']:.1f}%)",
        axis=1,
    )

    fig = px.bar(
        cruce,
        x="clase_accidente",
        y="casos",
        color=OBJETIVO,
        barmode="group",
        text="etiqueta",
        custom_data=["porcentaje_texto"],
        labels={
            "clase_accidente": "Clase de accidente",
            "casos": "Número de siniestros",
            OBJETIVO: "Gravedad",
        },
        title="Siniestros por clase de accidente y gravedad",
        color_discrete_map=COLORES_GRAVEDAD,
    )
    fig.update_traces(
        textposition="outside",
        textfont_size=10,
        cliponaxis=False,
        hovertemplate=(
            "<b>%{x}</b><br>"
            "Gravedad: %{fullData.name}<br>"
            "Casos: %{y}<br>"
            "Porcentaje dentro de la clase: %{customdata[0]}"
            "<extra></extra>"
        ),
    )
    fig.update_layout(
        xaxis_tickangle=-20,
        legend_title_text="Gravedad",
        height=430,
        margin=dict(t=60, b=80, l=20, r=20),
    )
    _mostrar(
        fig,
        "Las etiquetas muestran el conteo y el porcentaje de cada gravedad "
        "dentro de la clase de accidente.",
    )


def _grafico_anual(df: pd.DataFrame) -> None:
    por_anio = (
        _agregar_letalidad(df, "anio")
        .dropna(subset=["anio"])
        .sort_values("anio")
    )
    if por_anio.empty:
        st.info("No hay años válidos en la selección.")
        return

    por_anio["anio"] = por_anio["anio"].astype(int)
    fig = px.bar(
        por_anio,
        x="anio",
        y="casos",
        text="casos",
        labels={"anio": "Año", "casos": "Número de siniestros"},
        title="Evolución anual de siniestros y letalidad",
        color_discrete_sequence=["#2E86AB"],
    )
    fig.add_scatter(
        x=por_anio["anio"],
        y=por_anio["letalidad"],
        mode="lines+markers+text",
        text=[f"{valor:.1f}%" for valor in por_anio["letalidad"]],
        textposition="top center",
        name="Letalidad (%)",
        line=dict(color="#D1495B", width=3),
        marker=dict(size=8),
        yaxis="y2",
    )
    fig.update_traces(textposition="outside", selector=dict(type="bar"))
    limite = por_anio["letalidad"].max()
    limite = 0 if pd.isna(limite) else max(limite * 1.35, 1)
    fig.update_layout(
        yaxis=dict(title="Número de siniestros"),
        yaxis2=dict(
            title="Letalidad (%)",
            overlaying="y",
            side="right",
            tickformat=".1f",
            range=[0, limite],
        ),
        legend=dict(orientation="h", y=1.12, x=0),
        height=390,
        margin=dict(t=70, b=20, l=20, r=20),
    )

    primero = por_anio.iloc[0]
    ultimo = por_anio.iloc[-1]
    variacion = _porcentaje(
        ultimo["casos"] - primero["casos"], primero["casos"]
    )
    _mostrar(
        fig,
        f"{int(primero['anio'])}: {int(primero['casos']):,} casos "
        f"({primero['letalidad']:.1f}% letalidad); "
        f"{int(ultimo['anio'])}: {int(ultimo['casos']):,} casos "
        f"({ultimo['letalidad']:.1f}%). Variación de casos: {variacion:+.1f}%.",
    )


def _grafico_dia(df: pd.DataFrame) -> None:
    dia = _agregar_letalidad(df, "dia_semana")
    if dia.empty:
        st.info("No hay datos por día de la semana.")
        return

    dia["dia_semana"] = pd.Categorical(
        dia["dia_semana"], categories=DIAS_ORDEN, ordered=True
    )
    dia = dia.sort_values("dia_semana")

    dia_gravedad = (
        df.groupby(["dia_semana", OBJETIVO])
        .size()
        .reset_index(name="casos")
    )
    dia_gravedad["dia_semana"] = pd.Categorical(
        dia_gravedad["dia_semana"], categories=DIAS_ORDEN, ordered=True
    )
    dia_gravedad = dia_gravedad.sort_values("dia_semana")

    fig = px.bar(
        dia_gravedad,
        x="dia_semana",
        y="casos",
        color=OBJETIVO,
        barmode="stack",
        category_orders={"dia_semana": DIAS_ORDEN},
        labels={
            "dia_semana": "Día de la semana",
            "casos": "Número de siniestros",
            OBJETIVO: "Gravedad",
        },
        title="Siniestros por día de la semana y letalidad",
        color_discrete_map=COLORES_GRAVEDAD,
    )
    fig.add_scatter(
        x=dia["dia_semana"].astype(str),
        y=dia["letalidad"],
        mode="lines+markers+text",
        text=[f"{valor:.1f}%" for valor in dia["letalidad"]],
        textposition="top center",
        name="Letalidad (%)",
        line=dict(color="#333333", width=2),
        marker=dict(size=7),
        yaxis="y2",
    )
    limite = dia["letalidad"].max()
    limite = 0 if pd.isna(limite) else max(limite * 1.35, 1)
    fig.update_layout(
        yaxis=dict(title="Número de siniestros"),
        yaxis2=dict(
            title="Letalidad (%)",
            overlaying="y",
            side="right",
            tickformat=".1f",
            range=[0, limite],
        ),
        legend=dict(orientation="h", y=1.12, x=0),
        height=390,
        margin=dict(t=70, b=40, l=20, r=20),
        xaxis_tickangle=-20,
    )

    dia_max = dia.loc[dia["casos"].idxmax()]
    _mostrar(
        fig,
        f"{dia_max['dia_semana']}: {int(dia_max['casos']):,} casos "
        f"({_porcentaje(dia_max['casos'], len(df)):.1f}%) y letalidad "
        f"{dia_max['letalidad']:.1f}%.",
    )


def _grafico_hora(df: pd.DataFrame) -> None:
    temporal = _con_hora(df).dropna(subset=["hora_num"]).copy()
    if temporal.empty:
        st.info("No hay horas válidas en la selección.")
        return

    por_hora = _agregar_letalidad(temporal, "hora_num")
    por_hora = (
        pd.DataFrame({"hora_num": range(24)})
        .merge(por_hora, on="hora_num", how="left")
    )
    por_hora["casos"] = por_hora["casos"].fillna(0).astype(int)
    por_hora["muertos"] = por_hora["muertos"].fillna(0).astype(int)
    por_hora["letalidad"] = (
        100 * por_hora["muertos"] / por_hora["casos"].replace(0, pd.NA)
    )
    por_hora["etiqueta"] = por_hora["casos"].map(
        lambda valor: f"{int(valor)}" if valor >= 250 else ""
    )

    fig = px.bar(
        por_hora,
        x="hora_num",
        y="casos",
        text="etiqueta",
        labels={
            "hora_num": "Hora del día",
            "casos": "Número de siniestros",
        },
        title="Siniestros y letalidad por hora del día",
        color_discrete_sequence=["#2E86AB"],
    )
    fig.add_scatter(
        x=por_hora["hora_num"],
        y=por_hora["letalidad"],
        mode="lines+markers",
        name="Letalidad (%)",
        line=dict(color="#D1495B", width=2),
        marker=dict(size=6),
        yaxis="y2",
    )
    fig.add_vrect(
        x0=-0.5,
        x1=5.5,
        fillcolor="#D1495B",
        opacity=0.08,
        line_width=0,
        annotation_text="Madrugada",
        annotation_position="top left",
    )
    limite = por_hora["letalidad"].max()
    limite = 0 if pd.isna(limite) else max(limite * 1.3, 1)
    fig.update_layout(
        yaxis=dict(title="Número de siniestros"),
        yaxis2=dict(
            title="Letalidad (%)",
            overlaying="y",
            side="right",
            tickformat=".1f",
            range=[0, limite],
        ),
        legend=dict(orientation="h", y=1.12, x=0),
        height=390,
        margin=dict(t=70, b=20, l=20, r=20),
    )
    fig.update_xaxes(
        tickmode="array",
        tickvals=list(range(0, 24, 2)),
        title="Hora del día",
    )

    pico = por_hora.loc[por_hora["casos"].idxmax()]
    madrugada = temporal[temporal["hora_num"] < 6]
    pct_madrugada = _porcentaje(len(madrugada), len(temporal))
    letalidad_madrugada = _porcentaje(
        madrugada[OBJETIVO].eq("Con Muertos").sum(), len(madrugada)
    )
    _mostrar(
        fig,
        f"Pico: {int(pico['hora_num']):02d}:00 ({int(pico['casos'])} casos). "
        f"Madrugada (00:00-05:59): {len(madrugada):,} casos "
        f"({pct_madrugada:.1f}%) y letalidad {letalidad_madrugada:.1f}%.",
    )


def _grafico_mapa_calor(df: pd.DataFrame) -> None:
    temporal = _con_hora(df).dropna(subset=["hora_num"])
    if temporal.empty:
        st.info("No hay datos temporales para el mapa de calor.")
        return

    heat = (
        pd.crosstab(temporal["dia_semana"], temporal["hora_num"])
        .reindex(DIAS_ORDEN)
        .reindex(columns=range(24), fill_value=0)
    )
    fig = px.imshow(
        heat,
        aspect="auto",
        labels={
            "x": "Hora del día",
            "y": "Día de la semana",
            "color": "Siniestros",
        },
        title="Mapa de calor: día de la semana y hora",
        color_continuous_scale="YlOrRd",
    )
    fig.update_xaxes(
        tickmode="array",
        tickvals=list(range(0, 24, 2)),
        title="Hora del día",
    )
    fig.update_yaxes(title="Día de la semana")
    fig.update_layout(height=390, margin=dict(t=60, b=20, l=20, r=20))

    pila = heat.stack()
    if not pila.empty:
        dia_max, hora_max = pila.idxmax()
        caption = (
            f"Mayor concentración: {dia_max} a las {int(hora_max):02d}:00 "
            f"({int(pila.max())} siniestros)."
        )
    else:
        caption = "Sin datos temporales."
    _mostrar(fig, caption)


def _grafico_barrios(df: pd.DataFrame) -> None:
    barrios = _agregar_letalidad(df, "BARRIO")
    if barrios.empty:
        st.info("No hay barrios para graficar.")
        return

    barrios["participacion"] = 100 * barrios["casos"] / len(df)
    top = barrios.nlargest(10, "casos").sort_values("casos")

    fig = px.bar(
        top,
        x="casos",
        y="BARRIO",
        orientation="h",
        text="casos",
        custom_data=["participacion", "letalidad"],
        labels={"casos": "Número de siniestros", "BARRIO": "Barrio"},
        title="Top 10 barrios por número de siniestros",
        color_discrete_sequence=["#2E86AB"],
    )
    fig.update_traces(
        textposition="outside",
        hovertemplate=(
            "<b>%{y}</b><br>"
            "Siniestros: %{x}<br>"
            "Participación: %{customdata[0]:.2f}%<br>"
            "Letalidad: %{customdata[1]:.1f}%<extra></extra>"
        ),
    )
    fig.update_layout(height=400, margin=dict(t=60, b=20, l=20, r=40))

    barrio_max = barrios.loc[barrios["casos"].idxmax()]
    _mostrar(
        fig,
        f"{barrio_max['BARRIO']}: {int(barrio_max['casos']):,} casos "
        f"({barrio_max['participacion']:.2f}%) y letalidad "
        f"{barrio_max['letalidad']:.1f}%.",
    )


def _grafico_localidad(df: pd.DataFrame) -> None:
    localidades = _agregar_letalidad(df, "localidad")
    if localidades.empty:
        st.info("No hay localidades para graficar.")
        return

    localidades["localidad_etiqueta"] = (
        "Localidad " + localidades["localidad"].astype(str)
    )
    localidades = localidades.sort_values("localidad_etiqueta")

    fig = px.bar(
        localidades,
        x="localidad_etiqueta",
        y="casos",
        text="casos",
        labels={
            "localidad_etiqueta": "Localidad",
            "casos": "Número de siniestros",
        },
        title="Siniestros y letalidad por localidad",
        color_discrete_sequence=["#2E86AB"],
    )
    fig.add_scatter(
        x=localidades["localidad_etiqueta"],
        y=localidades["letalidad"],
        mode="lines+markers+text",
        text=[f"{valor:.1f}%" for valor in localidades["letalidad"]],
        textposition="top center",
        name="Letalidad (%)",
        line=dict(color="#D1495B", width=2),
        marker=dict(size=8),
        yaxis="y2",
    )
    limite = localidades["letalidad"].max()
    limite = 0 if pd.isna(limite) else max(limite * 1.35, 1)
    fig.update_layout(
        yaxis=dict(title="Número de siniestros"),
        yaxis2=dict(
            title="Letalidad (%)",
            overlaying="y",
            side="right",
            tickformat=".1f",
            range=[0, limite],
        ),
        legend=dict(orientation="h", y=1.12, x=0),
        height=400,
        margin=dict(t=60, b=20, l=20, r=20),
    )

    partes = [
        f"{fila['localidad_etiqueta']}: "
        f"{_porcentaje(fila['casos'], len(df)):.2f}% de los casos y "
        f"letalidad {fila['letalidad']:.2f}%"
        for _, fila in localidades.iterrows()
    ]
    _mostrar(fig, "; ".join(partes) + ".")


def _grafico_mapa(df: pd.DataFrame) -> None:
    mapa = df.dropna(subset=["latitud", "longitud"]).copy()
    if mapa.empty:
        st.info("No hay coordenadas válidas en la selección.")
        return

    mapa["fecha_texto"] = mapa["fecha"].dt.strftime("%Y-%m-%d")
    mapa["localidad_texto"] = "Localidad " + mapa["localidad"].astype(str)
    fig = px.scatter_map(
        mapa,
        lat="latitud",
        lon="longitud",
        color=OBJETIVO,
        color_discrete_map=COLORES_GRAVEDAD,
        hover_data={
            "BARRIO": True,
            "localidad_texto": True,
            "clase_accidente": True,
            "fecha_texto": True,
            "hora": True,
            "latitud": False,
            "longitud": False,
        },
        zoom=10,
        center={
            "lat": float(mapa["latitud"].median()),
            "lon": float(mapa["longitud"].median()),
        },
        map_style="open-street-map",
        title="Georreferenciación de los siniestros",
        height=540,
    )
    fig.update_traces(marker=dict(size=6, opacity=0.45))
    _mostrar(
        fig,
        f"{len(mapa):,} de {len(df):,} siniestros con coordenadas válidas "
        f"({_porcentaje(len(mapa), len(df)):.1f}%).",
    )


def _grafico_hipotesis_top(df: pd.DataFrame) -> None:
    top = (
        df.loc[df["hipotesis_1"].ne("Sin registro"), "hipotesis_1"]
        .value_counts()
        .head(10)
        .rename_axis("hipotesis_1")
        .reset_index(name="casos")
    )
    if top.empty:
        st.info("No hay códigos de hipotesis_1 diligenciados en la selección.")
        return

    fig = px.bar(
        top.sort_values("casos"),
        x="casos",
        y="hipotesis_1",
        orientation="h",
        text="casos",
        labels={
            "casos": "Número de siniestros",
            "hipotesis_1": "Código de hipotesis_1",
        },
        title="Top 10 códigos de hipotesis_1 más frecuentes",
        color_discrete_sequence=["#2E86AB"],
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(
        height=400,
        margin=dict(t=60, b=20, l=20, r=40),
        yaxis_title="Código de hipotesis_1",
        xaxis_title="Número de siniestros",
    )

    partes = [
        f"{fila['hipotesis_1']}: {int(fila['casos']):,} "
        f"({_porcentaje(fila['casos'], len(df)):.1f}%)"
        for _, fila in top.head(3).iterrows()
    ]
    _mostrar(
        fig,
        "Códigos más frecuentes: " + "; ".join(partes) + ". Se excluye "
        "'Sin registro'.",
    )


def _grafico_hipotesis_letalidad(df: pd.DataFrame) -> None:
    hipotesis = df.loc[df["hipotesis_1"].ne("Sin registro")]
    hipotesis = _agregar_letalidad(hipotesis, "hipotesis_1")
    hipotesis = (
        hipotesis.loc[hipotesis["casos"] >= 50]
        .nlargest(10, "letalidad")
        .sort_values("letalidad")
    )
    if hipotesis.empty:
        st.info(
            "No hay códigos de hipotesis_1 con al menos 50 casos en la "
            "selección."
        )
        return

    fig = px.bar(
        hipotesis,
        x="letalidad",
        y="hipotesis_1",
        orientation="h",
        text=hipotesis["letalidad"].map(lambda valor: f"{valor:.1f}%"),
        custom_data=["casos", "muertos"],
        labels={
            "letalidad": "Letalidad (%)",
            "hipotesis_1": "Código de hipotesis_1",
        },
        title="Letalidad por código de hipotesis_1 (mín. 50 casos)",
        color_discrete_sequence=["#D1495B"],
    )
    fig.update_traces(
        textposition="outside",
        hovertemplate=(
            "<b>Código %{y}</b><br>"
            "Letalidad: %{x:.1f}%<br>"
            "Casos: %{customdata[0]}<br>"
            "Con muertos: %{customdata[1]}<extra></extra>"
        ),
    )
    fig.update_xaxes(ticksuffix="%", title="Letalidad (%)")
    fig.update_layout(height=400, margin=dict(t=60, b=20, l=20, r=40))
    _mostrar(
        fig,
        "Solo códigos con al menos 50 siniestros. hipotesis_1 es una "
        "apreciación inicial del agente, no una causa comprobada.",
    )


# ---------------------------------------------------------------------------
# Pestaña 1: dashboard descriptivo
# ---------------------------------------------------------------------------
def render_dashboard(df: pd.DataFrame) -> None:
    st.subheader("Dashboard descriptivo")

    if df.empty:
        st.warning(
            "No hay siniestros que cumplan los filtros seleccionados. "
            "Cambia la gravedad o la clase de accidente en la barra lateral."
        )
        return

    total = len(df)
    n_muertos = int(df[OBJETIVO].eq("Con Muertos").sum())
    n_heridos = int(df[OBJETIVO].eq("Con Heridos").sum())
    pct_muertos = _porcentaje(n_muertos, total)
    pct_heridos = _porcentaje(n_heridos, total)

    conteo_clases = df["clase_accidente"].value_counts()
    clase_top = conteo_clases.idxmax()
    clase_top_casos = int(conteo_clases.max())

    hipotesis_validas = df.loc[
        df["hipotesis_1"].ne("Sin registro"), "hipotesis_1"
    ]
    if hipotesis_validas.empty:
        hipotesis_top = "Sin registro"
        hipotesis_top_casos = 0
    else:
        conteo_hipotesis = hipotesis_validas.value_counts()
        hipotesis_top = conteo_hipotesis.idxmax()
        hipotesis_top_casos = int(conteo_hipotesis.max())

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("Total de siniestros", f"{total:,}")
    kpi2.metric(
        "Siniestros con muertos",
        f"{pct_muertos:.2f}%",
        help=f"{n_muertos:,} de {total:,} siniestros filtrados.",
    )
    kpi3.metric(
        "Clase más frecuente",
        clase_top,
        help=f"{clase_top_casos:,} siniestros en la selección actual.",
    )
    kpi4.metric(
        "Hipotesis_1 predominante",
        hipotesis_top,
        help=(
            f"{hipotesis_top_casos:,} siniestros con ese código. "
            "Se excluye 'Sin registro'."
        ),
    )

    st.caption(
        f"Con heridos: {pct_heridos:.2f}% | Con muertos: {pct_muertos:.2f}% "
        f"(n = {total:,})."
    )
    st.caption(
        "Letalidad = porcentaje de siniestros con gravedad 'Con Muertos'. "
        "No es una tasa de fallecidos por lesionado, sino la proporción de "
        "eventos fatales registrados."
    )

    # Indicadores complementarios que aparecen en la tabla del informe.
    temporal = _con_hora(df)
    conteo_horas = temporal["hora_num"].value_counts()
    if conteo_horas.empty:
        hora_pico_txt = "Sin registro"
        hora_pico_casos = 0
    else:
        hora_pico = int(conteo_horas.idxmax())
        hora_pico_txt = f"{hora_pico:02d}:00"
        hora_pico_casos = int(conteo_horas.max())

    conteo_dias = df["dia_semana"].value_counts()
    dia_top = conteo_dias.idxmax()
    dia_top_casos = int(conteo_dias.max())

    conteo_localidades = df["localidad"].astype(str).value_counts()
    localidad_top = conteo_localidades.idxmax()
    localidad_top_casos = int(conteo_localidades.max())
    localidad_top_txt = (
        f"Localidad {localidad_top}"
        if localidad_top in {"1", "2", "3"}
        else localidad_top
    )

    conteo_barrios = df["BARRIO"].value_counts()
    barrio_top = conteo_barrios.idxmax()
    barrio_top_casos = int(conteo_barrios.max())

    st.markdown("#### Indicadores complementarios")
    extra1, extra2, extra3, extra4 = st.columns(4)
    extra1.metric(
        "Hora pico",
        hora_pico_txt,
        help=f"{hora_pico_casos:,} siniestros en esa hora.",
    )
    extra2.metric(
        "Día con más casos",
        dia_top,
        help=f"{dia_top_casos:,} siniestros en ese día.",
    )
    extra3.metric(
        "Localidad con más casos",
        localidad_top_txt,
        help=f"{localidad_top_casos:,} siniestros en esa localidad.",
    )
    extra4.metric(
        "Barrio con más casos",
        barrio_top,
        help=f"{barrio_top_casos:,} siniestros en ese barrio.",
    )
    st.caption("Indicadores calculados sobre la selección actual.")

    # -----------------------------------------------------------------------
    # Frecuencia y gravedad
    # -----------------------------------------------------------------------
    st.markdown("### Frecuencia y gravedad")
    columna1, columna2 = st.columns(2)
    with columna1:
        _grafico_gravedad(df)
    with columna2:
        _grafico_clase_letalidad(df)
    _grafico_clase_gravedad(df)

    # -----------------------------------------------------------------------
    # Temporalidad
    # -----------------------------------------------------------------------
    st.markdown("### Temporalidad")
    columna1, columna2 = st.columns(2)
    with columna1:
        _grafico_anual(df)
    with columna2:
        _grafico_dia(df)

    columna1, columna2 = st.columns(2)
    with columna1:
        _grafico_hora(df)
    with columna2:
        _grafico_mapa_calor(df)

    # -----------------------------------------------------------------------
    # Territorio
    # -----------------------------------------------------------------------
    st.markdown("### Territorio")
    columna1, columna2 = st.columns(2)
    with columna1:
        _grafico_barrios(df)
    with columna2:
        _grafico_localidad(df)
    _grafico_mapa(df)

    # -----------------------------------------------------------------------
    # Hipótesis iniciales
    # -----------------------------------------------------------------------
    st.markdown("### Hipótesis iniciales")
    columna1, columna2 = st.columns(2)
    with columna1:
        _grafico_hipotesis_top(df)
    with columna2:
        _grafico_hipotesis_letalidad(df)

    mostrar_pie_fuente()


# ---------------------------------------------------------------------------
# Pestaña 2: componente predictivo
# ---------------------------------------------------------------------------
def render_modelo(df: pd.DataFrame) -> None:
    st.subheader("Componente predictivo")
    st.markdown(
        "Este componente usa un modelo supervisado para estimar si un "
        "siniestro terminará con **heridos** o con **muertos** a partir de "
        "cuatro características registradas en el parte inicial."
    )

    total = len(df)
    pct_heridos = _porcentaje(int(df[OBJETIVO].eq("Con Heridos").sum()), total)
    pct_muertos = _porcentaje(int(df[OBJETIVO].eq("Con Muertos").sum()), total)

    st.markdown("#### 1. ¿Qué intenta predecir?")
    columna1, columna2 = st.columns([1.1, 1])
    with columna1:
        st.markdown(
            "- **Variable objetivo (y):** `gravedad`.\n"
            "- **Variables predictoras (X):** `clase_accidente`, "
            "`choque_con`, `objeto_fijo` e `hipotesis_1`.\n"
            "- **Modelo:** `RandomForestClassifier(n_estimators=50, "
            "max_depth=5)`.\n"
            "- **Partición:** 80% entrenamiento y 20% prueba, con "
            "`random_state=42`."
        )
    with columna2:
        st.info(
            f"La base completa está desbalanceada: **Con Heridos "
            f"{pct_heridos:.1f}%** y **Con Muertos {pct_muertos:.1f}%**. "
            "Por eso la exactitud puede ser alta aunque el modelo falle en la "
            "clase minoritaria; el F1-Score ponderado ayuda a matizar esa "
            "lectura."
        )

    st.markdown(
        "Las variables categóricas se transforman con **one-hot encoding** "
        "(`pd.get_dummies`). El modelo se entrena con la base completa, no con "
        "los filtros de la barra lateral, para mantener estable la evaluación."
    )

    artefactos = entrenar_modelo(df)

    st.markdown("#### 2. Desempeño del modelo")
    metrica1, metrica2, metrica3 = st.columns(3)
    metrica1.metric("Exactitud (Accuracy)", f"{artefactos['accuracy']:.2%}")
    metrica2.metric("F1-Score ponderado", f"{artefactos['f1_weighted']:.3f}")
    metrica3.metric(
        "Entrenamiento / prueba",
        f"{artefactos['n_train']:,} / {artefactos['n_test']:,}",
    )

    st.markdown(
        "- **Exactitud (Accuracy):** porcentaje de casos del conjunto de "
        "prueba clasificados correctamente. Es intuitiva, pero puede ser "
        "optimista cuando una clase domina.\n"
        "- **F1-Score ponderado:** combina precisión y recall y pondera por "
        "el tamaño de cada clase. Es más informativo con clases "
        "desbalanceadas."
    )

    st.markdown("##### Matriz de confusión")
    st.markdown(
        "Las **filas** representan la gravedad real y las **columnas** la "
        "gravedad predicha. La **diagonal** son los aciertos; los valores "
        "fuera de la diagonal son errores. Por ejemplo, un siniestro real "
        "'Con Muertos' predicho como 'Con Heridos' aparece en la fila "
        "'Con Muertos' y en la columna 'Con Heridos'."
    )
    fig_matriz = px.imshow(
        artefactos["matriz_confusion"],
        text_auto=True,
        x=artefactos["clases"],
        y=artefactos["clases"],
        labels={
            "x": "Gravedad predicha",
            "y": "Gravedad real",
            "color": "Casos",
        },
        title="Matriz de confusión sobre el conjunto de prueba",
        color_continuous_scale="Blues",
        aspect="auto",
    )
    fig_matriz.update_layout(margin=dict(t=60, b=20, l=20, r=20))
    st.plotly_chart(fig_matriz, width="stretch")

    st.markdown("#### 3. ¿En qué se fija el modelo?")
    importancias = (
        artefactos["importancias"]
        .rename_axis("Variable")
        .reset_index(name="Importancia")
    )
    fig_importancia = px.bar(
        importancias,
        x="Importancia",
        y="Variable",
        orientation="h",
        text=importancias["Importancia"].map(lambda valor: f"{valor:.1%}"),
        labels={
            "Importancia": "Importancia relativa",
            "Variable": "Variable predictora",
        },
        title="Importancia relativa de cada variable",
        color_discrete_sequence=["#2E86AB"],
    )
    fig_importancia.update_traces(textposition="outside")
    fig_importancia.update_xaxes(tickformat=".0%", title="Importancia relativa")
    fig_importancia.update_layout(
        height=350,
        margin=dict(t=60, b=20, l=20, r=40),
        yaxis_title="",
    )
    st.plotly_chart(fig_importancia, width="stretch")
    st.caption(
        "La importancia indica cuánto contribuye cada variable a las "
        "decisiones del bosque; no implica causalidad ni una relación "
        "comprobada."
    )

    st.markdown("#### 4. Simulador de escenarios")
    st.markdown(
        "Selecciona las características de un siniestro hipotético y pulsa "
        "**Estimar probabilidad**. El modelo devolverá la probabilidad "
        "estimada de cada nivel de gravedad. La etiqueta con mayor "
        "probabilidad es la predicción del modelo, pero no una certeza."
    )
    opciones = {
        columna: sorted(df[columna].dropna().astype(str).unique())
        for columna in VARIABLES_PREDICTORAS
    }
    valores_iniciales = {
        "clase_accidente": "Choque",
        "choque_con": "Vehiculo",
        "objeto_fijo": "No aplica",
        "hipotesis_1": "139",
    }

    with st.form("formulario_prediccion"):
        columna1, columna2 = st.columns(2)
        with columna1:
            clase_accidente = st.selectbox(
                "Clase de accidente",
                opciones["clase_accidente"],
                index=_indice_opcion(
                    opciones["clase_accidente"],
                    valores_iniciales["clase_accidente"],
                ),
            )
            objeto_fijo = st.selectbox(
                "Objeto fijo",
                opciones["objeto_fijo"],
                index=_indice_opcion(
                    opciones["objeto_fijo"],
                    valores_iniciales["objeto_fijo"],
                ),
            )
        with columna2:
            choque_con = st.selectbox(
                "Choque con",
                opciones["choque_con"],
                index=_indice_opcion(
                    opciones["choque_con"],
                    valores_iniciales["choque_con"],
                ),
            )
            hipotesis_1 = st.selectbox(
                "Código de hipotesis_1",
                opciones["hipotesis_1"],
                index=_indice_opcion(
                    opciones["hipotesis_1"],
                    valores_iniciales["hipotesis_1"],
                ),
            )

        enviado = st.form_submit_button("Estimar probabilidad")

    if enviado:
        entrada = pd.DataFrame(
            [
                {
                    "clase_accidente": clase_accidente,
                    "choque_con": choque_con,
                    "objeto_fijo": objeto_fijo,
                    "hipotesis_1": hipotesis_1,
                }
            ]
        )
        entrada_encoded = (
            pd.get_dummies(entrada, columns=VARIABLES_PREDICTORAS)
            .reindex(columns=artefactos["columnas"], fill_value=0)
            .astype(int)
        )
        probabilidades = artefactos["modelo"].predict_proba(entrada_encoded)[0]

        resultado = (
            pd.DataFrame(
                {
                    "Gravedad": artefactos["clases"],
                    "Probabilidad": probabilidades,
                }
            )
            .sort_values("Probabilidad", ascending=False)
            .reset_index(drop=True)
        )
        resultado["Probabilidad (%)"] = (
            resultado["Probabilidad"] * 100
        ).round(2)

        st.caption(
            f"Escenario simulado: clase={clase_accidente}; "
            f"choque_con={choque_con}; objeto_fijo={objeto_fijo}; "
            f"hipotesis_1={hipotesis_1}."
        )
        st.markdown("**Probabilidad estimada por nivel de gravedad:**")
        st.dataframe(
            resultado[["Gravedad", "Probabilidad (%)"]],
            hide_index=True,
            width="stretch",
        )

        prediccion = resultado.iloc[0]
        st.success(
            f"Mayor probabilidad: **{prediccion['Gravedad']}** "
            f"({prediccion['Probabilidad']:.1%})."
        )

        fig_probabilidad = px.bar(
            resultado,
            x="Gravedad",
            y="Probabilidad",
            text=resultado["Probabilidad"].map(lambda valor: f"{valor:.1%}"),
            labels={
                "Gravedad": "Nivel de gravedad",
                "Probabilidad": "Probabilidad estimada",
            },
            title="Probabilidad estimada para el siniestro simulado",
            color="Gravedad",
            color_discrete_map=COLORES_GRAVEDAD,
        )
        fig_probabilidad.update_traces(textposition="outside")
        fig_probabilidad.update_yaxes(tickformat=".0%", range=[0, 1])
        fig_probabilidad.update_layout(
            showlegend=False,
            margin=dict(t=60, b=20, l=20, r=20),
        )
        st.plotly_chart(fig_probabilidad, width="stretch")
        st.caption(
            "La probabilidad más alta no implica certeza: es una estimación "
            "estadística del modelo para las características elegidas."
        )

    st.markdown("#### 5. Limitaciones")
    st.warning(
        "Limitaciones del modelo: la base está desbalanceada (hay muchos más "
        "siniestros con heridos que con muertos) y no incluye variables "
        "climáticas, de velocidad, estado de la vía o comportamiento humano. "
        "Por eso, las probabilidades son estimaciones estadísticas y no "
        "certezas absolutas."
    )


# ---------------------------------------------------------------------------
# Aplicación principal
# ---------------------------------------------------------------------------
def main() -> None:
    st.set_page_config(
        page_title="Análisis de Siniestralidad Vial en Cartagena de Indias",
        layout="wide",
    )

    st.title("Análisis de Siniestralidad Vial en Cartagena de Indias")
    st.markdown(
        "Tablero descriptivo y componente predictivo construido a partir de "
        "los registros de accidentes de tránsito de la ciudad."
    )

    if not RUTA_DATOS.exists():
        st.error(
            "No se encontró data/accidentes_cartagena_limpio.csv. "
            "Ejecuta primero `python main.py limpiar` para generar la base depurada."
        )
        st.stop()

    df = cargar_datos(RUTA_DATOS)

    # Barra lateral: contexto y filtros.
    st.sidebar.header("Contexto del problema")
    st.sidebar.write(
        "La siniestralidad vial en Cartagena de Indias afecta la movilidad y "
        "la seguridad de peatones, pasajeros y conductores. Este tablero "
        "resume los siniestros registrados y explora la probabilidad de que "
        "un evento termine con heridos o con muertos."
    )
    st.sidebar.caption(f"**Fuente:** {FUENTE}")
    st.sidebar.markdown("---")

    st.sidebar.header("Filtros")
    gravedades = sorted(df[OBJETIVO].dropna().unique())
    clases = sorted(df["clase_accidente"].dropna().unique())

    gravedades_seleccionadas = st.sidebar.multiselect(
        "Gravedad",
        options=gravedades,
        default=gravedades,
    )
    clases_seleccionadas = st.sidebar.multiselect(
        "Clase de accidente",
        options=clases,
        default=clases,
    )
    localidades = sorted(df["localidad"].dropna().astype(str).unique())
    localidades_seleccionadas = st.sidebar.multiselect(
        "Localidad",
        options=localidades,
        default=localidades,
        format_func=lambda valor: f"Localidad {valor}",
    )
    barrios = sorted(df["BARRIO"].dropna().unique())
    barrios_seleccionados = st.sidebar.multiselect(
        "Barrio",
        options=barrios,
        default=barrios,
    )
    st.sidebar.caption(
        "Si no seleccionas opciones, se muestran todos los registros. "
        "El filtro de localidad también actualiza el mapa."
    )

    df_filtrado = filtrar_datos(
        df,
        gravedades_seleccionadas,
        clases_seleccionadas,
        localidades_seleccionadas,
        barrios_seleccionados
    )
    st.sidebar.metric("Siniestros filtrados", f"{len(df_filtrado):,}")
    st.sidebar.caption(
        f"Periodo: {df['fecha'].min():%Y-%m-%d} a {df['fecha'].max():%Y-%m-%d}"
    )

    tab_descriptivo = st.tabs(
        ["Dashboard descriptivo"]
    )

    with tab_descriptivo:
        render_dashboard(df_filtrado)


if __name__ == "__main__":
    main()
