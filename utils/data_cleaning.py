# -*- coding: utf-8 -*-
"""Funciones de carga, auditoría y limpieza del dataset de accidentes."""

from pathlib import Path

import pandas as pd

from config.config import (
    COLUMNAS_ANALISIS,
    DIAS_SEMANA,
    FORMATO_FECHA,
    FORMATO_HORA,
    PATRON_DATOS_PERSONALES,
    RANGO_LATITUD,
    RANGO_LONGITUD,
    VIA_COLS,
)
from utils.text_utils import limpiar_codigo, limpiar_texto


# ---------------------------------------------------------------------------
# Carga y auditoría
# ---------------------------------------------------------------------------
def cargar_datos(ruta: Path) -> pd.DataFrame:
    """Carga el CSV original conservando el BOM y los tipos inferidos."""
    if not ruta.exists():
        raise FileNotFoundError(f"No se encontró el archivo de entrada:\n{ruta}")
    return pd.read_csv(ruta, encoding="utf-8-sig")


def auditar_datos(df: pd.DataFrame) -> dict:
    """Calcula la auditoría inicial sin imprimir en consola."""
    tabla_auditoria = pd.DataFrame(
        {
            "tipo_dato": df.dtypes.astype(str),
            "valores_nulos": df.isna().sum(),
        }
    )
    frecuencias = {
        columna: df[columna].value_counts(dropna=False).head(10)
        for columna in ["gravedad", "clase_accidente", "choque_con", "objeto_fijo"]
    }
    return {
        "n_inicial": int(df.shape[0]),
        "n_columnas_inicial": int(df.shape[1]),
        "nulos_iniciales": int(df.isna().sum().sum()),
        "duplicados_fila_completa": int(df.duplicated().sum()),
        "columnas": list(df.columns),
        "tabla_auditoria": tabla_auditoria,
        "frecuencias": frecuencias,
    }


# ---------------------------------------------------------------------------
# Privacidad, selección y duplicados
# ---------------------------------------------------------------------------
def verificar_privacidad(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Detecta y elimina columnas con posibles datos personales."""
    nombres_columnas = pd.Series(df.columns)
    columnas_sensibles = nombres_columnas[
        nombres_columnas.str.contains(
            PATRON_DATOS_PERSONALES, case=False, regex=True, na=False
        )
    ].tolist()

    if columnas_sensibles:
        df = df.drop(columns=columnas_sensibles)

    return df, columnas_sensibles


def seleccionar_columnas(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Conserva solo las variables solicitadas para el análisis."""
    faltantes = [columna for columna in COLUMNAS_ANALISIS if columna not in df.columns]
    if faltantes:
        raise KeyError(f"Faltan columnas esperadas en el archivo: {faltantes}")

    columnas_eliminadas = [
        columna for columna in df.columns if columna not in COLUMNAS_ANALISIS
    ]
    return df[COLUMNAS_ANALISIS].copy(), columnas_eliminadas


def eliminar_duplicados(
    df: pd.DataFrame, subset: list[str] | None = None
) -> tuple[pd.DataFrame, int]:
    """Elimina duplicados exactos y devuelve cuántos registros quitó."""
    n_duplicados = int(df.duplicated(subset=subset).sum())
    if n_duplicados > 0:
        df = df.drop_duplicates(subset=subset, keep="first").copy()
    return df, n_duplicados


# ---------------------------------------------------------------------------
# Tratamiento de nulos y centinelas
# ---------------------------------------------------------------------------
def _tratar_gravedad(df: pd.DataFrame, metricas: dict) -> pd.DataFrame:
    df["gravedad"] = limpiar_texto(df["gravedad"]).str.title()
    faltantes = df["gravedad"].isna() | df["gravedad"].eq("")
    metricas["sin_gravedad"] = int(faltantes.sum())
    if metricas["sin_gravedad"] > 0:
        df = df.loc[~faltantes].copy()
    return df


def _tratar_clase_accidente(df: pd.DataFrame, metricas: dict) -> pd.DataFrame:
    df["clase_accidente"] = limpiar_texto(df["clase_accidente"]).str.title()
    metricas["clase_sin_registro"] = int(df["clase_accidente"].isna().sum())
    df["clase_accidente"] = df["clase_accidente"].fillna("Sin registro")
    return df


def _tratar_choque_con(df: pd.DataFrame, metricas: dict) -> pd.DataFrame:
    choque = limpiar_texto(df["choque_con"]).str.title()
    metricas["choque_na"] = int(choque.isna().sum())
    metricas["choque_no"] = int(choque.str.upper().eq("NO").fillna(False).sum())
    metricas["choque_malformado"] = int(
        choque.str.upper().eq("CHOQUE CON").fillna(False).sum()
    )

    choque_faltante = (
        choque.isna()
        | choque.str.upper()
        .isin(["N/A", "NA", "NO", "0", "CHOQUE CON"])
        .fillna(False)
    )
    es_choque = df["clase_accidente"].eq("Choque")

    df["choque_con"] = (
        choque.mask(choque_faltante & es_choque, "Sin registro")
        .mask(choque_faltante & ~es_choque, "No aplica")
    )
    return df


def _tratar_objeto_fijo(df: pd.DataFrame, metricas: dict) -> pd.DataFrame:
    objeto = limpiar_texto(df["objeto_fijo"]).str.title()
    metricas["objeto_na"] = int(objeto.isna().sum())
    metricas["objeto_no"] = int(objeto.str.upper().eq("NO").fillna(False).sum())
    metricas["objeto_cero"] = int(objeto.eq("0").fillna(False).sum())

    objeto_faltante = (
        objeto.isna()
        | objeto.str.upper().isin(["0", "N/A", "NA", "NO"]).fillna(False)
    )
    es_objeto_fijo = df["choque_con"].eq("Objeto Fijo")

    df["objeto_fijo"] = (
        objeto.mask(objeto_faltante & es_objeto_fijo, "Sin registro")
        .mask(objeto_faltante & ~es_objeto_fijo, "No aplica")
    )
    # Unifica variantes de escritura de la misma categoría.
    df["objeto_fijo"] = df["objeto_fijo"].mask(
        df["objeto_fijo"].str.upper().str.startswith("VALLA SE", na=False),
        "Valla Señal",
    )
    return df


def _tratar_hipotesis(df: pd.DataFrame, metricas: dict) -> pd.DataFrame:
    metricas["hipotesis_1_cero"] = int(
        limpiar_texto(df["hipotesis_1"]).eq("0").sum()
    )
    metricas["hipotesis_2_cero"] = int(
        limpiar_texto(df["hipotesis_2"]).eq("0").sum()
    )
    df["hipotesis_1"] = limpiar_codigo(df["hipotesis_1"])
    df["hipotesis_2"] = limpiar_codigo(df["hipotesis_2"])
    return df


def _tratar_ubicacion(df: pd.DataFrame, metricas: dict) -> pd.DataFrame:
    barrio = limpiar_texto(df["BARRIO"]).str.upper()
    metricas["barrio_sin_registro"] = int(barrio.isna().sum())
    df["BARRIO"] = barrio.fillna("SIN REGISTRO")

    metricas["localidad_sin_registro"] = int(df["localidad"].isna().sum())
    df["localidad"] = (
        df["localidad"].astype("string").str.strip().fillna("SIN REGISTRO")
    )

    lugar = limpiar_texto(df["lugar"]).str.upper()
    metricas["lugar_cero"] = int(lugar.eq("0").fillna(False).sum())
    metricas["lugar_sin_registro"] = int((lugar.isna() | lugar.eq("0")).sum())
    df["lugar"] = lugar.mask(lugar.eq("0"), "SIN REGISTRO").fillna("SIN REGISTRO")

    metricas["via_cero"] = {}
    for columna in VIA_COLS:
        via = limpiar_texto(df[columna]).str.upper()
        metricas["via_cero"][columna] = int(via.eq("0").fillna(False).sum())
        df[columna] = via.mask(via.eq("0") | via.isna(), "NO APLICA")

    return df


def _tratar_coordenadas(df: pd.DataFrame, metricas: dict) -> pd.DataFrame:
    for columna in ["latitud", "longitud"]:
        df[columna] = pd.to_numeric(
            limpiar_texto(df[columna]).str.replace(",", ".", regex=False),
            errors="coerce",
        )

    lat_min, lat_max = RANGO_LATITUD
    lon_min, lon_max = RANGO_LONGITUD
    df.loc[~df["latitud"].between(lat_min, lat_max), "latitud"] = float("nan")
    df.loc[~df["longitud"].between(lon_min, lon_max), "longitud"] = float("nan")

    # Si una coordenada es inválida, se anula el par para no dejar
    # georreferencias parciales.
    coordenada_incompleta = df["latitud"].isna() | df["longitud"].isna()
    df.loc[coordenada_incompleta, ["latitud", "longitud"]] = float("nan")
    metricas["coords_nulas_registros"] = int(coordenada_incompleta.sum())
    return df


def tratar_nulos(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Aplica reglas de negocio a nulos y centinelas."""
    metricas: dict = {}
    df = _tratar_gravedad(df, metricas)
    df = _tratar_clase_accidente(df, metricas)
    df = _tratar_choque_con(df, metricas)
    df = _tratar_objeto_fijo(df, metricas)
    df = _tratar_hipotesis(df, metricas)
    df = _tratar_ubicacion(df, metricas)
    df = _tratar_coordenadas(df, metricas)
    return df, metricas


# ---------------------------------------------------------------------------
# Estandarización, exportación y resumen
# ---------------------------------------------------------------------------
def estandarizar(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Convierte fechas/horas y agrega componentes de fecha."""
    metricas = {}

    df["fecha"] = pd.to_datetime(
        df["fecha"], format=FORMATO_FECHA, errors="coerce"
    )
    metricas["fechas_invalidas"] = int(df["fecha"].isna().sum())

    df["anio"] = df["fecha"].dt.year.astype("Int64")
    df["mes"] = df["fecha"].dt.month.astype("Int64")
    df["dia_semana"] = (
        df["fecha"]
        .dt.strftime("%w")
        .astype("Int64")
        .map(DIAS_SEMANA)
        .astype("string")
        .fillna("Sin registro")
    )

    hora = pd.to_datetime(df["hora"], format=FORMATO_HORA, errors="coerce")
    metricas["horas_invalidas"] = int(hora.isna().sum())
    df["hora"] = (
        hora.dt.strftime("%H:%M").astype("string").fillna("Sin registro")
    )

    orden_columnas = ["fecha", "anio", "mes", "dia_semana", "hora"] + [
        columna
        for columna in df.columns
        if columna not in {"fecha", "anio", "mes", "dia_semana", "hora"}
    ]
    return df[orden_columnas], metricas


def exportar(df: pd.DataFrame, ruta: Path) -> None:
    """Guarda la base depurada en UTF-8 con BOM para compatibilidad con Excel."""
    ruta.parent.mkdir(parents=True, exist_ok=True)
    df.reset_index(drop=True).to_csv(
        ruta,
        index=False,
        encoding="utf-8-sig",
        date_format="%Y-%m-%d",
    )


def total_celdas_tratadas(metricas: dict) -> int:
    """Suma las celdas nulas o con centinelas que recibieron un valor."""
    return (
        metricas["choque_na"]
        + metricas["choque_no"]
        + metricas["choque_malformado"]
        + metricas["objeto_cero"]
        + metricas["objeto_no"]
        + metricas["objeto_na"]
        + metricas["hipotesis_1_cero"]
        + metricas["hipotesis_2_cero"]
        + metricas["lugar_cero"]
        + sum(metricas["via_cero"].values())
        + metricas["barrio_sin_registro"]
        + metricas["localidad_sin_registro"]
        + metricas["clase_sin_registro"]
        + metricas["horas_invalidas"]
    )
