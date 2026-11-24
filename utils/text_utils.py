# -*- coding: utf-8 -*-
"""Utilidades de limpieza de texto reutilizables."""

import pandas as pd

from config.config import REEMPLAZOS_CP1252


def reparar_codificacion(valor):
    """Repara textos con doble codificación UTF-8/cp1252 (mojibake)."""
    if pd.isna(valor):
        return valor

    texto = str(valor)
    if "Ã" not in texto and "Â" not in texto:
        return texto

    # Recupera caracteres tipográficos de cp1252 antes de re-codificar.
    for tipografico, byte in REEMPLAZOS_CP1252.items():
        texto = texto.replace(tipografico, byte)

    try:
        return texto.encode("latin-1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return str(valor)


def limpiar_texto(serie: pd.Series) -> pd.Series:
    """Repara codificación, quita espacios repetidos y convierte vacíos en nulos."""
    return (
        serie.astype("string")
        .str.replace("\u00a0", " ", regex=False)  # espacio no separable
        .map(reparar_codificacion)
        .astype("string")
        # Correcciones puntuales observadas después de reparar la codificación.
        .str.replace("VIA\x8dA", "VÍA", regex=False)
        .str.replace("VÍAL", "VIAL", regex=False)
        .str.replace("MÁNDELA", "MANDELA", regex=False)
        .str.replace("TÊNEL", "TÚNEL", regex=False)
        .str.replace(r"[\x00-\x1f\x7f-\x9f]", "", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
        .mask(lambda s: s.eq(""), pd.NA)
    )


def limpiar_codigo(serie: pd.Series) -> pd.Series:
    """Normaliza un código de hipótesis a texto sin comas ni decimales."""
    codigo = (
        limpiar_texto(serie)
        .str.replace(",", "", regex=False)      # "1,369" -> "1369"
        .str.replace(r"\.0+$", "", regex=True)  # "112.0" -> "112"
    )
    # En este dataset el código 0 significa "sin registro".
    codigo = codigo.mask(codigo.isin(["0", "N/A", "NA"]), pd.NA)
    return codigo.fillna("Sin registro")
