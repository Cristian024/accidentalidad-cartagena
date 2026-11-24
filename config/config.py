# -*- coding: utf-8 -*-
"""Constantes y configuración compartida del proyecto."""

from pathlib import Path


# config/config.py vive dentro de config/, por lo que la raíz del proyecto
# es la carpeta padre.
BASE_DIR = Path(__file__).resolve().parent.parent
DIRECTORIO_DATOS = BASE_DIR / "data"

ARCHIVO_ENTRADA = DIRECTORIO_DATOS / (
    "Accidentes_de_Tránsito_Ocurridos_en_la_Cuidad_de_Cartagena_de_Indias._20261122.csv"
)
ARCHIVO_SALIDA = DIRECTORIO_DATOS / "accidentes_cartagena_limpio.csv"

# Variables conservadas para el análisis descriptivo y predictivo.
# Se excluyen:
#   - ipat: identificador del incidente.
#   - hipotesis_3 y hipotesis_4: fuera del alcance y muy dispersas.
#   - muertos y heridos: posibles fugas de información respecto a gravedad.
#   - coordenadas: redundante con latitud y longitud.
COLUMNAS_ANALISIS = [
    "fecha",
    "hora",
    "gravedad",
    "clase_accidente",
    "choque_con",
    "objeto_fijo",
    "hipotesis_1",
    "hipotesis_2",
    "BARRIO",
    "lugar",
    "via1",
    "via2",
    "via3",
    "via4",
    "via5",
    "via6",
    "localidad",
    "latitud",
    "longitud",
]

VIA_COLS = ["via1", "via2", "via3", "via4", "via5", "via6"]

# Claves de %w: 0 = domingo, 1 = lunes, ..., 6 = sábado.
DIAS_SEMANA = {
    0: "Domingo",
    1: "Lunes",
    2: "Martes",
    3: "Miércoles",
    4: "Jueves",
    5: "Viernes",
    6: "Sábado",
}

# Formato original de fecha/hora (12 horas, mes en inglés).
FORMATO_FECHA = "%Y %b %d %I:%M:%S %p"
FORMATO_HORA = "%Y %b %d %I:%M:%S %p"

# Rango geográfico aproximado de Cartagena de Indias.
RANGO_LATITUD = (8, 12)
RANGO_LONGITUD = (-77, -74)

# Traducciones para reparar textos con doble codificación (mojibake).
REEMPLAZOS_CP1252 = {
    "\u2018": "\x91",
    "\u2019": "\x92",
    "\u201c": "\x93",
    "\u201d": "\x94",
    "\u2013": "\x96",
    "\u2014": "\x97",
    "\u2026": "\x85",
}

# Patrón para detectar columnas con posibles datos personales.
PATRON_DATOS_PERSONALES = (
    r"nombre|apellido|cedula|cédula|documento|identificacion|identificación|"
    r"placa|propietario|conductor|victima|víctima|telefono|teléfono|"
    r"celular|correo|email|contacto"
)
