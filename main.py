# -*- coding: utf-8 -*-
"""
Punto de entrada único del proyecto de siniestralidad vial.

Acciones disponibles:
    python main.py              # Ejecuta la limpieza (acción por defecto).
    python main.py limpiar      # Genera data/accidentes_cartagena_limpio.csv.
    python main.py dashboard    # Abre la aplicación Streamlit.
    python main.py todo         # Limpia y luego abre la aplicación Streamlit.

La lógica está separada en:
    - config/config.py: constantes y rutas.
    - utils/text_utils.py: limpieza de texto y códigos.
    - utils/data_cleaning.py: carga, auditoría, nulos y exportación.
    - ui/app.py: página de Streamlit que consume la base limpia.

Requisitos:
    pip install -r requirements.txt
"""

import argparse
import importlib.util
from pathlib import Path
import subprocess
import sys

import pandas as pd

from config.config import ARCHIVO_ENTRADA, ARCHIVO_SALIDA
from utils.data_cleaning import (
    auditar_datos,
    cargar_datos,
    eliminar_duplicados,
    estandarizar,
    exportar,
    seleccionar_columnas,
    total_celdas_tratadas,
    tratar_nulos,
    verificar_privacidad,
)


RUTA_APP = Path(__file__).resolve().parent / "ui/app.py"


def imprimir_titulo(texto: str) -> None:
    """Imprime un encabezado para separar las etapas del script."""
    print("\n" + "=" * 80)
    print(texto)
    print("=" * 80)


def ejecutar_limpieza() -> None:
    """Ejecuta la carga, auditoría, limpieza y exportación de la base."""
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 160)

    # =======================================================================
    # 1. CARGA Y AUDITORÍA INICIAL
    # =======================================================================
    imprimir_titulo("1. CARGA Y AUDITORÍA INICIAL")

    df = cargar_datos(ARCHIVO_ENTRADA)
    auditoria = auditar_datos(df)

    print(f"Versión de pandas: {pd.__version__}")
    print(
        f"Dimensiones iniciales: {auditoria['n_inicial']} filas x "
        f"{auditoria['n_columnas_inicial']} columnas"
    )
    print(f"Valores nulos iniciales: {auditoria['nulos_iniciales']}")
    print(
        "Filas duplicadas exactas (fila completa): "
        f"{auditoria['duplicados_fila_completa']}"
    )

    print("\nNombres de columnas:")
    print(auditoria["columnas"])

    print("\nTipos de datos y conteo de nulos por columna:")
    print(auditoria["tabla_auditoria"].to_string())

    print("\nValores más frecuentes en variables categóricas clave:")
    for columna, frecuencia in auditoria["frecuencias"].items():
        print(f"\n{columna}:")
        print(frecuencia.to_string())

    # =======================================================================
    # 2. PRIVACIDAD Y FILTRO DE COLUMNAS
    # =======================================================================
    imprimir_titulo("2. PRIVACIDAD Y FILTRO DE COLUMNAS")

    df, columnas_sensibles = verificar_privacidad(df)
    if columnas_sensibles:
        print(f"Columnas con posibles datos personales detectadas: {columnas_sensibles}")
        print("Estas columnas fueron eliminadas del conjunto de datos.")
    else:
        print("No se detectaron columnas con nombres, cédulas, placas o contactos.")
        print("El campo 'ipat' es un identificador de incidente, no un dato personal.")

    df, n_duplicados_fila_completa = eliminar_duplicados(df)
    print(
        "Duplicados exactos de fila completa eliminados: "
        f"{n_duplicados_fila_completa}"
    )

    df, columnas_eliminadas = seleccionar_columnas(df)

    print(f"\nColumnas conservadas ({len(df.columns)}):")
    print(list(df.columns))
    print(f"\nColumnas eliminadas ({len(columnas_eliminadas)}):")
    print(columnas_eliminadas)
    print(
        "\nCriterio: se conservan las variables de gravedad, clase de accidente, "
        "colisión, objeto fijo, hipótesis y ubicación/fecha solicitadas. "
        "Se excluyen identificadores y variables que pueden filtrar la gravedad "
        "(muertos y heridos)."
    )

    # =======================================================================
    # 3. TRATAMIENTO DE DUPLICADOS Y NULOS
    # =======================================================================
    imprimir_titulo("3. TRATAMIENTO DE DUPLICADOS Y NULOS")

    df, n_duplicados_analisis = eliminar_duplicados(df)
    print(
        "Duplicados exactos eliminados en las variables de análisis: "
        f"{n_duplicados_analisis}"
    )

    df, metricas = tratar_nulos(df)

    print(
        f"Filas sin 'gravedad' eliminadas: {metricas['sin_gravedad']}. "
        "Para el resto de variables no se eliminan filas."
    )
    print("\nTratamiento aplicado a nulos y centinelas:")
    print(
        f"  - choque_con: {metricas['choque_na']} 'N/A', "
        f"{metricas['choque_no']} 'No' y "
        f"{metricas['choque_malformado']} 'Choque con' imputados según la "
        "clase de accidente."
    )
    print(
        f"  - objeto_fijo: {metricas['objeto_cero']} '0', "
        f"{metricas['objeto_no']} 'No' y {metricas['objeto_na']} 'N/A' "
        "imputados según el tipo de choque."
    )
    print(
        f"  - hipotesis_1: {metricas['hipotesis_1_cero']} códigos '0' -> "
        "'Sin registro'."
    )
    print(
        f"  - hipotesis_2: {metricas['hipotesis_2_cero']} códigos '0' -> "
        "'Sin registro'."
    )
    print(
        f"  - BARRIO: {metricas['barrio_sin_registro']} nulos -> "
        "'SIN REGISTRO'."
    )
    print(
        f"  - localidad: {metricas['localidad_sin_registro']} nulos -> "
        "'SIN REGISTRO'."
    )
    print(
        f"  - lugar: {metricas['lugar_cero']} valores '0' -> 'SIN REGISTRO' "
        f"({metricas['lugar_sin_registro']} celdas tratadas)."
    )
    print(
        f"  - via1 a via6: {sum(metricas['via_cero'].values())} valores '0' "
        "-> 'NO APLICA'."
    )
    print(
        f"  - latitud/longitud: {metricas['coords_nulas_registros']} registros "
        "con al menos una coordenada inválida; se dejaron como nulos "
        "(no imputados)."
    )
    print(
        f"  - clase_accidente: {metricas['clase_sin_registro']} nulos -> "
        "'Sin registro'."
    )

    # =======================================================================
    # 4. ESTANDARIZACIÓN Y FORMATOS
    # =======================================================================
    imprimir_titulo("4. ESTANDARIZACIÓN Y FORMATOS")

    df, metricas_estandarizacion = estandarizar(df)
    metricas.update(metricas_estandarizacion)

    print(f"Fechas no convertidas: {metricas['fechas_invalidas']}")
    print(f"Horas no convertidas: {metricas['horas_invalidas']}")
    print(
        "Se extrajeron 'anio', 'mes' y 'dia_semana'. "
        "La hora quedó en formato HH:MM."
    )
    print(
        "Las hipótesis se dejaron como texto (código limpio, sin comas ni "
        "decimales) para que hipotesis_1 e hipotesis_2 tengan el mismo tipo."
    )
    print("\nRango de fechas:")
    print(f"  Desde: {df['fecha'].min()}")
    print(f"  Hasta: {df['fecha'].max()}")

    df, n_duplicados_finales = eliminar_duplicados(df)
    print(
        "Duplicados exactos eliminados tras la limpieza final: "
        f"{n_duplicados_finales}"
    )

    # =======================================================================
    # 5. EXPORTACIÓN Y RESUMEN DE AUDITORÍA
    # =======================================================================
    imprimir_titulo("5. EXPORTACIÓN Y RESUMEN DE AUDITORÍA")

    exportar(df, ARCHIVO_SALIDA)

    n_final = len(df)
    nulos_finales = int(df.isna().sum().sum())
    n_registros_con_nulos = int(df.isna().any(axis=1).sum())
    registros_eliminados = auditoria["n_inicial"] - n_final
    celdas_tratadas = total_celdas_tratadas(metricas)

    resumen = pd.DataFrame(
        {
            "Métrica": [
                "Registros iniciales",
                "Registros finales",
                "Registros eliminados",
                "Columnas iniciales",
                "Columnas finales",
                "Nulos iniciales (total)",
                "Nulos finales (total)",
                "Registros con nulos finales",
                "Celdas nulas o centinelas tratados",
            ],
            "Valor": [
                auditoria["n_inicial"],
                n_final,
                registros_eliminados,
                auditoria["n_columnas_inicial"],
                len(df.columns),
                auditoria["nulos_iniciales"],
                nulos_finales,
                n_registros_con_nulos,
                celdas_tratadas,
            ],
        }
    )
    print(resumen.to_string(index=False))

    print("\nNulos finales por columna (solo columnas con nulos):")
    nulos_por_columna = df.isna().sum()
    nulos_por_columna = nulos_por_columna[nulos_por_columna > 0]
    if nulos_por_columna.empty:
        print("  No quedan valores nulos.")
    else:
        print(nulos_por_columna.to_string())

    print("\nDecisiones aplicadas:")
    print(
        f"  1. Se eliminaron {n_duplicados_fila_completa} duplicados exactos de "
        f"fila completa y {n_duplicados_analisis} duplicados exactos en las "
        "variables de análisis."
    )
    print(
        f"  2. Se eliminaron {metricas['sin_gravedad']} filas sin la variable "
        "clave 'gravedad' (no había otras filas que eliminar por nulos)."
    )
    print(
        "  3. En 'choque_con' y 'objeto_fijo' se imputó 'No aplica' cuando la "
        "categoría no correspondía y 'Sin registro' cuando faltaba el dato "
        "dentro de una categoría que sí lo requería."
    )
    print(
        "  4. Los códigos '0' de 'hipotesis_1' y 'hipotesis_2' se trataron como "
        "'Sin registro'; además se quitaron comas de miles y decimales '.0'."
    )
    print(
        "  5. En 'lugar' y en 'via1' a 'via6' los ceros estructurales se "
        "reemplazaron por 'SIN REGISTRO' y 'NO APLICA', respectivamente. La "
        "mayor parte de las celdas tratadas corresponde a estos ceros."
    )
    print(
        "  6. 'gravedad' y 'clase_accidente' se estandarizaron con espacios "
        "limpios y formato título; las variables de ubicación se pasaron a "
        "mayúsculas. También se repararon textos con doble codificación "
        "(mojibake) en las variables de texto."
    )
    print(
        "  7. 'fecha' se convirtió a datetime y se extrajeron 'anio', 'mes' y "
        "'dia_semana'; 'hora' se normalizó a HH:MM."
    )
    print(
        f"  8. Se detectaron {metricas['coords_nulas_registros']} registros con "
        "coordenadas inválidas; se dejaron como nulos porque no existe una "
        "regla de imputación confiable."
    )
    print(
        "  9. Se conservaron únicamente las variables solicitadas. Se "
        "excluyeron 'muertos' y 'heridos' para evitar fuga de información "
        "sobre 'gravedad'."
    )

    print(f"\nArchivo exportado: {ARCHIVO_SALIDA}")
    print(f"Dimensiones finales: {df.shape[0]} filas x {df.shape[1]} columnas")
    print("\nProceso terminado correctamente.")


def ejecutar_dashboard() -> None:
    """Lanza la aplicación Streamlit usando el mismo intérprete de Python."""
    if not RUTA_APP.exists():
        print(f"No se encontró la aplicación Streamlit: {RUTA_APP}")
        raise SystemExit(1)

    if importlib.util.find_spec("streamlit") is None:
        print(
            "Streamlit no está instalado. Ejecuta:\n"
            "    python -m pip install -r requirements.txt"
        )
        raise SystemExit(1)

    print(f"Iniciando dashboard Streamlit: {RUTA_APP}")
    print("Presiona Ctrl+C para detener la aplicación.")
    try:
        subprocess.run(
            [sys.executable, "-m", "streamlit", "run", str(RUTA_APP)],
            check=True,
        )
    except KeyboardInterrupt:
        print("\nDashboard detenido.")
    except subprocess.CalledProcessError as exc:
        print(
            "No se pudo iniciar Streamlit. Verifica las dependencias con:\n"
            "    python -m pip install -r requirements.txt"
        )
        raise SystemExit(exc.returncode)


def main() -> None:
    """Punto de entrada único: limpieza, dashboard o ambos."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(
        prog="main.py",
        description=(
            "Punto de entrada único del proyecto de siniestralidad vial. "
            "Permite limpiar los datos, abrir el dashboard o hacer ambas cosas."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Ejemplos:\n"
            "  python main.py\n"
            "  python main.py limpiar\n"
            "  python main.py dashboard\n"
            "  python main.py todo\n"
        ),
    )
    parser.add_argument(
        "accion",
        nargs="?",
        default="limpiar",
        choices=["limpiar", "dashboard", "todo"],
        help=(
            "limpiar: genera la base depurada (acción por defecto). "
            "dashboard: abre Streamlit. todo: limpia y luego abre Streamlit."
        ),
    )
    args = parser.parse_args()

    if args.accion == "limpiar":
        ejecutar_limpieza()
    elif args.accion == "dashboard":
        if not ARCHIVO_SALIDA.exists():
            print(
                "No se encontró la base limpia. Se ejecutará la limpieza "
                "antes de abrir el dashboard."
            )
            ejecutar_limpieza()
        ejecutar_dashboard()
    elif args.accion == "todo":
        ejecutar_limpieza()
        ejecutar_dashboard()


if __name__ == "__main__":
    main()
