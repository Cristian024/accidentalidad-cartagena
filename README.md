# Análisis de Siniestralidad Vial en Cartagena de Indias

Aplicación web interactiva en **Streamlit** con un **tablero descriptivo** y un
**componente predictivo** construidos a partir del registro oficial de
accidentes de tránsito de Cartagena de Indias (2020-2025).

- Dashboard: KPIs, gráficos por gravedad, temporalidad, territorio y un mapa.
- Predictivo: `RandomForestClassifier`, matriz de confusión, importancia de
  variables y un simulador de escenarios.
- Fuente: Alcaldía Mayor de Cartagena - Datos Abiertos Colombia
  (código `5z62-bn99`).

## Requisitos

- **Python 3.10 o superior** (el proyecto se probó con Python 3.14).
- **pip** actualizado.
- Opcional pero recomendado: entorno virtual (`.venv`).

## Estructura del proyecto

```text
proyecto/
├── main.py                      # Punto de entrada único (limpiar / dashboard / todo)
├── requirements.txt             # Dependencias
├── README.md                    # Esta guía
├── config/
│   ├── __init__.py
│   └── config.py                # Rutas y constantes
├── utils/
│   ├── __init__.py
│   ├── data_cleaning.py         # Carga, auditoría, nulos y exportación
│   └── text_utils.py            # Limpieza de texto y códigos
├── ui/
│   └── app.py                   # Aplicación Streamlit
├── data/
│   ├── Accidentes_de_Tránsito_Ocurridos_en_la_Cuidad_de_Cartagena_de_Indias._20261122.csv
│   └── accidentes_cartagena_limpio.csv
└── docs/
    └── Informe_Accidentes_Cartagena.pdf
```

## 1. Crear y activar el entorno

### Windows (PowerShell)

```powershell
cd "D:\Unimayor\Semestre 9\ElectivaIII\proyecto"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Si PowerShell bloquea la activación, ejecuta una vez:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## 2. Generar la base depurada

```powershell
python main.py limpiar
```

Esto lee `data/Accidentes_de_Tránsito_..._20261122.csv`, limpia los datos y
crea `data/accidentes_cartagena_limpio.csv`.

Debe terminar con:

- `Proceso terminado correctamente.`
- Dimensiones finales: **7,923 filas x 22 columnas**.

> `python main.py` sin argumentos ejecuta esta misma acción (la limpieza es la
> acción por defecto).

## 3. Abrir el dashboard en local

```powershell
python main.py dashboard
```

- Abre **http://localhost:8501** en el navegador.
- Si `data/accidentes_cartagena_limpio.csv` ya existe, abre directamente.
- Si no existe, genera la base limpia antes de abrir.

Alternativa directa:

```powershell
streamlit run ui/app.py
```

Para limpiar y abrir en un solo paso:

```powershell
python main.py todo
```

## 4. Qué verificar en el dashboard

En la pestaña **Dashboard descriptivo**:

- Título: **Análisis de Siniestralidad Vial en Cartagena de Indias**.
- Barra lateral con filtros de **Gravedad**, **Clase de accidente** y
  **Localidad**.
- KPIs: **Total 7,923**, **Siniestros con muertos 7.70%**, clase más frecuente
  **Choque**, `hipotesis_1` predominante **139**.
- Indicadores complementarios: hora pico **07:00**, día **Domingo**, localidad
  **Localidad 1**, barrio **MAMONAL**.
- Secciones con gráficos:
  - **Frecuencia y gravedad**: distribución y letalidad por clase.
  - **Temporalidad**: evolución anual, día de la semana, hora y mapa de calor.
  - **Territorio**: top de barrios, localidades y un **mapa** con
    7,916 siniestros georreferenciados.
  - **Hipótesis iniciales**: top de códigos y letalidad por código.

En la pestaña **Componente predictivo**:

- **Exactitud (Accuracy):** ~92.30%.
- **F1-Score ponderado:** ~0.886.
- Matriz de confusión explicada.
- Gráfico de importancia relativa de variables.
- **Simulador** para estimar la probabilidad de gravedad de un escenario.

> El mapa usa teselas de **OpenStreetMap**, por lo que necesita conexión a
> internet para mostrar el fondo del mapa.

## 5. Detener la aplicación

En la terminal donde corre Streamlit, presiona **Ctrl+C**.

## Problemas comunes

- **"No se encontró data/accidentes_cartagena_limpio.csv"**: ejecuta
  `python main.py limpiar`.
- **La activación del entorno falla**: usa
  `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` o ejecuta
  `.\.venv\Scripts\python.exe main.py dashboard` sin activar.
- **El primer arranque tarda**: Streamlit entrena el modelo una sola vez y lo
  cachea.
- **Puerto ocupado**: si 8501 está en uso, Streamlit abrirá otro puerto y
  mostrará la URL en consola.

## Despliegue (opcional)

En **Streamlit Community Cloud** (https://share.streamlit.io):

1. Sube `ui/app.py`, `requirements.txt` y
   `data/accidentes_cartagena_limpio.csv` a un repositorio de GitHub.
2. Crea la app y en **Main file path** escribe `ui/app.py`.
3. Copia el enlace público generado.

> `main.py` es solo el punto de entrada local; en la nube el archivo principal
> es `ui/app.py`.
