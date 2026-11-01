"""Constantes del proyecto. Todo lo que se pueda querer cambiar vive aca, no en las funciones."""
import os
from pathlib import Path

def _buscar_datos():
    """Carpeta del dataset. Orden: variable PSIB_DATOS, luego data/ds005293 en la carpeta
    actual o en cualquier carpeta por encima del paquete; si no hay, data/ds005293 de la actual."""
    env = os.environ.get("PSIB_DATOS")
    if env:
        return Path(env)
    for base in [Path.cwd(), *Path(__file__).resolve().parents]:
        candidata = base / "data" / "ds005293"
        if candidata.is_dir():
            return candidata
    return Path("data") / "ds005293"


RAIZ_DATOS = _buscar_datos()                        # donde descargar_sujeto deja el dataset
CARPETA_SALIDA = RAIZ_DATOS.parent / "derived_mine"  # resultados intermedios (epocas, figuras)

TAREA = "95ByBP"                    # nombre de la tarea en los archivos BIDS

# --- Canales -----------------------------------------------------------------
# Canales que los autores descartan (TP9/TP10: mastoides; HEOR: ojo horizontal)
CANALES_DESCARTADOS = ["TP9", "TP10", "HEOR"]
CANAL_EOG = "VEO"                   # ojo vertical: referencia para detectar parpadeos

# --- Eventos -----------------------------------------------------------------
# Regla de los autores (code/preprocessing_95.m): un trigger S 32..S 35 seguido
# inmediatamente por S100 es un pulso laser. Los otros codigos son otras modalidades.
CODIGOS_LASER = {32: 1, 33: 2, 34: 3, 35: 4}   # codigo del trigger -> nivel E1..E4
CODIGO_CONFIRMACION = 100
NIVELES = {"E1": 1, "E2": 2, "E3": 3, "E4": 4}

# Esperado en el dataset: 80 pulsos por sujeto, 20 por nivel
PULSOS_POR_NIVEL = 20

# Frecuencia de muestreo de los datos guardados
SFREQ_GUARDADO = 250                # Hz; se remuestrea antes de guardar las epocas (None = no remuestrear)

# --- Filtros (filtros.py) ----------------------------------------------------
PASA_ALTOS_HZ = 1.0
PASA_BAJOS_HZ = 100.0
NOTCH_HZ = 50.0                     # red electrica argentina/china: 50 Hz
NOTCH_ANCHO_HZ = 4.0                # rechaza 48-52 Hz (como filtrado.py)
ORDEN_IIR = 4                       # solo para la comparacion FIR vs IIR
PASA_BAJOS_LEP_HZ = 30.0            # para mirar y medir LEPs

# --- Segmentacion ------------------------------------------------------------
TMIN, TMAX = -1.0, 2.0              # ventana de la epoca respecto del pulso (s)
BASELINE = (-1.0, 0.0)              # linea de base: todo el pre-estimulo

# --- Control de calidad ------------------------------------------------------
CANAL_QC = "Cz"                     # vertex: donde N2 y P2 son maximos
VENTANA_N2 = (0.12, 0.30)           # s; busqueda del minimo (N2 en mano ~200-250 ms)
VENTANA_P2 = (0.25, 0.50)           # s; busqueda del maximo (P2 ~300-400 ms)
UMBRALES_PTP_UV = (100, 150)        # lineas de referencia en la figura de pico a pico

# --- Limpieza comun (canales_malos.py, preprocesado.py) ----------------------
# Sin usar el nivel (E1..E4): se aplica antes del localizador.
CANALES_SIN_CRITERIO_EEG = ["Fp1", "Fpz", "Fp2", "AF7", "AF3", "AF4", "AF8"]  # ojos -> los decide VEO
Z_CANAL_MALO = 5.0                  # z robusto del ruido pre-estimulo para marcar un canal (canales_malos.py)
FRAC_EPOCAS_CANAL_MALO = 0.5        # fraccion de epocas atipicas para considerarlo malo (y no epocas malas)
FACTOR_CANAL_PLANO = 0.1            # pico a pico tipico menor a esto x la mediana -> canal plano
UMBRAL_GROSERO_UV = 300             # pico a pico maximo (sin frontopolares, 30 Hz) -> epoca grosera (preprocesado.py)
VENTANA_GROSERO = (-1.0, 1.0)       # s; tramo donde se busca el artefacto grosero (el que se analiza)
REF_SIN_FRONTALES = True            # referencia promedio sin CANALES_SIN_CRITERIO_EEG: el ojo no se reparte a Cz
REGRESION_VEO = False               # descartada: el VEO explica poco de los frontales (R2 6-34%, pocos parpadeos espontaneos)
VENTANAS_BETA_VEO = [(-1.0, 0.0), (1.0, 2.0)]   # s; tramos sin LEP donde se estima el beta del VEO

# --- Rechazo (artefactos.py, a revisar) --------------------------------------
# Pico a pico medido sobre una copia con pasa-bajos de 30 Hz (banda de los LEPs).
VENTANA_VEO = (-0.2, 0.8)           # s; parpadeo en esta ventana (N2/P2) -> la epoca se descarta
UMBRAL_VEO_UV = 100                 # pico a pico maximo de VEO en VENTANA_VEO
UMBRAL_RECHAZO_UV = 100             # pico a pico maximo en cualquier canal EEG (epoca completa)

# --- Inclusion de sujetos (inclusion.py) ---
MIN_EPOCAS_LIMPIAS = 10             # epocas limpias minimas por nivel de NIVELES_MINIMOS
NIVELES_MINIMOS = ("E1", "E4")      # niveles que se comparan en la clasificacion
EPOCAS_ESPERADAS = 80               # base para el % de perdidas: las faltantes cuentan como perdidas
MAX_PCT_EPOCAS_PERDIDAS = 25        # > 25% perdidas (autores + groseras + rechazo) -> se excluye 
MAX_PCT_CANALES_INTERPOLADOS = 25   # > 25% de canales EEG interpolados -> se excluye
CANALES_ROI = ["Cz", "FC1", "FC2", "C1", "C2", "CP1", "CPz", "CP2"]   # ROI a priori: Cz y sus vecinos (no hay FCz)
MAX_ROI_INTERPOLADOS = 2            # mas de 2 canales de la ROI interpolados -> se excluye
# Cz interpolado -> se excluye siempre (criterio 4); sin canal VEO -> se excluye