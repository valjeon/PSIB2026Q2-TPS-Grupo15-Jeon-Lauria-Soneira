"""Parametros del proyecto. Lo que usan varios modulos y las decisiones de metodo viven aca, no en las funciones.

Los criterios de limpieza (canales, umbrales) se fijaron antes de mirar la diferencia entre niveles.
"""
import os
from pathlib import Path


def _buscar_datos():
    """Carpeta del dataset: variable PSIB_DATOS, o data/ds005293 en la carpeta actual o por encima del paquete."""
    env = os.environ.get("PSIB_DATOS")
    if env:
        return Path(env)
    for base in [Path.cwd(), *Path(__file__).resolve().parents]:
        candidata = base / "data" / "ds005293"
        if candidata.is_dir():
            return candidata
    return Path("data") / "ds005293"


RAIZ_DATOS = _buscar_datos()                         # donde descargar_sujeto.sh deja el dataset
CARPETA_SALIDA = RAIZ_DATOS.parent / "derived_mine"  # todo lo que genera el pipeline

# --- Lectura (analisis.py) ------------------------------------------------------------------------------
CANALES_DESCARTADOS = ["TP9", "TP10", "HEOR"]        # los descartan los autores (mastoides y ojo horizontal)
CANAL_EOG = "VEO"                                    # ojo vertical: se usa para el rechazo por parpadeo
# Regla de los autores: un trigger S 32..S 35 seguido inmediatamente por S 100 es un pulso laser.
CODIGOS_LASER = {32: 1, 33: 2, 34: 3, 35: 4}         # codigo del trigger -> nivel E1..E4
CODIGO_CONFIRMACION = 100
NIVELES = {"E1": 1, "E2": 2, "E3": 3, "E4": 4}
PULSOS_POR_NIVEL = 20
EPOCAS_ESPERADAS = 80                                # 20 pulsos por nivel

# --- Filtros (filtrado.py) ------------------------------------------------------------------------------
PASA_ALTOS_HZ = 1.0                                  # como los autores (code/preprocessing_95.m)
PASA_BAJOS_HZ = 100.0
NOTCH_HZ = 50.0                                      # red electrica
NOTCH_ANCHO_HZ = 4.0                                 # rechaza 48-52 Hz
PASA_BAJOS_LEP_HZ = 30.0                             # solo para medir el LEP (copias, no lo que se guarda)

# --- Epocas (epocas.py) ---------------------------------------------------------------------------------
TMIN, TMAX = -1.0, 2.0                               # s: 1000 ms antes y 2000 ms despues del pulso (paper)
BASELINE = (-1.0, 0.0)                               # linea de base: todo el pre-estimulo (paper)
SFREQ_GUARDADO = 250                                 # Hz; se remuestrea antes de guardar (None = no remuestrear)

# --- Canales ---------------------------------------------------------------------------------------------
CANALES_ROI = ["Cz", "FC1", "FC2", "C1", "C2", "CP1", "CPz", "CP2"]   # ROI (csoneira): Cz y vecinos (no hay FCz)
CANAL_QC = "Cz"                                      # vertex: donde se buscan las latencias de N2 y P2
CANALES_SIN_CRITERIO_EEG = ["Fp1", "Fpz", "Fp2", "AF7", "AF3", "AF4", "AF8"]   # frontopolares: amplitud ocular
REF_SIN_FRONTALES = True                             # referencia promedio sin los frontopolares (limpieza.py)
VENTANA_N2 = (0.12, 0.30)                            # s; donde se busca el minimo
VENTANA_P2 = (0.25, 0.50)                            # s; donde se busca el maximo
# Localizador (localizador.py)
FRACCION_VENTANA = 0.5                               # ventanas de N2/P2: ancho a media altura en el gran promedio
MEDIA_PICO_S = 0.02                                  # +-20 ms alrededor de cada pico, para la medida por canal
FRACCION_ROI_DATOS = 0.8                             # ROI que sale de los datos: canales con >= 80% del maximo

# --- Limpieza (limpieza.py) y rechazo final (medicion.py) -----------------------------------------------------------------------------
Z_CANAL_MALO = 5.0                                   # z robusto del pico a pico pre-estimulo para un canal ruidoso
FRAC_EPOCAS_CANAL_MALO = 0.5                         # ...y atipico en mas de esta fraccion de las epocas
FACTOR_CANAL_PLANO = 0.1                             # pico a pico tipico < 0.1 x la mediana -> canal plano
MAX_ROI_INTERPOLADOS = 2                             # Cz interpolado, o mas de 2 de la ROI -> ROI comprometida
UMBRAL_GROSERO_UV = 300                              # pico a pico en la cabeza (sin frontopolares) -> grosera
VENTANA_GROSERO = (-1.0, 1.0)                        # s; tramo de las groseras
# Rechazo final (medicion.py): de -1 s al fin de la ventana de P2 que sale del localizador
UMBRAL_RECHAZO_UV = 100                              # pico a pico maximo en los canales de la ROI
UMBRAL_VEO_UV = 150                                  # pico a pico maximo del VEO (variante principal)
UMBRALES_VEO_SENSIBILIDAD = [200, None]              # variantes de sensibilidad (None = sin rechazo por VEO)

# --- Inclusion de sujetos (inclusion.py) -------------------------------------------------------------------
MAX_PCT_EPOCAS_PERDIDAS = 25                         # > 25% de las 80 perdidas (todas las causas) -> se excluye
MAX_PCT_CANALES_INTERPOLADOS = 25                    # > 25% de los canales EEG malos (autores + propios) -> se excluye
MIN_EPOCAS_LIMPIAS = 10                              # minimo de epocas limpias en cada nivel de NIVELES_MINIMOS
NIVELES_MINIMOS = ("E1", "E4")                       # los niveles que se comparan
