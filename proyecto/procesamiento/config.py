"""Parametros del pipeline. Todo lo que se pueda querer cambiar vive aca, no en las funciones."""
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

# --- filtrado.py ------------------------------------------------------------
CANALES_DESCARTADOS = ["TP9", "TP10", "HEOR"]        # los descartan los autores (mastoides y ojo horizontal)
CANAL_EOG = "VEO"                                    # ojo vertical
PASA_ALTOS_HZ = 1.0                                  # como los autores (code/preprocessing_95.m)
PASA_BAJOS_HZ = 100.0
NOTCH_HZ = 50.0                                      # red electrica de 50 Hz
NOTCH_ANCHO_HZ = 4.0                                 # rechaza 48-52 Hz
PASA_BAJOS_LEP_HZ = 30.0                             # solo para mirar el LEP en las figuras de control

# --- epocas.py-----------------------------------------------------------
# autores del paper: un trigger S 32..S 35 seguido inmediatamente por S 100 es un pulso laser.
CODIGOS_LASER = {32: 1, 33: 2, 34: 3, 35: 4}         # codigo del trigger -> nivel E1..E4
CODIGO_CONFIRMACION = 100
NIVELES = {"E1": 1, "E2": 2, "E3": 3, "E4": 4}
EPOCAS_ESPERADAS = 80                                # 20 pulsos por nivel
TMIN, TMAX = -1.0, 2.0                               # s: 1000 ms antes y 2000 ms despues del pulso (paper)
BASELINE = (-1.0, 0.0)                               # linea de base: todo el pre-estimulo (paper)
SFREQ_GUARDADO = 250                                 # Hz; se remuestrea antes de guardar (None = no remuestrear)
CANAL_VERTEX = "Cz"                                  # donde se mira el LEP en la figura de control

COLORES = {"E1": "#9ecae1", "E2": "#4292c6", "E3": "#ef6548", "E4": "#b30000"}
