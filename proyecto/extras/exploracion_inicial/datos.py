
from pathlib import Path
from collections import Counter

import mne
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from pathlib import Path

BASE = Path(__file__).resolve().parent

# Buscar el archivo EEG tanto en la carpeta actual como en la interna.
RUTA_RELATIVA = Path(
    "data/ds005293/sub-001/ses-1/eeg/"
    "sub-001_ses-1_task-95ByBP_eeg.vhdr"
)

CANDIDATAS = [
    BASE / RUTA_RELATIVA,
    BASE / "PSIB_Grupo15_EEG" / RUTA_RELATIVA,
]

ARCHIVO_RAW = next(
    (ruta for ruta in CANDIDATAS if ruta.exists()),
    None
)

if ARCHIVO_RAW is None:
    raise FileNotFoundError(
        "No encuentro el archivo EEG. Busqué en estas ubicaciones:\n"
        + "\n".join(str(ruta) for ruta in CANDIDATAS)
    )

CARPETA_PROYECTO = ARCHIVO_RAW.parents[5]
CARPETA_EEG = ARCHIVO_RAW.parent
CARPETA_SALIDA = CARPETA_PROYECTO / "derivatives" / "epochs_raw"
CARPETA_SALIDA.mkdir(parents=True, exist_ok=True)

print("Proyecto:", CARPETA_PROYECTO)
print("Archivo EEG:", ARCHIVO_RAW)
print("Resultados:", CARPETA_SALIDA)

# ============================================================
# PARÁMETROS DE SEGMENTACIÓN
# ============================================================

TMIN = -1.0
TMAX = 2.0

# None significa incluir todos los marcadores de tipo Stimulus.
# Todavía no asumimos que todos sean estímulos láser.
CODIGOS_SELECCIONADOS = None

CANALES_GRAFICAR = ["Fz", "Cz", "Pz"]


# ============================================================
# 1. CARGA DEL EEG CRUDO
# ============================================================

def cargar_raw():

    if not ARCHIVO_RAW.exists():
        raise FileNotFoundError(
            f"No se encontró el archivo:\n{ARCHIVO_RAW}"
        )

    print("\n1. Cargando EEG crudo...")

    raw = mne.io.read_raw_brainvision(
        str(ARCHIVO_RAW),
        preload=True,
        verbose="ERROR"
    )

    print("EEG cargado correctamente.")
    print(f"Canales: {len(raw.ch_names)}")
    print(f"Frecuencia: {raw.info['sfreq']:.0f} Hz")
    print(f"Duración: {raw.times[-1]:.2f} s")

    return raw


# ============================================================
# 2. IDENTIFICACIÓN Y RESUMEN DE MARCADORES
# ============================================================

def obtener_marcadores(raw):

    print("\n2. Identificando marcadores...")

    # Convertimos las anotaciones del EEG en eventos de MNE.
    eventos, event_id = mne.events_from_annotations(
        raw,
        event_id="auto",
        verbose="ERROR"
    )

    # Conservamos solo anotaciones correspondientes a Stimulus.
    codigos_stimulus = {
        descripcion: codigo
        for descripcion, codigo in event_id.items()
        if "Stimulus/" in descripcion
    }

    if CODIGOS_SELECCIONADOS is not None:
        codigos_stimulus = {
            descripcion: codigo
            for descripcion, codigo in codigos_stimulus.items()
            if any(
                descripcion.endswith(codigo_texto)
                for codigo_texto in CODIGOS_SELECCIONADOS
            )
        }

    eventos_stimulus = eventos[
        np.isin(eventos[:, 2], list(codigos_stimulus.values()))
    ]

    if len(eventos_stimulus) == 0:
        raise ValueError(
            "No se encontraron marcadores Stimulus. "
            "Revisá las anotaciones del archivo."
        )

    descripcion_por_codigo = {
        codigo: descripcion
        for descripcion, codigo in codigos_stimulus.items()
    }

    tabla = pd.DataFrame({
        "muestra": eventos_stimulus[:, 0],
        "tiempo_s": eventos_stimulus[:, 0] / raw.info["sfreq"],
        "codigo_mne": eventos_stimulus[:, 2],
    })

    tabla["marcador"] = tabla["codigo_mne"].map(
        descripcion_por_codigo
    )

    print(f"Total de marcadores Stimulus: {len(tabla)}")

    print("\nCantidad de eventos por marcador:")
    print(
        tabla["marcador"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print("\nPrimeros 20 marcadores encontrados:")
    print(tabla.head(20).to_string(index=False))

    # Guardar tabla para revisar todos los eventos.
    archivo_csv = CARPETA_SALIDA / "marcadores_raw.csv"
    tabla.to_csv(archivo_csv, index=False)

    print(f"\nTabla de marcadores guardada en:\n{archivo_csv}")

    # Gráfico de cantidad de eventos por tipo.
    conteos = tabla["marcador"].value_counts().sort_index()

    plt.figure(figsize=(11, 5))
    conteos.plot(kind="bar")
    plt.title("Cantidad de eventos por marcador")
    plt.xlabel("Marcador")
    plt.ylabel("Cantidad")
    plt.xticks(rotation=45, ha="right")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()

    return eventos_stimulus, codigos_stimulus, tabla


# ============================================================
# 3. SEGMENTACIÓN
# ============================================================

def segmentar(raw, eventos, codigos_stimulus):

    print("\n3. Segmentando el EEG...")

    epochs = mne.Epochs(
        raw,
        events=eventos,
        event_id=codigos_stimulus,
        tmin=TMIN,
        tmax=TMAX,
        baseline=None,
        preload=True,
        reject_by_annotation=True,
        event_repeated="drop",
        verbose="ERROR"
    )

    print("\n========== RESULTADO DE SEGMENTACIÓN ==========")
    print(f"Épocas conservadas: {len(epochs)}")
    print(f"Canales: {len(epochs.ch_names)}")
    print(f"Frecuencia: {epochs.info['sfreq']:.0f} Hz")
    print(f"Muestras por época: {len(epochs.times)}")
    print(f"Ventana: {epochs.times[0]:.3f} a {epochs.times[-1]:.3f} s")
    print("Filtrado: no")
    print("Corrección de línea de base: no")

    archivo_epochs = (
        CARPETA_SALIDA
        / "sub-001_ses-1_desc-raw_all-stimulus_epo.fif"
    )

    epochs.save(archivo_epochs, overwrite=True)
    print(f"\nÉpocas guardadas en:\n{archivo_epochs}")

    return epochs


# ============================================================
# 4. VISUALIZACIÓN DE ÉPOCAS
# ============================================================

def graficar_epocas(epochs, cantidad=6):

    if len(epochs) == 0:
        print("No hay épocas para graficar.")
        return

    canales = [
        canal for canal in CANALES_GRAFICAR
        if canal in epochs.ch_names
    ]

    if not canales:
        canales = [epochs.ch_names[0]]

    cantidad = min(cantidad, len(epochs))
    datos = epochs.get_data(
        picks=canales,
        units="uV"
    )

    fig, axes = plt.subplots(
        len(canales),
        1,
        figsize=(12, 3 * len(canales)),
        sharex=True,
        squeeze=False
    )

    for i, canal in enumerate(canales):
        ax = axes[i, 0]

        for j in range(cantidad):
            ax.plot(
                epochs.times,
                datos[j, i],
                linewidth=0.8,
                label=f"Época {j + 1}"
            )

        ax.axvline(
            0,
            color="red",
            linestyle="--",
            label="Marcador"
        )
        ax.set_ylabel(f"{canal} (µV)")
        ax.grid(True, alpha=0.3)

        if i == 0:
            ax.legend(ncol=4)

    axes[-1, 0].set_xlabel("Tiempo (s)")
    fig.suptitle("Primeras épocas extraídas del EEG crudo")
    plt.tight_layout()
    plt.show()


# ============================================================
# 5. EJECUCIÓN COMPLETA
# ============================================================

if __name__ == "__main__":

    raw = cargar_raw()

    eventos, codigos, tabla = obtener_marcadores(raw)

    epochs = segmentar(raw, eventos, codigos)

    graficar_epocas(epochs, cantidad=6)

    print("\nProceso terminado.")
    print(
        "\nNota: se segmentaron todos los marcadores Stimulus "
        "detectados. Todavía hay que identificar cuáles "
        "corresponden específicamente a los estímulos láser."
    )
