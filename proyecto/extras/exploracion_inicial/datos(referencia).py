from pathlib import Path
import mne
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat

CARPETA_EEG = (
    Path(__file__).resolve().parent
    / "PSIB_Grupo15_EEG"
    / "data"
    / "ds005293"
    / "sub-001"
    / "ses-1"
    / "eeg"
)

ARCHIVO_SET = (
    CARPETA_EEG.parent.parent.parent
    / "derivatives"
    / "rerefer"
    / "sub-001_95ByBP.set"
)

# Canales que vamos a observar inicialmente
CANALES = ["Fz", "Cz", "Pz"]

# Cargamos las épocas y sus metadatos
def cargar_datos():

    if not ARCHIVO_SET.exists():
        raise FileNotFoundError(
            f"No se encontró el archivo:\n{ARCHIVO_SET}"
        )

    epochs = mne.io.read_epochs_eeglab(
        str(ARCHIVO_SET),
        verbose="ERROR"
    )

    # Recuperar intensidad del láser y rating de cada época
    datos_mat = loadmat(
        str(ARCHIVO_SET),
        struct_as_record=False,
        squeeze_me=True
    )

    eventos_set = np.atleast_1d(datos_mat["event"])
    filas = []

    for evento in eventos_set:
        filas.append({
            "tipo": str(evento.type),
            "laser_power": float(evento.laser_power),
            "rating": float(evento.rating)
        })

    metadata = pd.DataFrame(filas)

    # Verificar que cada época tenga sus metadatos
    if len(metadata) != len(epochs):
        raise ValueError(
            f"Hay {len(epochs)} épocas, pero "
            f"{len(metadata)} registros de eventos. "
            "No se pueden asociar automáticamente."
        )

    # Asociar cada intensidad y rating a su época
    # Así podemos identificar qué intensidad recibió el participante y qué puntuación de dolor asignó a cada estímulo
    epochs.metadata = metadata
    
    # Resumen de la carga
    senal = epochs.get_data()

    print("\n========== DATOS CARGADOS ==========")
    print(f"Épocas: {len(epochs)}")
    print(f"Canales: {len(epochs.ch_names)}")
    print(f"Frecuencia: {epochs.info['sfreq']:.0f} Hz")
    print(
        f"Intervalo: {epochs.times[0]:.3f} a "
        f"{epochs.times[-1]:.3f} s"
    )
    print(f"Forma de los datos: {senal.shape}")
    print(
        f"Rango global: {np.nanmin(senal) * 1e6:.2f} a "
        f"{np.nanmax(senal) * 1e6:.2f} µV"
    )

    print("\nCantidad de épocas por intensidad:")
    print(
        metadata.groupby("laser_power")
        .size()
        .rename("cantidad")
        .to_string()
    )

    print("\nResumen de dolor percibido:")
    print(
        metadata.groupby("laser_power")["rating"]
        .agg(["count", "mean", "std"])
        .round(2)
        .to_string()
    )

    return epochs

# Visualizamos una época específica para cada canal
# Seleccionamos el primer fragmento de los 80 disponibles 
# Muestra UNICAMENTE la primer época (índice 0) de los 80 disponibles (en este caso es de 3,5J)
def graficar_epoca(epochs, indice=0):

    if not 0 <= indice < len(epochs):
        raise IndexError("El índice de época no es válido.")

    # Corregir la línea de base usando el período previo
    # al estímulo, entre -1 y 0 segundos
    epocas = epochs.copy()
    epocas.apply_baseline(baseline=(-1, 0))

    metadata = epocas.metadata.iloc[indice]
    tiempo = epocas.times

    fig, axes = plt.subplots(
        len(CANALES), 1,
        figsize=(12, 8),
        sharex=True
    )

    for ax, canal in zip(axes, CANALES):

        senal_uv = epocas.get_data(
            picks=[canal] # primera época y canal específico
        )[indice, 0] * 1e6 # pasamos a microvoltios

        ax.plot(tiempo, senal_uv, linewidth=0.8)
        ax.axvline(
            0, color="red", linestyle="--",
            label="Estímulo láser"
        )
        ax.set_ylabel(f"{canal} (µV)")
        ax.grid(True, alpha=0.3)
        ax.legend()

    axes[-1].set_xlabel("Tiempo (s)")

    fig.suptitle(
        f"Época {indice + 1} | "
        f"Intensidad: {metadata['laser_power']} J | "
        f"Rating: {metadata['rating']}"
    )

    plt.tight_layout()
    plt.show()

# Calculamos y graficamos los potenciales evocados promedio por intensidad de estímulo
def graficar_potenciales_evocados(epochs):

    # Copia para no modificar los datos originales
    epocas = epochs.copy()
    epocas.apply_baseline(baseline=(-1, 0))

    intensidades = sorted(
        epocas.metadata["laser_power"].dropna().unique()
    )

    fig, axes = plt.subplots(
        len(CANALES), 1,
        figsize=(12, 9),
        sharex=True
    )

    # Selecciona las épocas que corresponden a una intensidad determinada
    # Se promedian las 20 épocas para cada intensidad
    # Se repite el procedimiento para las otras tres intensidades
    for ax, canal in zip(axes, CANALES):
        for intensidad in intensidades:
            seleccion = np.isclose(
                epocas.metadata["laser_power"].to_numpy(),
                intensidad
            )
            grupo = epocas[seleccion]
            if len(grupo) == 0:
                continue
            evoked = grupo.average(picks=[canal]) # Promedio de las épocas seleccionadas para el canal específico
            senal_uv = evoked.data[0] * 1e6
            ax.plot(
                evoked.times,
                senal_uv,
                label=f"{intensidad:g} J"
            )

        ax.axvline(
            0, color="black", linestyle="--",
            linewidth=0.8
        )
        ax.set_ylabel(f"{canal} (µV)")
        ax.grid(True, alpha=0.3)
        ax.legend(title="Intensidad")

    axes[-1].set_xlabel("Tiempo (s)")
    fig.suptitle(
        "Potenciales evocados por láser según intensidad"
    )

    plt.tight_layout()
    plt.show()

# Graficamos el dolor percibido según la intensidad del estímulo
def graficar_dolor(epochs):

    metadata = epochs.metadata

    resumen = (
        metadata.groupby("laser_power")["rating"]
        .agg(["mean", "std"])
        .sort_index()
    )

    plt.figure(figsize=(8, 5))

    plt.errorbar(
        resumen.index,
        resumen["mean"],
        yerr=resumen["std"].fillna(0),
        marker="o",
        capsize=5
    )

    plt.xlabel("Intensidad del láser (J)")
    plt.ylabel("Rating de dolor")
    plt.title("Dolor percibido según intensidad del estímulo")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    epochs = cargar_datos()
    graficar_epoca(epochs, indice=0)
    graficar_potenciales_evocados(epochs)
    graficar_dolor(epochs)
