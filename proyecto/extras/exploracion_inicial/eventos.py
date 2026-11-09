
from pathlib import Path
import pandas as pd

# Carpeta de los datos descargados
CARPETA_EEG = (
    Path(__file__).resolve().parent
    / "PSIB_Grupo15_EEG"
    / "data"
    / "ds005293"
    / "sub-001"
    / "ses-1"
    / "eeg"
)

# Buscar la tabla de eventos
archivos_eventos = list(
    CARPETA_EEG.glob("*_events.tsv")
)

if not archivos_eventos:
    raise FileNotFoundError(
        f"No se encontró events.tsv en {CARPETA_EEG}"
    )

archivo_eventos = archivos_eventos[0]

print("Archivo de eventos:")
print(archivo_eventos.name)

# Leer la tabla
eventos = pd.read_csv(
    archivo_eventos,
    sep="\t",
    na_values=["n/a", "NA", ""]
)

print("\nDimensiones de la tabla:")
print(eventos.shape)

print("\nColumnas disponibles:")
print(eventos.columns.tolist())

print("\nPrimeras 20 filas:")
print(eventos.head(20).to_string(index=False))

# Mostrar las etiquetas y su frecuencia
for columna in eventos.columns:
    if eventos[columna].dtype == "object":
        print(f"\nValores de la columna '{columna}':")
        print(
            eventos[columna]
            .value_counts(dropna=False)
            .head(30)
        )

# Comprobar el rango temporal
if "onset" in eventos.columns:
    print("\nPrimer onset:", eventos["onset"].min())
    print("Último onset:", eventos["onset"].max())
