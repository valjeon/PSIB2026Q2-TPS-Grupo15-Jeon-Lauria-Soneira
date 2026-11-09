# Se usa un filtro FIR, como se decidió en comparacion_filtros.py

from pathlib import Path
import mne


# Parámetros
RAIZ = Path(__file__).resolve().parent.parent / "data" / "ds005293"
SALIDA = RAIZ / "derivatives" / "filtered"
SOBREESCRIBIR = False           # esto se puede cambiar a True si queremos subir algún cambio

CANALES_EOG = ["VEO", "HEOR", "HEOL"]

# Pasabanda: frecuencias de -6 dB y ancho de transición (Hz)
F_ALTA_6DB, TRANS_ALTA = 1.0, 1.0      # pasa-altos en 1 Hz
F_BAJA_6DB, TRANS_BAJA = 100.0, 25.0   # pasa-bajos en 100 Hz

# Notch: rechazo total entre 48 y 52 Hz (el -6 dB queda en 47,5 y 52,5 Hz)
F_NOTCH, ANCHO_NOTCH, TRANS_NOTCH = 50.0, 4.0, 1.0

# Bordes de banda pasante que espera MNE
L_FREQ = F_ALTA_6DB + TRANS_ALTA / 2
H_FREQ = F_BAJA_6DB - TRANS_BAJA / 2


def filtrar_raw(raw):
    """Marca los canales EOG y aplica pasabanda + notch FIR. Modifica raw."""
    eog = {c: "eog" for c in CANALES_EOG if c in raw.ch_names}
    raw.set_channel_types(eog)

    raw.filter(
        l_freq=L_FREQ, h_freq=H_FREQ,
        l_trans_bandwidth=TRANS_ALTA, h_trans_bandwidth=TRANS_BAJA,
        method="fir", fir_design="firwin", verbose="ERROR",
    )
    raw.notch_filter(
        freqs=F_NOTCH, notch_widths=ANCHO_NOTCH, trans_bandwidth=TRANS_NOTCH,
        method="fir", fir_design="firwin", verbose="ERROR",
    )
    return raw


def filtrar_archivo(vhdr):
    raw = mne.io.read_raw_brainvision(str(vhdr), preload=True, verbose="ERROR")
    return filtrar_raw(raw)


def buscar_archivos():
    return sorted((RAIZ / "sub-001").glob("ses-*/eeg/*_eeg.vhdr"))


def main():
    SALIDA.mkdir(parents=True, exist_ok=True)

    archivos = buscar_archivos()
    print(f"Archivos encontrados: {len(archivos)}")
    for a in archivos:
        print(" -", a.relative_to(RAIZ))

    for i, vhdr in enumerate(archivos, start=1):
        destino = SALIDA / (vhdr.stem.replace("_eeg", "_desc-filt_raw") + ".fif")

        if destino.exists():
            print(f"[{i}/{len(archivos)}] Ya existe, salto: {destino.name}")
            continue

        print(f"[{i}/{len(archivos)}] Filtrando {vhdr.name} ...")
        raw = filtrar_archivo(vhdr)
        raw.save(destino, overwrite=SOBREESCRIBIR, verbose="ERROR")
        print(f"    Guardado: {destino.name}")
        del raw

    print("\nListo.")


if __name__ == "__main__":
    main()

# Señal sin filtrar --> se ve que tiene el offset de continua, centrada alrededor de 30 microV
# se ve como una señal "gruesa" por el ruido de línea
# Señal filtrada --> se ve que esta alrededor de 0 microV (por el pasa altos de 1 Hz)