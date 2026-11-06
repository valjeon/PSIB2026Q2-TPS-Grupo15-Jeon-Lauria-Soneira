# Filtros a comparar:
# 1) FIR de fase lineal (MNE, firwin)
# 2) IIR Butterworth de fase cero (MNE, filtfilt)
# 3) IIR Butterworth causal (MNE, una sola pasada)

# En todos los casos se aplica: pasabanda 1-100 Hz + rechazo de banda 48-52 Hz.

from pathlib import Path
import numpy as np
import mne
import matplotlib.pyplot as plt
import scipy.signal as ss

ARCHIVO = (
    Path(__file__).resolve().parent.parent
    / "data" / "ds005293" / "sub-001" / "ses-1" / "eeg"
    / "sub-001_ses-1_task-95ByBP_eeg.vhdr"
)

CANALES_EOG = ["VEO", "HEOR", "HEOL"]
TRAMO = (100, 160)               # segundos a analizar
CANAL_GRAFICO = "Cz"
VENTANA_GRAFICO = (20, 24)       # segundos, relativos al inicio del tramo recortado

F_ALTA, F_BAJA = 1.0, 100.0      # pasabanda
F_NOTCH, ANCHO_NOTCH = 50.0, 4.0 # notch en 48-52 Hz
ORDEN_IIR = 4                    # se suele usar este orden para análisis de EEG

IIR_PARAMS = dict(order=ORDEN_IIR, ftype="butter", output="sos")


# Carga de datos, se recorta la señal según el tramo seleccionado
def cargar_raw():
    raw = mne.io.read_raw_brainvision(str(ARCHIVO), preload=True, verbose="ERROR")
    raw.set_channel_types({c: "eog" for c in CANALES_EOG if c in raw.ch_names})
    return raw.crop(*TRAMO)


# Filtros, cada uno devuelve un raw nuevo
def filtrar_fir(raw):
    r = raw.copy()
    r.filter(l_freq=1.5, h_freq=87.5, l_trans_bandwidth=1.0, h_trans_bandwidth=25.0, method="fir", fir_design="firwin", verbose="ERROR")
    r.notch_filter(F_NOTCH, notch_widths=ANCHO_NOTCH, method="fir",
                   fir_design="firwin", verbose="ERROR")
    return r


def filtrar_iir_mne(raw, fase="zero"):
    """IIR Butterworth de MNE. fase='zero' (filtfilt) o 'forward' (causal)."""
    r = raw.copy()
    r.filter(F_ALTA, F_BAJA, method="iir", iir_params=dict(IIR_PARAMS),
             phase=fase, verbose="ERROR")
    r.notch_filter(F_NOTCH, notch_widths=ANCHO_NOTCH, method="iir",
                   iir_params=dict(IIR_PARAMS), phase=fase, verbose="ERROR")
    return r


# Análisis de respuesta en frecuencia y al impulso
def graficar_respuestas(sfreq):
    disenios = {
        "FIR 1-100 Hz": mne.filter.create_filter(
            None, sfreq, F_ALTA, F_BAJA, method="fir", fir_design="firwin"),
        f"IIR Butterworth orden {ORDEN_IIR}, 1-100 Hz": mne.filter.create_filter(
            None, sfreq, F_ALTA, F_BAJA, method="iir", iir_params=dict(IIR_PARAMS)),
    }
    for titulo, h in disenios.items():
        mne.viz.plot_filter(h, sfreq, flim=(0.1, 150), fscale="log", title=titulo)

    notch = {
        "Notch FIR 48-52 Hz": mne.filter.create_filter(
            None, sfreq, F_NOTCH + ANCHO_NOTCH / 2, F_NOTCH - ANCHO_NOTCH / 2,
            l_trans_bandwidth=1.0, h_trans_bandwidth=1.0,
            method="fir", fir_design="firwin"),
        "Notch IIR 48-52 Hz": mne.filter.create_filter(
            None, sfreq, F_NOTCH + ANCHO_NOTCH / 2, F_NOTCH - ANCHO_NOTCH / 2,
            method="iir", iir_params=dict(IIR_PARAMS)),
    }
    for titulo, h in notch.items():
        mne.viz.plot_filter(h, sfreq, flim=(30, 70), fscale="linear", title=titulo)


def respuesta_impulso(sfreq, filtrar, n=int(20 * 1000)):
    """Respuesta de un candidato a un impulso unitario en el centro."""
    imp = np.zeros((1, n))
    imp[0, n // 2] = 1.0
    r = mne.io.RawArray(imp, mne.create_info(["x"], sfreq, "eeg"), verbose="ERROR")
    return filtrar(r).get_data()[0]


def duracion_efectiva(h, fs, umbral=0.01):
    """Tiempo entre la primera y la última muestra por encima del umbral del pico."""
    idx = np.where(np.abs(h) > umbral * np.abs(h).max())[0]
    return (idx[-1] - idx[0]) / fs


# Métricas y gráficos resultantes
def rms(x):
    return float(np.sqrt(np.mean(x ** 2)))


def uv(raw):
    return raw.get_data(picks="eeg") * 1e6


def imprimir_metricas(candidatos, impulsos, sfreq):
    ref = uv(candidatos["FIR"])
    print(f"\nRMS de la señal FIR: {rms(ref):.2f} µV")
    for nombre, r in candidatos.items():
        if nombre == "FIR":
            continue
        print(f"RMS(FIR - {nombre}): {rms(ref - uv(r)):.3f} µV")

    print("\nDuración efectiva de la respuesta al impulso (>1% del pico):")
    for nombre, h in impulsos.items():
        print(f"  {nombre}: {duracion_efectiva(h, sfreq):.2f} s")


def graficar_tramo(candidatos):
    ini, fin = VENTANA_GRAFICO
    fig, ax = plt.subplots(figsize=(12, 4))
    for nombre, r in candidatos.items():
        y = r.copy().crop(ini, fin).get_data(picks=CANAL_GRAFICO)[0] * 1e6
        ax.plot(np.linspace(ini, fin, len(y)), y, label=nombre, linewidth=0.8)
    ax.set(xlabel="Tiempo (s)", ylabel=f"{CANAL_GRAFICO} (µV)")
    ax.legend()
    fig.tight_layout()


def graficar_impulsos(impulsos, sfreq):
    fig, ax = plt.subplots(figsize=(12, 3))
    for nombre, h in impulsos.items():
        t = (np.arange(len(h)) - len(h) // 2) / sfreq
        ax.plot(t, h, label=nombre, linewidth=0.8)
    ax.set(xlim=(-1, 1), xlabel="Tiempo (s)", title="Respuesta al impulso")
    ax.legend()
    fig.tight_layout()


def graficar_espectros(candidatos, original):
    fig, ax = plt.subplots(figsize=(10, 4))
    for nombre, r in {"Original": original, **candidatos}.items():
        psd = r.compute_psd(fmax=120, picks="eeg", verbose="ERROR")
        ax.plot(psd.freqs, 10 * np.log10(psd.get_data().mean(axis=0) * 1e12),
                label=nombre, linewidth=0.9)
    ax.set(xlabel="Frecuencia (Hz)", ylabel="PSD media (dB re 1 µV²/Hz)")
    ax.legend()
    fig.tight_layout()


def main():
    raw = cargar_raw()
    sfreq = raw.info["sfreq"]

    candidatos = {
        "FIR": filtrar_fir(raw),
        "IIR MNE (filtfilt)": filtrar_iir_mne(raw, "zero"),
        "IIR causal": filtrar_iir_mne(raw, "forward"),
    }
    impulsos = {
        "FIR": respuesta_impulso(sfreq, filtrar_fir),
        "IIR (filtfilt)": respuesta_impulso(sfreq, filtrar_iir_mne),
    }

    imprimir_metricas(candidatos, impulsos, sfreq)

    graficar_respuestas(sfreq)
    graficar_espectros(candidatos, raw)
    graficar_tramo(candidatos)
    graficar_impulsos(impulsos, sfreq)
    plt.show()


if __name__ == "__main__":
    main()

# Elegimos FIR, habría que justificar bien porque pero en el paper usan FIR