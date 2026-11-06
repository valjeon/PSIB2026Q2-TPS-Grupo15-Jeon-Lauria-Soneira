
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from extras.anterior import config, datos

BANDAS = {"delta": (1, 4), "theta": (4, 8), "alfa": (8, 13), "beta": (13, 30), "gamma": (30, 100)}
CANALES_DEFECTO = ["Fz", "Cz", "Pz", "Fp1", "Fp2"]


def calcular_psd(raw, fmax=200.0, n_fft=2048):
    """PSD de Welch de los canales EEG. Devuelve (frecuencias, psd en uV^2/Hz, nombres de canal)."""
    psd = raw.compute_psd(method="welch", fmin=0.1, fmax=fmax, picks="eeg",
                          n_fft=n_fft, n_overlap=n_fft // 2, verbose="ERROR")
    return psd.freqs, psd.get_data() * 1e12, psd.ch_names


def _potencia(freqs, psd, f1, f2):
    m = (freqs >= f1) & (freqs < f2)
    integrar = getattr(np, "trapezoid", None) or np.trapz  # numpy 2 renombro trapz
    return integrar(psd[:, m], freqs[m], axis=1)


def _db_sobre_fondo(freqs, psd, f0, ancho=1.0, flanco=(3.0, 8.0)):
    """dB del pico en f0 (+-ancho) sobre la mediana de los flancos (f0-8..f0-3 y f0+3..f0+8), por canal."""
    pico = (freqs >= f0 - ancho) & (freqs <= f0 + ancho)
    fl = ((freqs >= f0 - flanco[1]) & (freqs <= f0 - flanco[0])) | \
         ((freqs >= f0 + flanco[0]) & (freqs <= f0 + flanco[1]))
    if not pico.any() or not fl.any():
        return np.full(psd.shape[0], np.nan)
    return 10 * np.log10(psd[:, pico].max(axis=1) / np.median(psd[:, fl], axis=1))


def diagnostico(raw, fmax=200.0):
    """Numeros que orientan la eleccion de filtros (mediana entre canales EEG)."""
    freqs, psd, _ = calcular_psd(raw, fmax=min(fmax, raw.info["sfreq"] / 2 - 1))
    total = _potencia(freqs, psd, 0.1, 100.0)
    bandas = {n: float(np.median(_potencia(freqs, psd, a, b) / total) * 100) for n, (a, b) in BANDAS.items()}
    return {
        "pasaaltos_amplificador_hz": raw.info["highpass"],
        "pasabajos_amplificador_hz": raw.info["lowpass"],
        "sfreq_hz": raw.info["sfreq"],
        "red_50hz_db": float(np.nanmedian(_db_sobre_fondo(freqs, psd, 50.0))),
        "red_100hz_db": float(np.nanmedian(_db_sobre_fondo(freqs, psd, 100.0))) if freqs[-1] > 105 else None,
        "deriva_pct_menor_1hz": float(np.median(_potencia(freqs, psd, 0.1, 1.0) / total) * 100),
        "bandas_pct_1_100hz": {k: round(v, 1) for k, v in bandas.items()},
    }


def graficar(raws_por_sesion, canales, ruta):
    """Izquierda: PSD media de todos los canales por sesion. Derecha: canales elegidos (sesion 1)."""
    fig, ax = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for nombre, raw in raws_por_sesion.items():
        f, p, _ = calcular_psd(raw, fmax=min(200.0, raw.info["sfreq"] / 2 - 1))
        ax[0].plot(f, 10 * np.log10(p.mean(axis=0)), lw=1, label=nombre)

        #p.mean hace el promedio entre canales y frecuencia por frecuencia
        # Con 10*np.log10() se pasa de µV²/Hz a dB
        
    primera = next(iter(raws_por_sesion.values()))
    f, p, nombres = calcular_psd(primera, fmax=min(200.0, primera.info["sfreq"] / 2 - 1))
    for c in canales:
        if c in nombres:
            ax[1].plot(f, 10 * np.log10(p[nombres.index(c)]), lw=1, label=c)
    for a in ax:
        a.set_xscale("log")
        a.set_xlim(0.3, 200)
        a.axvspan(*getattr(config, "BANDA_PASO", (1.0, 100.0)), color="g", alpha=0.06)
        for x in (1, 50, 100):
            a.axvline(x, color="k", ls="--", lw=0.7)
        a.set_xlabel("Frecuencia (Hz)")
        a.grid(alpha=0.3, which="both")
        a.legend(fontsize=8)
    ax[0].set_ylabel("Densidad espectral (dB re 1 uV²/Hz)")
    ax[0].set_title("Media de los canales EEG, por sesion")
    ax[1].set_title(f"Canales elegidos ({next(iter(raws_por_sesion))})")
    fig.suptitle("Espectro del EEG crudo")
    fig.tight_layout()
    fig.savefig(ruta, dpi=130)
    plt.close(fig)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("sujetos", nargs="+")
    p.add_argument("--canales", nargs="+", default=CANALES_DEFECTO)
    a = p.parse_args(argv)
    salida = Path(config.CARPETA_SALIDA)
    salida.mkdir(parents=True, exist_ok=True)
    for sub in a.sujetos:
        raws, nombres = datos.cargar_sujeto(sub)
        print(f"\n== {sub} ==")
        # for nombre, raw in zip(nombres, raws):
        #     d = diagnostico(raw)
        #     print(f"{nombre}: amplificador {d['pasaaltos_amplificador_hz']:g}-{d['pasabajos_amplificador_hz']:g} Hz | "
        #           f"red 50 Hz: +{d['red_50hz_db']:.1f} dB | "
        #           f"100 Hz: {'+%.1f dB' % d['red_100hz_db'] if d['red_100hz_db'] is not None else 'n/d'} | "
        #           f"deriva <1 Hz: {d['deriva_pct_menor_1hz']:.1f}% | bandas %: {d['bandas_pct_1_100hz']}")
        ruta = salida / f"{sub}_espectro_crudo.png"
        graficar(dict(zip(nombres, raws)), a.canales, ruta)
        print("Figura:", ruta)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
