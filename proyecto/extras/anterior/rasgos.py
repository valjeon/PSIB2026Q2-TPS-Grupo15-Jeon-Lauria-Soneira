"""Rasgos por ensayo, a partir de las epocas limpias de un sujeto.

Una fila por epoca limpia (con su metadata: sujeto, sesion, nivel, condicion, ...) y columnas:

  N2-P2 en Cz (potencial evocado de cada ensayo)
      N2_uV, N2_lat_ms   minimo en VENTANA_N2        (medido sobre una copia con pasa-bajos de 30 Hz,
      P2_uV, P2_lat_ms   maximo en VENTANA_P2         igual que qc.py y artefactos.py)
      N2P2_uV            P2 - N2
  Potencia por banda (delta, theta, alfa, beta, gamma) en CANALES_RASGOS
      <canal>_<banda>_post_dB     10*log10 de la potencia (uV^2) en VENTANA_POST
      <canal>_<banda>_cambio_dB   post - pre, en dB (negativo = baja respecto del pre-estimulo)

Los valores de abajo son una PRIMERA ELECCION razonable, no un resultado: se ajustan mirando los
datos. La potencia se mide sobre la senal 1-100 Hz (sin el pasa-bajos de 30 Hz) para poder ver gamma.

Entrada: sub-XXX_limpias-epo.fif (de artefactos.py). Salida en derived_mine/sub-XXX/:
    sub-XXX_rasgos.csv
y con --reunir, derived_mine/rasgos_todos.csv (todos los sujetos, para el clasificador).

Uso desde la carpeta que contiene `procesamiento/`:
    python -m extras.anterior.rasgos sub-001 sub-002
    python -m extras.anterior.rasgos --reunir
"""
import argparse
from pathlib import Path

import mne
import numpy as np
import pandas as pd

from extras.anterior import artefactos, config

CANALES_RASGOS = ["Fz", "Cz", "Pz", "C3", "C4"]
BANDAS = {"delta": (1, 4), "theta": (4, 8), "alfa": (8, 13), "beta": (13, 30), "gamma": (30, 100)}
VENTANA_POST = (0.0, 1.0)      # s respecto del pulso
VENTANA_PRE = (-1.0, 0.0)      # s: mismo largo que la post, para que la comparacion sea justa


# ---------------------------------------------------------------------------
# N2-P2 por ensayo
# ---------------------------------------------------------------------------

def n2p2_por_ensayo(ep, canal=None, ventana_n2=None, ventana_p2=None):
    """N2 (minimo) y P2 (maximo) de cada epoca en un canal. Devuelve un DataFrame (una fila por epoca).

    Sobre una copia con pasa-bajos de 30 Hz. En un ensayo suelto el pico es ruidoso: el minimo y el
    maximo siempre existen, aunque no haya LEP (E1 casi no lo tiene), asi que N2-P2 nunca es cero.
    """
    canal = config.CANAL_QC if canal is None else canal
    a_n2, b_n2 = config.VENTANA_N2 if ventana_n2 is None else ventana_n2
    a_p2, b_p2 = config.VENTANA_P2 if ventana_p2 is None else ventana_p2

    e = ep.copy().filter(None, config.PASA_BAJOS_LEP_HZ, picks="eeg", verbose="ERROR")
    x = e.get_data(picks=canal)[:, 0, :] * 1e6
    t = e.times
    out = {}
    for nombre, (a, b), buscar in (("N2", (a_n2, b_n2), np.argmin), ("P2", (a_p2, b_p2), np.argmax)):
        idx = np.flatnonzero((t >= a) & (t <= b))
        pos = idx[buscar(x[:, idx], axis=1)]
        out[f"{nombre}_uV"] = x[np.arange(len(x)), pos]
        out[f"{nombre}_lat_ms"] = t[pos] * 1000
    out["N2P2_uV"] = out["P2_uV"] - out["N2_uV"]
    return pd.DataFrame(out)


# ---------------------------------------------------------------------------
# Potencia por bandas
# ---------------------------------------------------------------------------

def potencia_por_banda(ep, canales, ventana, bandas=None):
    """Potencia (uV^2) por epoca, canal y banda, en una ventana. Devuelve (n_epocas, n_canales, n_bandas).

    Welch con un solo segmento del largo de la ventana: la resolucion en frecuencia es 1/duracion
    (1 Hz con una ventana de 1 s), asi que delta (1-4 Hz) queda con pocos puntos.
    """
    bandas = BANDAS if bandas is None else bandas
    n_fft = int(round((ventana[1] - ventana[0]) * ep.info["sfreq"]))
    espectro = ep.compute_psd(method="welch", fmin=1.0, fmax=100.0, tmin=ventana[0], tmax=ventana[1],
                              picks=canales, n_fft=n_fft, verbose="ERROR")
    psd, freqs = espectro.get_data(return_freqs=True)          # (epocas, canales, frecuencias), V^2/Hz
    df = float(np.mean(np.diff(freqs)))
    psd = psd * 1e12                                           # uV^2/Hz
    return np.stack([psd[..., (freqs >= a) & (freqs < b)].sum(axis=-1) * df
                     for a, b in bandas.values()], axis=-1)


def rasgos_de_potencia(ep, canales=None, bandas=None):
    """DataFrame con la potencia post (dB) y el cambio post - pre (dB) por canal y banda."""
    canales = [c for c in (CANALES_RASGOS if canales is None else canales) if c in ep.ch_names]
    bandas = BANDAS if bandas is None else bandas
    post = potencia_por_banda(ep, canales, VENTANA_POST, bandas)
    pre = potencia_por_banda(ep, canales, VENTANA_PRE, bandas)
    cols = {}
    for i, canal in enumerate(canales):
        for j, banda in enumerate(bandas):
            cols[f"{canal}_{banda}_post_dB"] = 10 * np.log10(post[:, i, j])
            cols[f"{canal}_{banda}_cambio_dB"] = 10 * np.log10(post[:, i, j] / pre[:, i, j])
    return pd.DataFrame(cols)


# ---------------------------------------------------------------------------
# Tabla por sujeto
# ---------------------------------------------------------------------------

def tabla_rasgos(ep):
    """Una fila por epoca: metadata + N2-P2 en Cz + potencia por bandas."""
    if ep.metadata is None:
        raise ValueError("Las epocas no tienen metadata (se necesita sujeto, nivel y condicion)")
    meta = ep.metadata.reset_index(drop=True)
    return pd.concat([meta, n2p2_por_ensayo(ep), rasgos_de_potencia(ep)], axis=1)


def ruta_rasgos(sub, carpeta=None):
    return artefactos.rutas(sub, carpeta)["limpias"].with_name(f"{sub}_rasgos.csv")


def rasgos_sujeto(sub, carpeta=None):
    """Lee las epocas limpias del sujeto, calcula los rasgos y los guarda. Devuelve la tabla."""
    limpias = artefactos.rutas(sub, carpeta)["limpias"]
    if not limpias.exists():
        raise FileNotFoundError(f"No esta {limpias}. Corre antes: python -m extras.anterior.artefactos {sub}")
    tabla = tabla_rasgos(mne.read_epochs(limpias, verbose="ERROR"))
    ruta = ruta_rasgos(sub, carpeta)
    tabla.to_csv(ruta, index=False)
    return tabla


def reunir_rasgos(carpeta=None):
    """Junta los rasgos de todos los sujetos en rasgos_todos.csv. Devuelve la tabla (o None si no hay)."""
    carpeta = Path(config.CARPETA_SALIDA if carpeta is None else carpeta)
    archivos = sorted(carpeta.glob("sub-*/sub-*_rasgos.csv"))
    if not archivos:
        return None
    todos = pd.concat([pd.read_csv(a) for a in archivos], ignore_index=True)
    todos.to_csv(carpeta / "rasgos_todos.csv", index=False)
    return todos


def main(argv=None):
    p = argparse.ArgumentParser(description="Rasgos por ensayo (N2-P2 y potencia por bandas)")
    p.add_argument("sujetos", nargs="*")
    p.add_argument("--reunir", action="store_true", help="juntar los rasgos de todos los sujetos")
    a = p.parse_args(argv)
    if not a.sujetos and not a.reunir:
        p.error("indicar sujetos y/o --reunir")
    for s in a.sujetos:
        t = rasgos_sujeto(s)
        print(f"{s}: {len(t)} ensayos, {t.shape[1]} columnas -> {ruta_rasgos(s)}")
        print(t.groupby("condicion")["N2P2_uV"].agg(["count", "mean", "std"]).round(1).to_string())
    if a.reunir:
        todos = reunir_rasgos()
        print("Sin rasgos para reunir" if todos is None else f"rasgos_todos.csv: {len(todos)} filas, "
              f"{todos['sujeto'].nunique()} sujetos")


if __name__ == "__main__":
    main()
