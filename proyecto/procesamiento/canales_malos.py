"""Canales malos que no marcaron los autores.

Mismo criterio para todos los sujetos, sin usar el nivel (E1..E4):
    1. Se sacan las epocas malas de los autores y se dejan afuera sus electrodos malos (se interpolan despues).
    2. Medida: pico a pico de cada canal en el PRE-estimulo (-1 a 0 s), epoca por epoca, con la referencia
       original (nariz). Antes del pulso no hay LEP, asi que mide ruido. Sin pasa-bajos: el ruido muscular
       (alta frecuencia) tambien cuenta.
    3. z robusto entre canales (mediana y MAD), sin los frontopolares (el ojo siempre da alto) ni el VEO.
    4. Por canal:
         z del pico a pico tipico (mediana entre epocas) > Z_CANAL      -> ruidoso
         pico a pico tipico < FACTOR_PLANO x mediana de los canales    -> plano (desconectado)
         % de epocas en que el canal es atipico (z > Z_CANAL en esa epoca)
       Decision:
         'malo'    plano, o ruidoso en mas de FRAC_EPOCAS de las epocas -> se interpola
         'epocas'  atipico solo en algunas epocas -> lo resuelve el rechazo de epocas (no se interpola)
    5. Registro en derived_mine/malos_propios.csv (una fila por canal marcado) y una figura por canal
       (ensayo por ensayo) en derived_mine/sub-XXX/canales_malos/ para la revision a mano.

Los umbrales se fijan una vez (config.py, con estos valores por defecto) antes de correr los 30 sujetos.

Uso desde proyecto/:
    python -m procesamiento.canales_malos sub-010 sub-025 sub-050
    python -m procesamiento.canales_malos                (todos los sujetos segmentados)
"""
import argparse
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd

from procesamiento import config, marcas_autores, segmentacion

Z_CANAL = getattr(config, "Z_CANAL_MALO", 5.0)
FRAC_EPOCAS = getattr(config, "FRAC_EPOCAS_CANAL_MALO", 0.5)
FACTOR_PLANO = getattr(config, "FACTOR_CANAL_PLANO", 0.1)
PCT_MIN_EPOCAS = 5                            # % minimo de epocas atipicas para reportar el canal
VENTANA_RUIDO = (config.TMIN, 0.0)            # pre-estimulo
COLUMNAS = ["sujeto", "canal", "decision", "z_tipico", "pct_epocas_atipicas", "ptp_tipico_uV",
            "ptp_mediana_canales_uV", "motivo"]


def _z_robusto(x, eje):
    med = np.median(x, axis=eje, keepdims=True)
    mad = np.median(np.abs(x - med), axis=eje, keepdims=True) * 1.4826
    return (x - med) / np.where(mad > 0, mad, np.inf)


def evaluar(ep, excluir=()):
    """DataFrame con una fila por canal EEG evaluado. `ep`: epocas con referencia original.

    `excluir`: canales que no se evaluan ni entran en la referencia del z (malos de los autores).
    """
    fuera = set(excluir) | set(config.CANALES_SIN_CRITERIO_EEG)
    canales = [c for c in ep.copy().pick("eeg").ch_names if c not in fuera]
    x = ep.copy().crop(*VENTANA_RUIDO).get_data(picks=canales) * 1e6     # epocas x canales x tiempo
    ptp = x.max(axis=2) - x.min(axis=2)                                    # epocas x canales

    tipico = np.median(ptp, axis=0)
    z_tip = _z_robusto(tipico, eje=0)
    atipica = _z_robusto(ptp, eje=1) > Z_CANAL                             # z entre canales, en cada epoca
    pct = atipica.mean(axis=0) * 100
    mediana = float(np.median(tipico))

    d = pd.DataFrame({"canal": canales, "z_tipico": z_tip.round(1), "pct_epocas_atipicas": pct.round(0),
                      "ptp_tipico_uV": tipico.round(1), "ptp_mediana_canales_uV": round(mediana, 1)})
    plano = d["ptp_tipico_uV"] < FACTOR_PLANO * mediana
    malo = (d["z_tipico"] > Z_CANAL) & (d["pct_epocas_atipicas"] > FRAC_EPOCAS * 100)
    algunas = (d["pct_epocas_atipicas"] >= PCT_MIN_EPOCAS) & ~malo & ~plano
    d["decision"] = np.select([plano, malo, algunas], ["malo", "malo", "epocas"], default="ok")
    d["motivo"] = np.select(
        [plano, malo, algunas],
        [f"plano (< {FACTOR_PLANO:g} x mediana)",
         f"ruidoso: z > {Z_CANAL:g} y atipico en > {FRAC_EPOCAS:.0%} de las epocas",
         "atipico solo en algunas epocas"], default="")
    return d


def figura_canal(ep, canal, sub, fila):
    """Ensayo por ensayo del canal (pre y post estimulo) para la revision a mano."""
    figs = ep.plot_image(picks=canal, show=False, colorbar=True, evoked=True,
                         title=f"{sub} {canal}: {fila['decision']} (z={fila['z_tipico']}, "
                               f"{fila['pct_epocas_atipicas']:.0f}% epocas atipicas)")
    return figs[0]


def guardar_registro(tabla, sub, carpeta):
    """Reemplaza las filas de `sub` en malos_propios.csv (uno solo para todos los sujetos)."""
    ruta = carpeta / "malos_propios.csv"
    marcados = tabla[tabla["decision"] != "ok"].assign(sujeto=sub)[COLUMNAS]
    if ruta.exists():
        viejo = pd.read_csv(ruta)
        marcados = pd.concat([viejo[viejo["sujeto"] != sub], marcados], ignore_index=True)
    marcados.sort_values(["sujeto", "decision", "canal"]).to_csv(ruta, index=False)
    return ruta


def malos_propios(sub, carpeta=None):
    """Canales con decision 'malo' para `sub` segun malos_propios.csv (lista vacia si no hay registro)."""
    ruta = Path(config.CARPETA_SALIDA if carpeta is None else carpeta) / "malos_propios.csv"
    if not ruta.exists():
        return []
    t = pd.read_csv(ruta)
    return t[(t["sujeto"] == sub) & (t["decision"] == "malo")]["canal"].tolist()


def revisar_sujeto(sub, carpeta=None):
    carpeta = Path(config.CARPETA_SALIDA if carpeta is None else carpeta)
    m = marcas_autores.marcas_de(sub)
    ep = mne.read_epochs(segmentacion.ruta_epocas(sub, carpeta), verbose="ERROR")
    if m.epocas and len(ep) == 4 * config.PULSOS_POR_NIVEL:
        ep = ep[~ep.metadata["ensayo"].isin(m.epocas).to_numpy()]
    tabla = evaluar(ep, excluir=m.electrodos)

    print(f"\n=== {sub} ({len(ep)} epocas; malos de los autores: {m.electrodos or 'ninguno'}) ===")
    marcados = tabla[tabla["decision"] != "ok"].sort_values("z_tipico", ascending=False)
    if marcados.empty:
        print("  sin canales marcados")
    else:
        print(marcados[["canal", "decision", "z_tipico", "pct_epocas_atipicas", "ptp_tipico_uV"]]
              .to_string(index=False))
    print(f"  pico a pico tipico (mediana de los canales): {tabla['ptp_mediana_canales_uV'].iloc[0]} uV")
    top = tabla.nlargest(5, "z_tipico")
    print("  z mas altos: " + ", ".join(f"{r.canal} {r.z_tipico} ({r.pct_epocas_atipicas:.0f}%)"
                                        for r in top.itertuples()))

    figuras = carpeta / sub / "canales_malos"
    for _, fila in marcados.iterrows():
        figuras.mkdir(parents=True, exist_ok=True)
        fig = figura_canal(ep, fila["canal"], sub, fila)
        fig.savefig(figuras / f"{sub}_{fila['canal']}.png", dpi=110)
        plt.close(fig)
    ruta = guardar_registro(tabla, sub, carpeta)
    return tabla, ruta


def main(argv=None):
    p = argparse.ArgumentParser(description="Paso 2: canales malos propios (registro en malos_propios.csv)")
    p.add_argument("sujetos", nargs="*", help="por defecto, todos los que tienen epocas segmentadas")
    a = p.parse_args(argv)
    matplotlib.use("Agg")
    print(f"Umbrales: z > {Z_CANAL:g}, atipico en > {FRAC_EPOCAS:.0%} de las epocas, plano < {FACTOR_PLANO:g} x mediana")
    ruta = None
    for sub in a.sujetos or marcas_autores.sujetos_segmentados():
        _, ruta = revisar_sujeto(sub)
    if ruta:
        print(f"\nRegistro: {ruta}\nFiguras: derived_mine/sub-XXX/canales_malos/")


if __name__ == "__main__":
    main()