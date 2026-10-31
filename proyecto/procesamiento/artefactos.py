"""Artefactos: rechazo de epocas contaminadas (sin ICA, segun la consigna).

Se descarta una epoca si:
    - hay un parpadeo en la ventana critica: VEO con pico a pico > UMBRAL_VEO_UV dentro de
      VENTANA_VEO (ahi estan N2 y P2). Un parpadeo fuera de la ventana no molesta.
    - hay un artefacto grande en el EEG: pico a pico > UMBRAL_RECHAZO_UV en algun canal
      (sin contar los frontopolares de CANALES_SIN_CRITERIO_EEG: ahi se ven los ojos, y eso ya
      lo decide VEO; si no, un parpadeo fuera de la ventana critica descartaria la epoca igual).
No modifica la senal: solo elige que ensayos se usan.

El pico a pico se mide sobre una copia con pasa-bajos de 30 Hz (la banda de los LEPs), igual que
en qc.py; la senal guardada sigue siendo 1-100 Hz.

Entrada: sub-XXX_laser-epo.fif (de segmentacion.py). Salida en derived_mine/sub-XXX/:
    sub-XXX_limpias-epo.fif     epocas limpias, referencia promedio
    sub-XXX_rechazo.csv         epocas por nivel: antes, rechazadas por VEO, por EEG, despues
    limpieza/*.png              figuras de control

Uso desde proyecto/:
    python -m procesamiento.artefactos sub-001
"""
import argparse
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from procesamiento import config, marcas_autores, qc, segmentacion


# Medidas


def con_referencia_promedio(ep):
    """Copia con referencia promedio (solo canales EEG; VEO queda como esta)."""
    return ep.copy().set_eeg_reference("average", projection=False, verbose="ERROR")


def _pasa_bajos(ep):
    return ep.copy().filter(None, config.PASA_BAJOS_LEP_HZ, picks=["eeg", "eog"], verbose="ERROR")


def pico_a_pico_uV(ep, picks="eeg", ventana=None, filtrar=True):
    """Maximo pico a pico por epoca (sobre los canales `picks`), en uV.

    ventana: (tmin, tmax) en s, o None para toda la epoca.
    filtrar: medir sobre una copia con pasa-bajos de 30 Hz (no cuenta ruido muscular de alta frecuencia).
    """
    e = _pasa_bajos(ep) if filtrar else ep
    if ventana is not None:
        e = e.copy().crop(*ventana)
    x = e.get_data(picks=picks)
    return (x.max(axis=2) - x.min(axis=2)).max(axis=1) * 1e6



# Rechazo


def rechazar(ep, umbral_veo_uV=None, ventana_veo=None, umbral_eeg_uV=None, usar_veo=True):
    """Descarta epocas con parpadeo en la ventana critica y/o artefacto grande en el EEG.

    Devuelve (epocas que quedan, tabla por nivel, DataFrame con el pico a pico de cada epoca).
    La metadata se recorta sola con las epocas.
    """
    umbral_veo = config.UMBRAL_VEO_UV if umbral_veo_uV is None else umbral_veo_uV
    ventana_veo = config.VENTANA_VEO if ventana_veo is None else ventana_veo
    umbral_eeg = config.UMBRAL_RECHAZO_UV if umbral_eeg_uV is None else umbral_eeg_uV

    # El criterio EEG no mira los frontopolares: ahi se ven los ojos, y de eso se encarga VEO.
    # Si los incluyera, un parpadeo FUERA de la ventana critica descartaria la epoca igual
    # (paso en sub-002: 73 de 80 epocas rechazadas por Fp1/Fp2).
    excluir = set(config.CANALES_SIN_CRITERIO_EEG)
    canales_eeg = [ep.ch_names[i] for i in mne.pick_types(ep.info, eeg=True) if ep.ch_names[i] not in excluir]
    medidas = pd.DataFrame({"condicion": ep.metadata["condicion"].to_numpy(),
                            "ptp_eeg_uV": pico_a_pico_uV(ep, canales_eeg)})
    tiene_veo = usar_veo and config.CANAL_EOG in ep.ch_names
    medidas["ptp_veo_uV"] = (pico_a_pico_uV(ep, config.CANAL_EOG, ventana_veo) if tiene_veo else np.nan)
    medidas["por_veo"] = medidas["ptp_veo_uV"] > umbral_veo if tiene_veo else False
    medidas["por_eeg"] = medidas["ptp_eeg_uV"] > umbral_eeg
    malas = np.flatnonzero(medidas["por_veo"] | medidas["por_eeg"])

    limpio = ep.copy().drop(malas, reason="artefacto", verbose="ERROR")

    niveles = list(config.NIVELES)
    g = medidas.groupby("condicion")
    tabla = pd.DataFrame({
        "antes": g.size(),
        "por_VEO": g["por_veo"].sum(),
        "por_EEG": g["por_eeg"].sum(),
        "despues": limpio.metadata["condicion"].value_counts(),
    }).reindex(niveles).fillna(0).astype(int).rename_axis("condicion").reset_index()
    tabla["rechazadas"] = tabla["antes"] - tabla["despues"]
    return limpio, tabla, medidas


# Figuras



def fig_antes_despues(antes, despues, canales=("Fp1", "Fz", "Cz"), titulo=""):
    """ERP por nivel con todas las epocas (punteado) y sin las rechazadas (lleno)."""
    canales = [c for c in canales if c in antes.ch_names]
    a, d = qc.copia_para_ver(antes), qc.copia_para_ver(despues)
    fig, axes = plt.subplots(len(canales), 1, figsize=(9, 2.6 * len(canales)), sharex=True)
    for ax, canal in zip(np.atleast_1d(axes), canales):
        for c in config.NIVELES:
            if c in set(a.metadata["condicion"]):
                ax.plot(a.times * 1000, a[c].average().get_data(picks=canal)[0] * 1e6,
                        color=qc.COLORES[c], lw=1, ls=":")
            if c in set(d.metadata["condicion"]):
                ax.plot(d.times * 1000, d[c].average().get_data(picks=canal)[0] * 1e6,
                        color=qc.COLORES[c], lw=1.6, label=f"{c} (n={len(d[c])})")
        ax.axvline(0, color="k", lw=0.8)
        ax.axhline(0, color="k", lw=0.5)
        ax.set_ylabel(f"{canal} (uV)")
    ax.set_xlim(-200, 1000)
    ax.set_xlabel("tiempo desde el pulso (ms)   ·   punteado: todas las epocas, lleno: sin las rechazadas")
    np.atleast_1d(axes)[0].legend(frameon=False, ncol=4, fontsize=8)
    if titulo:
        np.atleast_1d(axes)[0].set_title(titulo)
    fig.tight_layout()
    return fig


def fig_criterios(medidas):
    """Pico a pico de VEO (ventana critica) y del EEG por epoca, con los umbrales."""
    fig, axes = plt.subplots(2, 1, figsize=(9, 5), sharex=True)
    pares = [("ptp_veo_uV", config.UMBRAL_VEO_UV,
              f"VEO {config.VENTANA_VEO[0]:g} a {config.VENTANA_VEO[1]:g} s"),
             ("ptp_eeg_uV", config.UMBRAL_RECHAZO_UV, "EEG, epoca completa")]
    for ax, (col, umbral, nombre) in zip(axes, pares):
        if medidas[col].notna().any():
            for c in config.NIVELES:
                m = (medidas["condicion"] == c).to_numpy()
                ax.scatter(np.flatnonzero(m), medidas.loc[m, col], s=14, color=qc.COLORES[c], label=c)
        ax.axhline(umbral, ls="--", color="0.4", lw=0.9)
        ax.text(1.0, umbral, f" {umbral} uV", transform=ax.get_yaxis_transform(), va="center", fontsize=8)
        ax.set_ylabel(f"pico a pico (uV)\n{nombre}")
        ax.set_yscale("log")
    for ax in axes:
        if ax.get_legend_handles_labels()[0]:
            ax.legend(frameon=False, ncol=4)
            break
    axes[-1].set_xlabel("ensayo")
    fig.tight_layout()
    return fig



def rutas(sub, carpeta=None):
    base = Path(config.CARPETA_SALIDA if carpeta is None else carpeta) / sub
    return {
        "epocas": segmentacion.ruta_epocas(sub, carpeta),
        "limpias": base / f"{sub}_limpias-epo.fif",
        "rechazo": base / f"{sub}_rechazo.csv",
        "figuras": base / "limpieza",
    }


def limpiar_sujeto(sub, carpeta=None):
    r = rutas(sub, carpeta)
    if not r["epocas"].exists():
        raise FileNotFoundError(f"No esta {r['epocas']}. Corre antes: python -m procesamiento.qc {sub}")
    ep = mne.read_epochs(r["epocas"], verbose="ERROR")
    ep = marcas_autores.aplicar_marcas(ep, marcas_autores.marcas_de(sub))   # antes de la ref. promedio
    ep = con_referencia_promedio(ep)
    final, tabla, medidas = rechazar(ep)

    final.save(r["limpias"], overwrite=True, verbose="ERROR")
    tabla.to_csv(r["rechazo"], index=False)
    r["figuras"].mkdir(parents=True, exist_ok=True)
    for nombre, fig in {"1_criterios": fig_criterios(medidas),
                        "2_antes_despues": fig_antes_despues(ep, final)}.items():
        fig.savefig(r["figuras"] / f"{sub}_{nombre}.png", dpi=120)
        plt.close(fig)

    print(f"Epocas por nivel (umbral VEO {config.UMBRAL_VEO_UV} uV en {config.VENTANA_VEO} s, "
          f"umbral EEG {config.UMBRAL_RECHAZO_UV} uV):")
    print(tabla.to_string(index=False))
    print(f"\nEpocas limpias: {r['limpias']}\nFiguras: {r['figuras']}")
    return final, tabla


def main(argv=None):
    p = argparse.ArgumentParser(description="Rechazo de epocas con artefactos")
    p.add_argument("sujeto")
    a = p.parse_args(argv)
    matplotlib.use("Agg")
    limpiar_sujeto(a.sujeto)


if __name__ == "__main__":
    main()