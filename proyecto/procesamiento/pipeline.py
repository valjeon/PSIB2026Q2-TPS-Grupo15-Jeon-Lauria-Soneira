"""Pipeline: de los registros crudos al resultado. Se arma de a un paso (rama pipeline-simple).

    Antes, aparte: analisis.py (PSD y N2-P2 por canal: evidencia para filtros y electrodos)
    Por sujeto:
      1-2. filtrado.py + epocas.py   leer, filtrar la senal continua y cortar de -1 a 2 s   (lento)
      3.   limpieza.py               marcas de los autores, canales malos, referencia, epocas groseras
    Con todos los sujetos (etapa de grupo, modulos de csoneira):
      4.   inclusion.py (previa)     quien entra al localizador
      5.   localizador.py            ventanas de N2 y P2 con el promedio colapsado (los 4 niveles juntos)
      6.   medicion.py               N2/P2 por ensayo y rechazo final por ROI y VEO (variante principal y de
                                     sensibilidad)
      7.   inclusion.py (final)      quien entra al analisis
      8.   ensambles.py              promedio ponderado por sujeto y nivel, SNR par/impar, gran promedio
      9.   rasgos_ensayo.py          latencia de N2 y gamma por ensayo
      10.  estadistica.py            Spearman y AUROC E1 vs E4, para cada variante

Uso desde proyecto/:
    python -m procesamiento.pipeline sub-004 sub-005
    python -m procesamiento.pipeline --todos              todos los sujetos con crudo o con epocas ya extraidas
    python -m procesamiento.pipeline sub-004 --rehacer    vuelve a extraer las epocas desde el crudo
    python -m procesamiento.pipeline --grupo              solo la etapa de grupo (con lo ya preprocesado)
    python -m procesamiento.pipeline --todos --sin-grupo  solo los pasos por sujeto

Si las epocas de un sujeto ya estan extraidas, se reusan y solo se rehace la limpieza: no hace falta el crudo
(sirven tambien las sub-XXX_laser-epo.fif que dejo lote.py, que segmentaba igual). Un sujeto que falla no
frena a los demas. Los resumenes se actualizan despues de cada sujeto. La etapa de grupo usa todos los sujetos
de preprocesado_resumen.csv, no solo los de esta corrida.
"""
import argparse
import time
from pathlib import Path

import mne

from procesamiento import (config, ensambles, epocas, estadistica, inclusion, limpieza, localizador, medicion,
                           rasgos_ensayo)


def procesar_sujeto(sub, marcas, rehacer=False):
    """Pasos 1 a 3 de un sujeto. Devuelve (fila de epocas_resumen, fila de preprocesado_resumen)."""
    if rehacer or not epocas.ruta_epocas(sub).exists():
        ep = epocas.extraer_sujeto(sub)                       # lento: crudo -> filtrado -> epocas
        epocas.guardar(ep, sub)
    else:
        ep = mne.read_epochs(epocas.ruta_epocas(sub), verbose="ERROR")
    fila_ep = epocas.resumen(ep, sub)
    pre, fila = limpieza.limpiar_sujeto(ep, sub, marcas)
    limpieza.guardar(pre, sub)
    print(f"{sub}: {fila['segmentadas']} extraidas -> {fila['quedan']} (autores {fila['por_autores']}, "
          f"groseras {fila['groseras']}) | malos: {fila['interpolados_autores'] or '-'} / "
          f"{fila['interpolados_propios'] or '-'}" + (f"   (aviso: {fila['aviso']})" if fila["aviso"] else ""))
    return fila_ep, fila


def _guardar_inclusion(etapa, carpeta):
    t = inclusion.evaluar(etapa, carpeta=carpeta)
    t.to_csv(Path(carpeta) / f"inclusion_{etapa}.csv", index=False)
    print(f"\nInclusion {etapa}: {int(t['incluido'].sum())} de {len(t)} sujetos")
    for _, f in t[~t["incluido"]].iterrows():
        print(f"  excluido {f['sujeto']}: {f['motivo']}")
    return t


def etapa_grupo(carpeta=None):
    """Pasos 4 a 10, con los modulos de csoneira."""
    carpeta = Path(config.CARPETA_SALIDA if carpeta is None else carpeta)
    print("\n=== Etapa de grupo ===")
    if not _guardar_inclusion("previa", carpeta)["incluido"].any():
        print("Ningun sujeto pasa la inclusion previa: no se sigue")
        return
    localizador.correr(carpeta=carpeta)
    medicion.correr(carpeta=carpeta)
    if not _guardar_inclusion("final", carpeta)["incluido"].any():
        print("Ningun sujeto pasa la inclusion final: no se sigue")
        return
    ensambles.correr(carpeta=carpeta)
    rasgos_ensayo.correr(carpeta=carpeta)
    for variante in medicion.VARIANTES:                       # principal y sensibilidad
        print(f"\n--- Estadistica, variante {variante} ---")
        estadistica.correr(variante, carpeta=carpeta)


def sujetos_disponibles(raiz=None, carpeta=None):
    """Sujetos con crudo en data/ds005293 o con epocas ya extraidas en derived_mine (por ejemplo, las que dejo
    lote.py despues de borrar el crudo). Ordenados."""
    raiz = Path(config.RAIZ_DATOS if raiz is None else raiz)
    carpeta = Path(config.CARPETA_SALIDA if carpeta is None else carpeta)
    con_crudo = {p.name for p in raiz.glob("sub-*") if any(p.glob("ses-*/eeg/*_eeg.vhdr"))}
    con_epocas = {p.parent.name for p in carpeta.glob("sub-*/sub-*_laser-epo.fif")}
    return sorted(con_crudo | con_epocas)


def main(argv=None):
    import matplotlib
    matplotlib.use("Agg")
    p = argparse.ArgumentParser(description="Pipeline: crudo -> epocas -> limpieza -> LEP y estadistica")
    p.add_argument("sujetos", nargs="*", help="ej. sub-004 sub-005")
    p.add_argument("--todos", action="store_true", help="todos los sujetos con crudo o con epocas ya extraidas")
    p.add_argument("--rehacer", action="store_true", help="volver a extraer las epocas desde el crudo")
    p.add_argument("--grupo", action="store_true", help="solo la etapa de grupo")
    p.add_argument("--sin-grupo", action="store_true", help="solo los pasos por sujeto")
    a = p.parse_args(argv)
    if a.grupo:
        etapa_grupo()
        return
    sujetos = sujetos_disponibles() if a.todos else a.sujetos
    if not sujetos:
        p.error("indicar sujetos, --todos o --grupo")
    marcas = limpieza.leer_marcas()
    errores, t0 = {}, time.time()
    for k, sub in enumerate(sujetos, start=1):
        print(f"[{k}/{len(sujetos)}] ", end="", flush=True)
        try:
            f_ep, f_pre = procesar_sujeto(sub, marcas, a.rehacer)
        except Exception as e:                                # un sujeto roto no frena a los demas
            errores[sub] = f"{type(e).__name__}: {e}"
            print(f"{sub}: ERROR {errores[sub]}")
            continue
        epocas.guardar_resumen([f_ep])                        # despues de cada sujeto
        epocas.guardar_resumen([f_pre], "preprocesado_resumen.csv")
    print(f"\nListo: {len(sujetos) - len(errores)} de {len(sujetos)} sujetos en {(time.time() - t0) / 60:.0f} min. "
          f"Resumenes en {config.CARPETA_SALIDA}/epocas_resumen.csv y preprocesado_resumen.csv")
    if errores:
        print("Con error:\n  " + "\n  ".join(f"{s}: {e}" for s, e in errores.items()))
    if not a.sin_grupo:
        etapa_grupo()


if __name__ == "__main__":
    main()