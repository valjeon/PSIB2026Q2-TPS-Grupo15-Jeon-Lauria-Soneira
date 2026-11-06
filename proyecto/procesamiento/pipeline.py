"""Pipeline: de los registros crudos a las epocas limpias. Se arma de a un paso (rama pipeline-simple).

    Antes, aparte: analisis.py (PSD y N2-P2 por canal: evidencia para filtros y electrodos)
    Por sujeto:
      1-2. filtrado.py + epocas.py   leer, filtrar la senal continua y cortar de -1 a 2 s   (lento)
      3.   limpieza.py               marcas de los autores, canales malos, referencia, rechazo, inclusion (rapido)

Uso desde proyecto/:
    python -m procesamiento.pipeline sub-004 sub-005
    python -m procesamiento.pipeline --todos              todos los sujetos descargados en data/ds005293
    python -m procesamiento.pipeline sub-004 --rehacer    vuelve a extraer las epocas desde el crudo

Si las epocas de un sujeto ya estan extraidas, se reusan y solo se rehace la limpieza. Un sujeto que falla no
frena a los demas. Los resumenes se actualizan despues de cada sujeto, asi una corrida cortada no pierde nada.
"""
import argparse
import time
from pathlib import Path

import mne

from procesamiento import config, epocas, limpieza


def procesar_sujeto(sub, marcas, rehacer=False):
    """Pasos 1 a 3 de un sujeto. Devuelve (fila de epocas, fila de limpieza)."""
    if rehacer or not epocas.ruta_epocas(sub).exists():
        ep = epocas.extraer_sujeto(sub)                       # lento: crudo -> filtrado -> epocas
        epocas.guardar(ep, sub)
    else:
        ep = mne.read_epochs(epocas.ruta_epocas(sub), verbose="ERROR")
    fila_ep = epocas.resumen(ep, sub)
    limpias, antes, medidas, fila = limpieza.limpiar_sujeto(ep, sub, marcas)
    limpieza.guardar(limpias, antes, medidas, sub)
    print(f"{sub}: {fila['extraidas']} extraidas -> {fila['limpias']} limpias (autores {fila['por_autores']}, "
          f"groseras {fila['por_grosera']}, ROI {fila['por_ROI']}, VEO {fila['por_VEO']}) | "
          + ", ".join(f"{c} {fila[c]}" for c in config.NIVELES)
          + ("" if fila["incluido"] else f"   <- EXCLUIDO: {fila['motivo_exclusion']}")
          + (f"   (aviso: {fila['aviso']})" if fila["aviso"] else ""))
    return fila_ep, fila


def sujetos_descargados(raiz=None):
    """Sujetos con al menos una sesion en data/ds005293, ordenados."""
    raiz = Path(config.RAIZ_DATOS if raiz is None else raiz)
    return sorted(p.name for p in raiz.glob("sub-*") if any(p.glob("ses-*/eeg/*_eeg.vhdr")))


def main(argv=None):
    import matplotlib
    matplotlib.use("Agg")
    p = argparse.ArgumentParser(description="Pipeline: crudo -> epocas -> epocas limpias")
    p.add_argument("sujetos", nargs="*", help="ej. sub-004 sub-005")
    p.add_argument("--todos", action="store_true", help="todos los sujetos descargados")
    p.add_argument("--rehacer", action="store_true", help="volver a extraer las epocas desde el crudo")
    a = p.parse_args(argv)
    sujetos = sujetos_descargados() if a.todos else a.sujetos
    if not sujetos:
        p.error("indicar sujetos o --todos")
    marcas = limpieza.leer_marcas()
    errores, t0 = {}, time.time()
    for k, sub in enumerate(sujetos, start=1):
        print(f"[{k}/{len(sujetos)}] ", end="", flush=True)
        try:
            f_ep, f_limp = procesar_sujeto(sub, marcas, a.rehacer)
        except Exception as e:                                # un sujeto roto no frena a los demas
            errores[sub] = f"{type(e).__name__}: {e}"
            print(f"{sub}: ERROR {errores[sub]}")
            continue
        epocas.guardar_resumen([f_ep])                        # despues de cada sujeto
        epocas.guardar_resumen([f_limp], "limpieza_resumen.csv")
    print(f"\nListo: {len(sujetos) - len(errores)} de {len(sujetos)} sujetos en {(time.time() - t0) / 60:.0f} min. "
          f"Resumenes en {config.CARPETA_SALIDA}/epocas_resumen.csv y limpieza_resumen.csv")
    if errores:
        print("Con error:\n  " + "\n  ".join(f"{s}: {e}" for s, e in errores.items()))


if __name__ == "__main__":
    main()