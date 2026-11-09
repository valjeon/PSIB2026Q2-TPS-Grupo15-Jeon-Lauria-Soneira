"""Pipeline

Uso desde proyecto/:
    python -m procesamiento.pipeline sub-004 sub-005
    python -m procesamiento.pipeline --todos              todos los sujetos con crudo o con epocas ya extraidas
    python -m procesamiento.pipeline sub-004 --rehacer    vuelve a extraer las epocas desde el crudo
    python -m procesamiento.pipeline --grupo              solo la etapa de grupo (con lo ya preprocesado)
    python -m procesamiento.pipeline --todos --sin-grupo  solo los pasos por sujeto

Si las epocas de un sujeto ya estan extraidas, se reusan y solo se rehace la limpieza: no hace falta el crudo
(se usan los sub-XXX_laser-epo.fif). 
Un sujeto que falla no frena a los demas. Los resumenes se actualizan despues de cada sujeto. La etapa de grupo usa todos los sujetos
de preprocesado_resumen.csv, no solo los de la corrida actual.
"""
import argparse
import time
from pathlib import Path

import mne

from procesamiento import (config, ensambles, epocas, estadistica, inclusion, limpieza, localizador, medicion,
                           rasgos_ensayo)

def procesar_sujeto(sub, marcas, rehacer=False):
    if rehacer or not epocas.ruta_epocas(sub).exists():
        ep = epocas.extraer_sujeto(sub)  # se filtra el crudo y se extraen las epocas
        epocas.guardar(ep, sub)
    else: #si el archivo de las epocas ya existe
        ep = mne.read_epochs(epocas.ruta_epocas(sub), verbose="ERROR")
    fila_ep = epocas.resumen(ep, sub) #genera un resumen de las epocas ya extraidas
    pre, fila = limpieza.limpiar_sujeto(ep, sub, marcas) #se hace la limpieza elminando las marcas de los autores, canales malos, etc
    limpieza.guardar(pre, sub)
    #se imprime un resumen 
    print(f"{sub}: {fila['segmentadas']} extraidas -> {fila['quedan']} (autores {fila['por_autores']}, "
          f"groseras {fila['groseras']}) | malos: {fila['interpolados_autores'] or '-'} / "
          f"{fila['interpolados_propios'] or '-'}" + (f"   (aviso: {fila['aviso']})" if fila["aviso"] else ""))
    return fila_ep, fila #Devuelve fila de epocas_resumen, fila de preprocesado_resumen


def _guardar_inclusion(etapa, carpeta):
    t = inclusion.evaluar(etapa, carpeta=carpeta) #evalúa qué sujetos cumplen los criterios de inclusión para continuar con el análisis
    t.to_csv(Path(carpeta) / f"inclusion_{etapa}.csv", index=False) #guarda los archivos en csv
    print(f"\nInclusion {etapa}: {int(t['incluido'].sum())} de {len(t)} sujetos")
    for _, f in t[~t["incluido"]].iterrows():
        print(f"  excluido {f['sujeto']}: {f['motivo']}") #imprime cuántos sujetos fueron incluidos y cuáles quedaron excluidos, junto con el motivo
    return t


def etapa_grupo(carpeta=None):
    #ejecuta el procesamiento grupal,
    carpeta = Path(config.CARPETA_SALIDA if carpeta is None else carpeta) #se establece dónde están los resultados
    print("\n=== Etapa de grupo ===")
    if not _guardar_inclusion("previa", carpeta)["incluido"].any(): #verifica que los sujetos cumplan con los criterios de inclusion
        print("Ningun sujeto pasa la inclusion previa: no se sigue")
        return
    localizador.correr(carpeta=carpeta) #identifica la ubicación temporal de las respuestas LEP
    medicion.correr(carpeta=carpeta) #cuantifica las respuestas identificadas,
    if not _guardar_inclusion("final", carpeta)["incluido"].any(): #verifica que haya sujetos suficientes y válidos para continuar
        print("Ningun sujeto pasa la inclusion final: no se sigue")
        return
    ensambles.correr(carpeta=carpeta) #construye los ensambles de respuestas
    rasgos_ensayo.correr(carpeta=carpeta) #extrae características o variables de cada ensayo
    for variante in medicion.VARIANTES:                     
        print(f"\n--- Estadistica, variante {variante} ---")
        estadistica.correr(variante, carpeta=carpeta) #realiza el análisis estadístico para cada variante de medición


def sujetos_disponibles(raiz=None, carpeta=None):
    raiz = Path(config.RAIZ_DATOS if raiz is None else raiz)
    carpeta = Path(config.CARPETA_SALIDA if carpeta is None else carpeta)
    con_crudo = {p.name for p in raiz.glob("sub-*") if any(p.glob("ses-*/eeg/*_eeg.vhdr"))} #busca carpetas de sujetos que tengan archivos EEG crudos
    con_epocas = {p.parent.name for p in carpeta.glob("sub-*/sub-*_laser-epo.fif")} #busca sujetos que ya tengan épocas extraídas y guardadas
    return sorted(con_crudo | con_epocas) #combina ambos y devuelve una lista ordenada sin duplicados


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
    sujetos = sujetos_disponibles() if a.todos else a.sujetos #busca automáticamente los sujetos disponibles o los indicados
    if not sujetos:
        p.error("indicar sujetos, --todos o --grupo")
    marcas = limpieza.leer_marcas() #procesamiento individual
    errores, t0 = {}, time.time()
    for k, sub in enumerate(sujetos, start=1): #recorre todos los sujetos
        print(f"[{k}/{len(sujetos)}] ", end="", flush=True)
        try:
            f_ep, f_pre = procesar_sujeto(sub, marcas, a.rehacer) #ejecuta el procesamiento
        except Exception as e:                                
            errores[sub] = f"{type(e).__name__}: {e}" #si un sujeto presenta un error, lo registra y continúa con el siguiente
            print(f"{sub}: ERROR {errores[sub]}")
            continue
        epocas.guardar_resumen([f_ep])                     
        epocas.guardar_resumen([f_pre], "preprocesado_resumen.csv") #actualiza los archivos de resultados a medida que avanza el procesamiento
    print(f"\nListo: {len(sujetos) - len(errores)} de {len(sujetos)} sujetos en {(time.time() - t0) / 60:.0f} min. "
          f"Resumenes en {config.CARPETA_SALIDA}/epocas_resumen.csv y preprocesado_resumen.csv")
    if errores:
        print("Con error:\n  " + "\n  ".join(f"{s}: {e}" for s, e in errores.items()))
    if not a.sin_grupo:
        etapa_grupo() # si no se indicó "sin grupo", continúa automáticamente con la etapa grupal


if __name__ == "__main__":
    main()
