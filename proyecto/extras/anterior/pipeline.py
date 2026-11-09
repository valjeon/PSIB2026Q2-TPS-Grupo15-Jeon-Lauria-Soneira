"""Pipeline completo de un sujeto: de los crudos a las epocas limpias.

Encadena los modulos que ya existen (cada uno se puede seguir corriendo por separado):

    1. segmentacion  crudo -> filtrar (FIR 1-100 Hz + notch 48-52) -> epocas -1..+2 s -> <sub>_laser-epo.fif
    2. qc            figuras de control de las epocas crudas                           -> qc/
    3. artefactos    referencia promedio + rechazo por VEO (en N2/P2) y por EEG        -> <sub>_limpias-epo.fif
    4. qc            figuras de control de las epocas limpias                          -> qc_limpias/
    5. rasgos        N2-P2 por ensayo en Cz y potencia por bandas                      -> <sub>_rasgos.csv

Al final del lote junta los rasgos de todos los sujetos en rasgos_todos.csv (para el clasificador).
Los rasgos salen de las epocas limpias guardadas, no del crudo: despues de procesar un sujeto se
puede borrar su crudo (~1,3 GB) sin perder nada de lo que viene despues.

Y agrega una fila por sujeto a  resumen_pipeline.csv  (en la carpeta de salida): pulsos, epocas que
quedan por nivel, N2-P2 por nivel y cuanto tardo cada etapa. Sirve para estimar el costo del lote.

Uso desde la carpeta que contiene `procesamiento/`:
    python -m extras.anterior.pipeline sub-001
    python -m extras.anterior.pipeline sub-001 sub-002 sub-003
    python -m extras.anterior.pipeline --todos              # todos los sujetos descargados
    python -m extras.anterior.pipeline sub-001 --rehacer    # volver a segmentar aunque exista el .fif
    python -m extras.anterior.pipeline sub-001 --filtro iir # alternativa IIR (sale en <salida>/iir/)
    python -m extras.anterior.pipeline sub-001 --sin-qc     # sin figuras (mas rapido)
    python -m extras.anterior.pipeline sub-001 --sin-rasgos # solo hasta las epocas limpias

Si un sujeto falla, el lote sigue con el siguiente y el error queda en el resumen.
"""
import argparse
import time
import traceback
from pathlib import Path

import matplotlib
import mne
import pandas as pd

from extras.anterior import artefactos, config, datos, filtros, qc, rasgos, segmentacion

TOTAL_ESPERADO = len(config.NIVELES) * config.PULSOS_POR_NIVEL   # 80 pulsos por sujeto


def carpeta_de_salida(filtro="fir", carpeta=None):
    """Carpeta donde se guarda todo. El filtro FIR (el de referencia) usa la carpeta base;
    el IIR va en una subcarpeta para no pisar los archivos del FIR."""
    base = Path(config.CARPETA_SALIDA if carpeta is None else carpeta)
    return base if filtro == "fir" else base / filtro


def procesar_sujeto(sub, filtro="fir", rehacer=False, con_qc=True, carpeta=None, con_rasgos=True):
    """Corre las 5 etapas para un sujeto y devuelve su fila de resumen (dict).

    No captura errores: eso lo hace correr(), para poder seguir con el lote.
    """
    carpeta = carpeta_de_salida(filtro, carpeta)
    t0 = time.time()

    # 1. segmentacion (lo pesado: lee y filtra las 6 sesiones)
    ruta = segmentacion.ruta_epocas(sub, carpeta)
    if ruta.exists() and not rehacer:
        print(f"[1/5] Epocas ya existen, se leen (usar --rehacer para volver a segmentar): {ruta}")
        ep = mne.read_epochs(ruta, verbose="ERROR")
    else:
        print(f"[1/5] Segmentando {sub} (filtro {filtro})...")
        ep = segmentacion.epocas_sujeto(sub, filtrar=lambda raw: filtros.filtrar(raw, metodo=filtro))
        segmentacion.guardar_epocas(ep, sub, carpeta)
    t_seg = time.time() - t0

    # 2. qc de las epocas crudas
    if con_qc:
        print("[2/5] Control de calidad de las epocas crudas...")
        qc.correr_qc(ep, sub, carpeta=carpeta)
    t1 = time.time()

    # 3. rechazo de artefactos
    print("[3/5] Rechazo de artefactos...")
    _, tabla = artefactos.limpiar_sujeto(sub, carpeta)
    t_lim = time.time() - t1

    # 4. qc de las epocas limpias
    n2p2 = {}
    if con_qc:
        print("[4/5] Control de calidad de las epocas limpias...")
        limpias = mne.read_epochs(artefactos.rutas(sub, carpeta)["limpias"], verbose="ERROR")
        resumen = qc.correr_qc(limpias, sub, carpeta=carpeta, subcarpeta="qc_limpias")
        n2p2 = resumen.set_index("condicion")["N2P2_uV"].to_dict()

    # 5. rasgos por ensayo (desde las epocas limpias guardadas)
    n_rasgos, t_ras = 0, 0.0
    if con_rasgos:
        print("[5/5] Rasgos por ensayo...")
        t2 = time.time()
        n_rasgos = len(rasgos.rasgos_sujeto(sub, carpeta))
        t_ras = time.time() - t2

    return armar_fila(sub, filtro, tabla, n2p2, t_seg, t_lim, time.time() - t0, n_rasgos, t_ras)


def armar_fila(sub, filtro, tabla, n2p2, t_seg, t_lim, t_total, n_rasgos=0, t_ras=0.0):
    """Fila de resumen a partir de la tabla de rechazo de artefactos.limpiar_sujeto."""
    t = tabla.set_index("condicion")
    segmentadas = int(t["antes"].sum())
    fila = {
        "sujeto": sub,
        "filtro": filtro,
        "estado": "ok",
        "segmentadas": segmentadas,
        "completo": segmentadas == TOTAL_ESPERADO,   # 80 pulsos, 20 por nivel, todos con ventana completa
        "rechazadas": int(t["rechazadas"].sum()),
        "por_VEO": int(t["por_VEO"].sum()),
        "por_EEG": int(t["por_EEG"].sum()),
        "limpias": int(t["despues"].sum()),
    }
    for c in config.NIVELES:
        fila[f"{c}_limpias"] = int(t["despues"].get(c, 0))
    for c in config.NIVELES:
        fila[f"N2P2_{c}_uV"] = round(float(n2p2[c]), 1) if c in n2p2 else None
    fila["ensayos_con_rasgos"] = n_rasgos
    fila.update(seg_s=round(t_seg, 1), limpieza_s=round(t_lim, 1), rasgos_s=round(t_ras, 1),
                total_s=round(t_total, 1))
    return fila


def guardar_resumen(filas, carpeta):
    """Agrega las filas a resumen_pipeline.csv; una fila nueva reemplaza a la anterior del mismo sujeto."""
    ruta = Path(carpeta) / "resumen_pipeline.csv"
    ruta.parent.mkdir(parents=True, exist_ok=True)
    nuevas = pd.DataFrame(filas)
    if ruta.exists():
        previas = pd.read_csv(ruta)
        previas = previas[~previas["sujeto"].isin(nuevas["sujeto"])]
        nuevas = pd.concat([previas, nuevas], ignore_index=True)
    nuevas.sort_values("sujeto").to_csv(ruta, index=False)
    return ruta


def correr(sujetos, filtro="fir", rehacer=False, con_qc=True, carpeta=None, con_rasgos=True):
    """Procesa varios sujetos, uno tras otro. Un error en uno no frena a los demas.

    El resumen se guarda despues de cada sujeto: si el lote se corta, no se pierde lo hecho.
    Devuelve la lista de filas.
    """
    filas = []
    destino = carpeta_de_salida(filtro, carpeta)
    for i, sub in enumerate(sujetos, start=1):
        print(f"\n=== {sub} ({i}/{len(sujetos)}) ===")
        try:
            fila = procesar_sujeto(sub, filtro=filtro, rehacer=rehacer, con_qc=con_qc, carpeta=carpeta,
                                   con_rasgos=con_rasgos)
        except Exception as e:           # un sujeto roto no tiene que tirar el lote
            traceback.print_exc()
            fila = {"sujeto": sub, "filtro": filtro, "estado": f"error: {type(e).__name__}: {e}"}
        filas.append(fila)
        guardar_resumen([fila], destino)
    return filas


def imprimir_resumen(filas):
    ok = [f for f in filas if f["estado"] == "ok"]
    print("\n" + "=" * 60)
    for f in filas:
        if f["estado"] != "ok":
            print(f"{f['sujeto']}: {f['estado']}")
            continue
        aviso = "" if f["completo"] else f"  <- ATENCION: {f['segmentadas']} segmentadas, se esperaban {TOTAL_ESPERADO}"
        print(f"{f['sujeto']}: {f['segmentadas']} epocas -> {f['limpias']} limpias "
              f"({f['rechazadas']} rechazadas: VEO {f['por_VEO']}, EEG {f['por_EEG']}) en {f['total_s']} s{aviso}")
    if ok:
        print(f"\n{len(ok)}/{len(filas)} sujetos ok; tiempo medio {sum(f['total_s'] for f in ok) / len(ok):.0f} s por sujeto")


def main(argv=None):
    p = argparse.ArgumentParser(description="Pipeline completo: crudo -> epocas limpias",
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("sujetos", nargs="*", help="ej. sub-001 sub-002")
    p.add_argument("--todos", action="store_true", help="todos los sujetos descargados")
    p.add_argument("--filtro", choices=["fir", "iir"], default="fir")
    p.add_argument("--rehacer", action="store_true", help="volver a segmentar aunque exista el .fif")
    p.add_argument("--sin-qc", action="store_true", help="no generar figuras de control")
    p.add_argument("--sin-rasgos", action="store_true", help="no calcular los rasgos por ensayo")
    a = p.parse_args(argv)

    sujetos = datos.listar_sujetos() if a.todos else a.sujetos
    if not sujetos:
        p.error("indicar al menos un sujeto, o --todos (y que haya sujetos descargados)")

    matplotlib.use("Agg")                # solo guardar figuras, sin abrir ventanas
    print("Datos en:", config.RAIZ_DATOS, "| salida:", carpeta_de_salida(a.filtro))
    filas = correr(sujetos, filtro=a.filtro, rehacer=a.rehacer, con_qc=not a.sin_qc,
                   con_rasgos=not a.sin_rasgos)
    imprimir_resumen(filas)
    if not a.sin_rasgos and any(f["estado"] == "ok" for f in filas):
        todos = rasgos.reunir_rasgos(carpeta_de_salida(a.filtro))
        if todos is not None:
            print(f"rasgos_todos.csv: {len(todos)} ensayos de {todos['sujeto'].nunique()} sujeto(s)")
    return 0 if all(f["estado"] == "ok" for f in filas) else 1


if __name__ == "__main__":
    raise SystemExit(main())
