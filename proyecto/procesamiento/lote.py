"""Lote: bajar -> segmentar -> borrar el crudo, sujeto por sujeto.

Nunca acumula crudos: como mucho hay UN sujeto descargado (~1.6 GB) a la vez.
Lo que queda por sujeto en derived_mine/sub-XXX/ son las epocas (~26 MB) y los CSV.

- Se puede cortar y volver a lanzar: los sujetos ya terminados (tienen rasgos.csv) se saltean.
- Si un sujeto falla, su crudo NO se borra (para poder revisarlo) y el lote sigue con el siguiente.
- Deja un registro en derived_mine/lote_registro.csv (estado, minutos, error de cada sujeto).

Uso desde proyecto/:
    python -m procesamiento.lote sub-XXX          # sujetos sueltos
    python -m procesamiento.lote --desde 1 --hasta 32    # rango (sub-001 .. sub-032)
    python -m procesamiento.lote --desde 1 --hasta 32 --simular   # muestra que haria, sin hacer nada

Para que la Mac no se duerma en un lote largo:
    caffeinate -i python -m procesamiento.lote --desde 1 --hasta 32
"""
import argparse
import csv
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from procesamiento import config, segmentacion

URL_BASE = "https://s3.amazonaws.com/openneuro.org/ds005293"
ARCHIVOS = ["eeg.vhdr", "eeg.vmrk", "eeg.eeg", "eeg.json", "channels.tsv", "electrodes.tsv", "events.tsv"]
SESIONES = range(1, 7)
ESPACIO_MINIMO_GB = 5

def rutas(sub):
    salida = Path(config.CARPETA_SALIDA) / sub
    return {
        "crudo": Path(config.RAIZ_DATOS) / sub,
    }


def ya_hecho(sub):
    return segmentacion.ruta_epocas(sub).exists()


def espacio_libre_gb(carpeta):
    carpeta.mkdir(parents=True, exist_ok=True)
    return shutil.disk_usage(carpeta).free / 1e9


def descargar(sub, simular=False):
    """Baja las sesiones del sujeto con curl. Devuelve cuantos .vhdr hay al final."""
    faltan = 0
    for s in SESIONES:
        carpeta = rutas(sub)["crudo"] / f"ses-{s}" / "eeg"
        for f in ARCHIVOS:
            nombre = f"{sub}_ses-{s}_task-{config.TAREA}_{f}"
            destino = carpeta / nombre
            if destino.exists() and destino.stat().st_size > 0:
                continue                                  # ya estaba (lote cortado a mitad)
            url = f"{URL_BASE}/{sub}/ses-{s}/eeg/{nombre}"
            if simular:
                faltan += 1
                continue
            carpeta.mkdir(parents=True, exist_ok=True)
            r = subprocess.run(["curl", "-fsSL", "--retry", "3", "-o", str(destino), url])
            if r.returncode != 0:
                destino.unlink(missing_ok=True)           # no dejar archivos a medias
                if f == "eeg.vhdr" and s == 1:
                    raise RuntimeError(f"no se pudo bajar {nombre} (¿existe el sujeto?)")
    if simular:
        print(f"    bajaria {faltan} archivos ({len(SESIONES)} sesiones)")
    return len(list(rutas(sub)["crudo"].glob("ses-*/eeg/*_eeg.vhdr")))


def procesar(sub):
    """Segmenta el sujeto (filtra, epoca y guarda -epo.fif + metadata). El exito se verifica por el archivo."""
    subprocess.run([sys.executable, "-m", "procesamiento.segmentacion", sub])
    if not ya_hecho(sub):
        raise RuntimeError("la segmentacion no genero el -epo.fif (ver el error mas arriba)")


def anotar(fila):
    registro = Path(config.CARPETA_SALIDA) / "lote_registro.csv"
    nuevo = not registro.exists()
    registro.parent.mkdir(parents=True, exist_ok=True)
    with open(registro, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["fecha", "sujeto", "estado", "sesiones", "minutos", "detalle"])
        if nuevo:
            w.writeheader()
        w.writerow(fila)


def correr_sujeto(sub, borrar=True, simular=False):
    t0 = time.time()
    fila = {"fecha": datetime.now().isoformat(timespec="seconds"), "sujeto": sub,
            "estado": "", "sesiones": "", "minutos": "", "detalle": ""}
    if ya_hecho(sub):
        print(f"{sub}: ya procesado, se saltea")
        return "salteado"
    libre = espacio_libre_gb(Path(config.RAIZ_DATOS))
    if libre < ESPACIO_MINIMO_GB:
        raise SystemExit(f"Quedan {libre:.1f} GB libres (< {ESPACIO_MINIMO_GB}). Corto el lote.")
    try:
        print(f"\n=== {sub} ===\n  descargando...")
        n = descargar(sub, simular)
        fila["sesiones"] = n
        if simular:
            print(f"  procesaria y borraria {rutas(sub)['crudo']}")
            return "simulado"
        if n < 6:
            print(f"  aviso: {sub} tiene {n} sesiones (se esperaban 6)")
        print("  procesando...")
        procesar(sub)
        if borrar:
            shutil.rmtree(rutas(sub)["crudo"], ignore_errors=True)
            print(f"  crudo borrado: {rutas(sub)['crudo']}")
        fila["estado"] = "ok"
    except Exception as e:                                # noqa: BLE001 (seguir con el resto)
        fila["estado"] = "error"
        fila["detalle"] = f"{type(e).__name__}: {e}"
        print(f"  ERROR en {sub}: {fila['detalle']}  (el crudo se conserva para revisar)")
    fila["minutos"] = round((time.time() - t0) / 60, 1)
    anotar(fila)
    return fila["estado"]


def main(argv=None):
    p = argparse.ArgumentParser(description="Bajar, procesar y borrar crudos, sujeto por sujeto")
    p.add_argument("sujetos", nargs="*", help="ej. sub-002 sub-003")
    p.add_argument("--desde", type=int, help="numero del primer sujeto (ej. 1 = sub-001)")
    p.add_argument("--hasta", type=int, help="numero del ultimo sujeto (incluido)")
    p.add_argument("--conservar-crudos", action="store_true", help="no borrar los crudos al terminar")
    p.add_argument("--simular", action="store_true", help="mostrar lo que haria, sin bajar ni procesar")
    a = p.parse_args(argv)

    sujetos = list(a.sujetos)
    if a.desde is not None:
        sujetos += [f"sub-{i:03d}" for i in range(a.desde, (a.hasta or a.desde) + 1)]
    if not sujetos:
        p.error("indica sujetos (sub-002 ...) o un rango (--desde 1 --hasta 32)")

    print(f"Lote de {len(sujetos)} sujetos. Datos: {config.RAIZ_DATOS} | salida: {config.CARPETA_SALIDA}")
    estados = [correr_sujeto(s, borrar=not a.conservar_crudos, simular=a.simular) for s in sujetos]
    print("\nResumen: " + ", ".join(f"{e}: {estados.count(e)}" for e in sorted(set(estados))))
    malos = [s for s, e in zip(sujetos, estados) if e == "error"]
    if malos:
        print(f"Con error (crudo conservado): {' '.join(malos)}")


if __name__ == "__main__":
    main()