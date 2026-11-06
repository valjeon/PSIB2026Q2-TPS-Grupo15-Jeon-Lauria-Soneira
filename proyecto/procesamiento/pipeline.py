"""
Uso desde proyecto/:
    python -m procesamiento.pipeline sub-004 sub-005
"""
import argparse

from procesamiento import config, epocas


def procesar_sujeto(sub):
    """Pasos 1 y 2 de un sujeto: filtrar cada sesion y extraer las epocas. Devuelve la fila de resumen."""
    ep = epocas.extraer_sujeto(sub)
    ruta = epocas.guardar(ep, sub)
    fila = epocas.resumen(ep, sub)
    print(f"{sub}: {fila['epocas']} epocas en {fila['sesiones']} sesiones ("
          + ", ".join(f"{c} {fila[c]}" for c in config.NIVELES) + f") -> {ruta}"
          + ("" if fila["completo"] else f"   <- se esperaban {config.EPOCAS_ESPERADAS}"))
    return fila


def main(argv=None):
    p = argparse.ArgumentParser(description="Pipeline (por ahora: filtrado y extraccion de epocas)")
    p.add_argument("sujetos", nargs="+", help="ej. sub-004 sub-005")
    filas = [procesar_sujeto(sub) for sub in p.parse_args(argv).sujetos]
    print(f"Resumen: {epocas.guardar_resumen(filas)}")


if __name__ == "__main__":
    main()