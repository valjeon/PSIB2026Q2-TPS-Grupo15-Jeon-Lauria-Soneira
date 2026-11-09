"""Verifica un o mas sujetos: cuenta los pulsos laser por sesion y en total.

Uso:
    python -m extras.anterior.verificar sub-001 sub-002
Debe dar 20 pulsos por nivel (E1..E4) sumando las 6 sesiones de cada sujeto.
"""
import sys
import warnings

from extras.anterior import config, datos, eventos


def verificar(sub, raiz=None):
    raws, nombres = datos.cargar_sujeto(sub, raiz)
    total = dict.fromkeys(config.NIVELES, 0)
    for nombre, raw in zip(nombres, raws):
        c = eventos.contar_por_nivel(eventos.pulsos_laser(raw))
        print(f"  {nombre}: {c}")
        for k, v in c.items():
            total[k] += v
    with warnings.catch_warnings(record=True) as avisos:
        warnings.simplefilter("always")
        ok = eventos.validar_pulsos(total, sujeto=sub)
    print(f"  TOTAL {sub}: {total} -> {'OK' if ok else 'NO COINCIDE: ' + str(avisos[0].message)}")
    return ok


def main(argv=None):
    subs = sys.argv[1:] if argv is None else argv
    if not subs:
        subs = datos.listar_sujetos()
        print("Sin argumentos: verificando los sujetos descargados:", subs)
    print("Datos en:", config.RAIZ_DATOS)
    resultados = []
    for s in subs:
        print(s)
        resultados.append(verificar(s))
    return 0 if all(resultados) else 1


if __name__ == "__main__":
    sys.exit(main())
