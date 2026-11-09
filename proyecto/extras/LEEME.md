# extras/

Código que todavía no pasó al pipeline nuevo (`procesamiento/`, que se arma de a un paso), guardado para
consulta y para reproducir análisis que se citan en el informe. No se mantiene: si algo de acá vuelve a hacer falta, se pasa a
`procesamiento/`.

- `anterior/`: el paquete `procesamiento` tal como estaba en `main` (commit c278a8a, 2/11/2026), con sus
  tests. Incluye lo que no entró al pipeline simple: análisis de sensibilidad del rechazo, control del VEO
  (`control_veo`, `forma_veo`, `sensibilidad`), control de la gamma, clasificador E1 vs E4 con AUROC
  (`estadistica`), rasgos espectrales de Zhuang et al. (`rasgos_ensayo`), elección de umbrales
  (`umbrales_rechazo`), `lote` (bajar, procesar y borrar el crudo), `qc` y `verificar`.
  Se corre desde `proyecto/`, por ejemplo: `python -m extras.anterior.estadistica`.
  Usa sus propios archivos intermedios en `derived_mine/` (mismos nombres que antes).
- `exploracion/`, `exploracion_inicial/`: scripts sueltos de las primeras semanas.

Tests del paquete anterior (no corren con `pytest` a secas): `python -m pytest extras/anterior/tests -q`
