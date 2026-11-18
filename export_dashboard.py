"""Regenera la carpeta export_dashboard/ para el dashboard de Antamina.

Corrida semanal (desde la raíz del repo, tras actualizar los archivos de
src/data/: dataIncidentes.xlsx, dataFormulario.xlsx, reportes):

    python export_dashboard.py --actualizar-fuentes

Opciones: --con-red (re-descarga Defensoría), --sin-reentrenar,
--recalibrar (recalcula umbrales: cambia los textos del dashboard).
Validar sin regenerar:  python -m src.export.validar_dashboard
"""
from src.export.exportar_dashboard import main

if __name__ == "__main__":
    main()
