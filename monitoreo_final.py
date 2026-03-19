import os
import sys

# Intentamos forzar a Python a que mire en todas las carpetas posibles de librerías
import site
for path in site.getsitepackages():
    if path not in sys.path:
        sys.path.append(path)

try:
    import pandas as pd
    from evidently.report import Report
    from evidently.metric_preset import DataDriftPreset, TargetDriftPreset
    print("Librerías cargadas correctamente.")
except ImportError as e:
    print(f"Error al importar: {e}")
    print("Intentando solución alternativa...")
    # Si falla, te daré un plan C aquí mismo
    sys.exit(1)

def generar_reporte():
    url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
    print("Cargando datos...")
    df = pd.read_csv(url)

    print("Generando reporte de monitoreo...")
    reporte = Report(metrics=[DataDriftPreset(), TargetDriftPreset()])
    reporte.run(reference_data=df, current_data=df)

    nombre_archivo = "reporte_evidently_titanic.html"
    reporte.save_html(nombre_archivo)
    print(f"¡EXITO! Reporte creado en: {os.path.abspath(nombre_archivo)}")

if __name__ == "__main__":
    generar_reporte()