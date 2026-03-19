import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib
import os
import glob

def train():
    # Buscamos el archivo CSV dentro de la carpeta que crea Valohai
    x_search = glob.glob("/valohai/inputs/X_preprocessed.csv/*.csv")
    y_search = glob.glob("/valohai/inputs/y_preprocessed.csv/*.csv")
    
    if not x_search or not y_search:
        print("Error: No se encontraron los archivos preprocesados.")
        return

    print(f"Cargando X desde: {x_search[0]}")
    print(f"Cargando y desde: {y_search[0]}")
    
    X = pd.read_csv(x_search[0])
    y = pd.read_csv(y_search[0])
    
    # Entrenamiento
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y.values.ravel())
    
    # Guardar el modelo en la carpeta de salidas de Valohai
    output_path = os.getenv('VH_OUTPUTS_DIR', '.')
    joblib.dump(model, os.path.join(output_path, 'model.pkl'))
    print("¡Modelo entrenado y guardado exitosamente como model.pkl!")

if __name__ == "__main__":
    train()
