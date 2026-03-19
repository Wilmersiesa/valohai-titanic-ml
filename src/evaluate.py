import pandas as pd
import joblib
import os
import glob
from sklearn.metrics import accuracy_score

def evaluate():
    # 1. Buscamos los archivos usando glob para evitar errores de carpeta
    x_path = glob.glob("/valohai/inputs/X_preprocessed.csv/*.csv")[0]
    y_path = glob.glob("/valohai/inputs/y_preprocessed.csv/*.csv")[0]
    model_path = glob.glob("/valohai/inputs/model.pkl/*.pkl")[0]
    
    print(f"Evaluando con: {x_path}")
    print(f"Cargando modelo desde: {model_path}")
    
    # 2. Cargar datos y modelo
    X = pd.read_csv(x_path)
    y = pd.read_csv(y_path)
    model = joblib.load(model_path)
    
    # 3. Realizar predicciones
    predictions = model.predict(X)
    acc = accuracy_score(y, predictions)
    
    # 4. Imprimir métricas en formato JSON (Valohai las captura automáticamente)
    print(f'{{"accuracy": {acc}}}')
    print(f"---")
    print(f"Evaluación finalizada.")
    print(f"Precisión del modelo (Accuracy): {acc * 100:.2f}%")

if __name__ == "__main__":
    evaluate()
