import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

def train():
    input_path = os.getenv('VH_INPUTS_DIR', '.')
    X = pd.read_csv(os.path.join(input_path, 'X_preprocessed.csv'))
    y = pd.read_csv(os.path.join(input_path, 'y_preprocessed.csv'))
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y.values.ravel())
    
    # Guardar modelo
    output_path = os.getenv('VH_OUTPUTS_DIR', '.')
    joblib.dump(model, os.path.join(output_path, 'model.pkl'))
    print("Entrenamiento completado.")

if __name__ == "__main__":
    train()