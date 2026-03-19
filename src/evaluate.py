import pandas as pd
import joblib
import os
from sklearn.metrics import accuracy_score

def evaluate():
    input_path = os.getenv('VH_INPUTS_DIR', '.')
    X = pd.read_csv(os.path.join(input_path, 'X_preprocessed.csv'))
    y = pd.read_csv(os.path.join(input_path, 'y_preprocessed.csv'))
    
    # Cargar modelo entrenado
    model = joblib.load(os.path.join(input_path, 'model.pkl'))
    
    predictions = model.predict(X)
    acc = accuracy_score(y, predictions)
    
    # Valohai lee este JSON para mostrar gráficas
    print(f'{{"accuracy": {acc}}}') 
    print(f"Evaluación finalizada. Accuracy: {acc}")

if __name__ == "__main__":
    evaluate()