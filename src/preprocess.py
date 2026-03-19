import pandas as pd
from sklearn.preprocessing import LabelEncoder
import os
import glob

def preprocess():
    # Valohai guarda los inputs en esta ruta
    input_base = os.getenv('VH_INPUTS_DIR', '/valohai/inputs/train.csv')
    
    # Buscamos cualquier archivo .csv dentro de esa carpeta de input
    csv_files = glob.glob(os.path.join(input_base, "*.csv"))
    
    if not csv_files:
        # Si no lo encuentra ahí, intenta en la carpeta local (por si acaso)
        csv_files = glob.glob("data/*.csv")
        
    if not csv_files:
        print("Error: No se encontró ningún archivo CSV en los inputs.")
        return

    print(f"Leyendo archivo: {csv_files[0]}")
    df = pd.read_csv(csv_files[0])
    
    # El resto del proceso sigue igual
    features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']
    X = df[features].copy()
    y = df['Survived']
    
    X['Age'] = X['Age'].fillna(X['Age'].median())
    X['Fare'] = X['Fare'].fillna(X['Fare'].median())
    le = LabelEncoder()
    X['Sex'] = le.fit_transform(X['Sex'])
    
    # Guardar en la carpeta de salidas de Valohai
    output_path = os.getenv('VH_OUTPUTS_DIR', '.')
    X.to_csv(os.path.join(output_path, 'X_preprocessed.csv'), index=False)
    y.to_csv(os.path.join(output_path, 'y_preprocessed.csv'), index=False)
    print("Preprocesamiento completado exitosamente.")

if __name__ == "__main__":
    preprocess()
