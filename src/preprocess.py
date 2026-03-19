import pandas as pd
from sklearn.preprocessing import LabelEncoder
import os
import glob

def preprocess():
    # Buscamos el archivo en la carpeta de inputs que definimos en el yaml
    # Valohai monta los inputs en /valohai/inputs/<nombre-del-input>/archivo.csv
    input_path = "/valohai/inputs/training_data/"
    
    # Buscamos cualquier CSV dentro de esa carpeta de input específica
    csv_files = glob.glob(os.path.join(input_path, "*.csv"))
    
    if not csv_files:
        print("No se encontró el archivo en inputs, intentando en data/train.csv local...")
        csv_files = ["data/train.csv"]

    print(f"Leyendo archivo para entrenamiento: {csv_files[0]}")
    df = pd.read_csv(csv_files[0])
    
    # Verificamos si existe la columna Survived
    if 'Survived' not in df.columns:
        print(f"Error: El archivo {csv_files[0]} no tiene la columna 'Survived'.")
        print(f"Columnas encontradas: {df.columns.tolist()}")
        return

    # Proceso de limpieza
    features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']
    X = df[features].copy()
    y = df['Survived']
    
    X['Age'] = X['Age'].fillna(X['Age'].median())
    X['Fare'] = X['Fare'].fillna(X['Fare'].median())
    
    le = LabelEncoder()
    X['Sex'] = le.fit_transform(X['Sex'])
    
    # Guardar resultados
    output_path = os.getenv('VH_OUTPUTS_DIR', '.')
    X.to_csv(os.path.join(output_path, 'X_preprocessed.csv'), index=False)
    y.to_csv(os.path.join(output_path, 'y_preprocessed.csv'), index=False)
    print("Preprocesamiento completado exitosamente con el set de entrenamiento.")

if __name__ == "__main__":
    preprocess()
