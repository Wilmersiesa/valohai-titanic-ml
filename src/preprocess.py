import pandas as pd
from sklearn.preprocessing import LabelEncoder
import os

def preprocess():
    # Leer datos (Valohai los pone en /valohai/inputs/)
    input_path = os.getenv('VH_INPUTS_DIR', 'data')
    df = pd.read_csv(os.path.join(input_path, 'train.csv'))
    
    # Limpieza
    features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']
    X = df[features].copy()
    y = df['Survived']
    
    X['Age'] = X['Age'].fillna(X['Age'].median())
    X['Fare'] = X['Fare'].fillna(X['Fare'].median())
    le = LabelEncoder()
    X['Sex'] = le.fit_transform(X['Sex'])
    
    # Guardar datos procesados
    output_path = os.getenv('VH_OUTPUTS_DIR', '.')
    X.to_csv(os.path.join(output_path, 'X_preprocessed.csv'), index=False)
    y.to_csv(os.path.join(output_path, 'y_preprocessed.csv'), index=False)
    print("Preprocesamiento completado.")

if __name__ == "__main__":
    preprocess()