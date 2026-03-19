import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib
import os

def run_model():
    # 1. Cargar datos
    if not os.path.exists('train.csv'):
        print("Error: No se encuentra train.csv")
        return

    df = pd.read_csv('train.csv')
    
    # 2. Preprocesamiento simple
    features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']
    X = df[features].copy()
    y = df['Survived']
    
    # Llenar nulos
    X['Age'] = X['Age'].fillna(X['Age'].median())
    X['Fare'] = X['Fare'].fillna(X['Fare'].median())
    
    # Convertir Sexo a números
    le = LabelEncoder()
    X['Sex'] = le.fit_transform(X['Sex'])
    
    # 3. Entrenar
    print("Entrenando el modelo RandomForest...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    # 4. Guardar resultados
    joblib.dump(model, 'titanic_model.pkl')
    print("¡Modelo guardado exitosamente como titanic_model.pkl!")

if __name__ == "__main__":
    run_model()