"""Entrena el Pipeline del Taller 1 (Titanic) y lo guarda en pipeline.joblib."""
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

URL = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv"
COLS_NUM = ["age", "fare", "sibsp", "parch"]
COLS_CAT = ["pclass", "sex", "embarked"]

# 1. Datos
df = pd.read_csv(URL)
X = df[COLS_NUM + COLS_CAT]
y = df["survived"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# 2. Pipeline: preprocesamiento + modelo con sus hiperparámetros
preprocesador = ColumnTransformer([
    ("num", Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler()),
    ]), COLS_NUM),
    ("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ]), COLS_CAT),
])
pipeline = Pipeline([
    ("preprocesador", preprocesador),
    ("modelo", LogisticRegression(C=0.1, max_iter=1000)),
])

# 3. Entrenar y evaluar
pipeline.fit(X_train, y_train)
print(pipeline)
print(f"\nAccuracy en test: {accuracy_score(y_test, pipeline.predict(X_test)):.3f}")

# 4. Guardar el pipeline completo
joblib.dump(pipeline, "pipeline.joblib")
print("Pipeline guardado en pipeline.joblib")

# 5. Comprobar que el archivo guardado funciona con datos crudos
cargado = joblib.load("pipeline.joblib")
ejemplo = pd.DataFrame([{"age": 22, "fare": 7.25, "sibsp": 1, "parch": 0,
                         "pclass": 3, "sex": "male", "embarked": "S"}])
print("Prueba con el pipeline cargado -> probabilidad de supervivencia:",
      round(cargado.predict_proba(ejemplo)[0, 1], 3))