import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Cargando los datos
df = pd.read_csv("datos_limpios.csv")

# Eliminando la columna categoria_edad
df = df.drop(columns=["edad_categoria"])

# Obteniendo el vector objetivo
y = df["is_dead"].values

# Dividiendo el dataset en conjunto de entrenamiento y test
X_train, X_test, y_train, y_test = train_test_split(
    df.drop(columns=["is_dead"]).values, y, stratify=y, test_size=0.25
)

# Ajustando el random forest
clf = RandomForestClassifier(n_estimators=100, max_depth=4)
clf.fit(X_train, y_train)

# Calculando la matriz de confusión
confusion_matrix = clf.predict_proba(X_test)[:, 1]

# Calculando el F1-Score
f1_score = sklearn.metrics.f1_score(y_test, clf.predict(X_test))

# Imprimiendo el accuracy y el F1-Score
print("Accuracy:", clf.score(X_test, y_test))
print("F1-Score:", f1_score)

# Grafica la matriz de confusión
plt.matshow(confusion_matrix, cmap="Blues")
plt.title("Matriz de confusión")
plt.xlabel("Predicción")
plt.ylabel("Real")
plt.show()
