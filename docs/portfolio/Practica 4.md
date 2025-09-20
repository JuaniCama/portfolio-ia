# Práctica 4 — Regresión Lineal y Logística

## Contexto

En esta práctica trabajé dos modelos clásicos:

* **Regresión Lineal**: predice un valor continuo (p. ej., precio de casas).
* **Regresión Logística**: predice una clase (0/1) a partir de una probabilidad.

El objetivo fue completar y ejecutar el código, entender qué cambia entre ambos modelos y leer correctamente sus métricas.

---

## 🏠 Parte 1: Regresión Lineal — Predecir Precios de Casas

### 🔧 Paso 1: Setup Inicial

**Entrada:**

```python
# Importar librerías que vamos a usar
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Para los modelos de machine learning
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, classification_report, confusion_matrix,
    precision_score, recall_score, f1_score
)
from sklearn.datasets import load_breast_cancer

print("✅ Setup completo!")
```

**Salida:**

```text
✅ Setup completo!
```

### 🏠 Paso 2: Cargar Dataset de Boston Housing

**Entrada:**

```python
# === CARGAR DATOS DE CASAS EN BOSTON ===

# 1. Cargar el dataset desde una URL
url = "https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv"
boston_data = pd.read_csv(url)

print("🏠 DATASET: Boston Housing")
print(f"   📊 Forma: {boston_data.shape}")
print(f"   📋 Columnas: {list(boston_data.columns)}")

# 2. Explorar los datos básicamente
print("\n🔍 Primeras 5 filas:")
print(boston_data.head())

# 3. Preparar X (variables independientes) e y (variable dependiente)
# La columna 'medv' es el precio de la casa que queremos predecir
```

**Salida:**

```text
🏠 DATASET: Boston Housing
   📊 Forma: (506, 14)
   📋 Columnas: ['crim', 'zn', 'indus', 'chas', 'nox', 'rm', 'age', 'dis', 'rad', 'tax', 'ptratio', 'b', 'lstat', 'medv']

🔍 Primeras 5 filas:
     crim    zn  indus  chas    nox     rm   age     dis  rad  tax  ptratio       b  lstat  medv
0  0.0063  18.0   2.31     0  0.538  6.575  65.2  4.0900    1  296     15.3  396.90   4.98  24.0
1  0.0273   0.0   7.07     0  0.469  6.421  78.9  4.9671    2  242     17.8  396.90   9.14  21.6
2  0.0273   0.0   7.07     0  0.469  7.185  61.1  4.9671    2  242     17.8  392.83   4.03  34.7
3  0.0324   0.0   2.18     0  0.458  6.998  45.8  6.0622    3  222     18.7  394.63   2.94  33.4
4  0.0691   0.0   2.18     0  0.458  7.147  54.2  6.0622    3  222     18.7  396.90   5.33  36.2
```

### 🔬 Paso 3: Entrenar Regresión Lineal

**Entrada:**

```python
# 🔬 Paso 3: Entrenar Regresión Lineal

# X = todas las columnas EXCEPTO la que queremos predecir
X = boston_data.drop('medv', axis=1)      # Todas las columnas EXCEPTO 'medv'
y = boston_data['medv']                   # Solo la columna que queremos predecir

print(f"\n📊 X tiene forma: {X.shape}")
print(f"📊 y tiene forma: {y.shape}")
print(f"🎯 Queremos predecir: Precio de casas en miles de USD")
print(f"📈 Precio mínimo: ${y.min():.1f}k, Precio máximo: ${y.max():.1f}k")

# === ENTRENAR MODELO DE REGRESIÓN LINEAL ===

# 1. Dividir datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"📊 Datos de entrenamiento: {X_train.shape[0]} casas")
print(f"📊 Datos de prueba: {X_test.shape[0]} casas")

# 2. Crear y entrenar el modelo
modelo_regresion = LinearRegression()
modelo_regresion.fit(X_train, y_train)

print("✅ Modelo entrenado!")

# 3. Hacer predicciones
predicciones = modelo_regresion.predict(X_test)

print(f"\n🔮 Predicciones hechas para {len(predicciones)} casas")

# 4. Evaluar qué tan bueno es el modelo con MÚLTIPLES MÉTRICAS
mae  = mean_absolute_error(y_test, predicciones)
mse  = mean_squared_error(y_test, predicciones)
rmse = np.sqrt(mse)
r2   = r2_score(y_test, predicciones)

# Calcular MAPE manualmente
mape = np.mean(np.abs((y_test - predicciones) / y_test)) * 100

print(f"\n📈 MÉTRICAS DE EVALUACIÓN:")
print(f"   📊 MAE (Error Absoluto Medio): ${mae:.2f}k")
print(f"   📊 MSE (Error Cuadrático Medio): {mse:.2f}")
print(f"   📊 RMSE (Raíz del Error Cuadrático): ${rmse:.2f}k")
print(f"   📊 R² (Coeficiente de determinación): {r2:.3f}")
print(f"   📊 MAPE (Error Porcentual Absoluto): {mape:.1f}%")

print(f"\n🔍 INTERPRETACIÓN:")
print(f"   💰 En promedio nos equivocamos por ${mae:.2f}k (MAE)")
print(f"   📈 El modelo explica {r2*100:.1f}% de la variabilidad (R²)")
print(f"   📊 Error porcentual promedio: {mape:.1f}% (MAPE)")

# 5. Comparar algunas predicciones reales vs predichas
print(f"\n🔍 EJEMPLOS (Real vs Predicho):")
for i in range(5):
    real = y_test.iloc[i]
    predicho = predicciones[i]
    print(f"   Casa {i+1}: Real ${real:.1f}k vs Predicho ${predicho:.1f}k")
```

**Salida:**

```text
📊 X tiene forma: (506, 13)
📊 y tiene forma: (506,)
🎯 Queremos predecir: Precio de casas en miles de USD
📈 Precio mínimo: $5.0k, Precio máximo: $50.0k
📊 Datos de entrenamiento: 404 casas
📊 Datos de prueba: 102 casas
✅ Modelo entrenado!

🔮 Predicciones hechas para 102 casas

📈 MÉTRICAS DE EVALUACIÓN:
   📊 MAE (Error Absoluto Medio): $3.20k
   📊 MSE (Error Cuadrático Medio): 24.29
   📊 RMSE (Raíz del Error Cuadrático): $4.93k
   📊 R² (Coeficiente de determinación): 0.668
   📊 MAPE (Error Porcentual Absoluto): 16.2%

🔍 INTERPRETACIÓN:
   💰 En promedio nos equivocamos por $3.20k (MAE)
   📈 El modelo explica 66.8% de la variabilidad (R²)
   📊 Error porcentual promedio: 16.2% (MAPE)

🔍 EJEMPLOS (Real vs Predicho):
   Casa 1: Real $22.6k vs Predicho $22.2k
   Casa 2: Real $50.0k vs Predicho $36.1k
   Casa 3: Real $23.0k vs Predicho $26.6k
   Casa 4: Real $8.5k vs Predicho $13.4k
   Casa 5: Real $18.9k vs Predicho $21.3k
```

---

## 🩺 Parte 2: Regresión Logística — Diagnóstico Médico

### 🩺 Paso 4: Cargar Datos Médicos

**Entrada:**

```python
# === CARGAR DATOS DE DIAGNÓSTICO DE CÁNCER ===

# 1. Cargar el dataset de cáncer de mama (que viene con sklearn)
cancer_data = load_breast_cancer()

# 2. Convertir a DataFrame para verlo mejor
X_cancer = pd.DataFrame(cancer_data.data, columns=cancer_data.feature_names)
y_cancer = cancer_data.target  # 0 = maligno, 1 = benigno

print("🏥 DATASET: Breast Cancer (Diagnóstico)")
print(f"   📊 Pacientes: {X_cancer.shape[0]}")
print(f"   📊 Características: {X_cancer.shape[1]}")
print(f"   🎯 Objetivo: Predecir si tumor es benigno (1) o maligno (0)")

# 3. Ver balance de clases
casos_malignos = (y_cancer == 0).sum()
casos_benignos = (y_cancer == 1).sum()

print(f"\n📊 DISTRIBUCIÓN:")
print(f"   ❌ Casos malignos: {casos_malignos}")
print(f"   ✅ Casos benignos: {casos_benignos}")
```

**Salida:**

```text
🏥 DATASET: Breast Cancer (Diagnóstico)
   📊 Pacientes: 569
   📊 Características: 30
   🎯 Objetivo: Predecir si tumor es benigno (1) o maligno (0)

📊 DISTRIBUCIÓN:
   ❌ Casos malignos: 212
   ✅ Casos benignos: 357
```

### 🧪 Paso 5: Entrenar Regresión Logística

**Entrada:**

```python
# === ENTRENAR MODELO DE CLASIFICACIÓN ===

# 1. Dividir datos en entrenamiento y prueba
X_train_cancer, X_test_cancer, y_train_cancer, y_test_cancer = train_test_split(
    X_cancer, y_cancer, test_size=0.2, random_state=42, stratify=y_cancer
)

print(f"📊 Datos de entrenamiento: {X_train_cancer.shape[0]} pacientes")
print(f"📊 Datos de prueba: {X_test_cancer.shape[0]} pacientes")

# Baseline simple para comparar
baseline = DummyClassifier(strategy="most_frequent", random_state=42)
baseline.fit(X_train_cancer, y_train_cancer)
y_base = baseline.predict(X_test_cancer)
acc_base = accuracy_score(y_test_cancer, y_base)

# 2. Crear y entrenar modelo de regresión logística
modelo_clasificacion = LogisticRegression(max_iter=5000, random_state=42, solver="liblinear")
modelo_clasificacion.fit(X_train_cancer, y_train_cancer)

print("✅ Modelo de clasificación entrenado!")

# 3. Hacer predicciones
predicciones_cancer = modelo_clasificacion.predict(X_test_cancer)

# 4. Evaluar con MÚLTIPLES MÉTRICAS de clasificación
exactitud = accuracy_score(y_test_cancer, predicciones_cancer)
precision = precision_score(y_test_cancer, predicciones_cancer)
recall = recall_score(y_test_cancer, predicciones_cancer)
f1 = f1_score(y_test_cancer, predicciones_cancer)

print(f"\n📈 MÉTRICAS DE CLASIFICACIÓN:")
print(f"   🎯 Baseline (most_frequent) – Accuracy: {acc_base:.3f}")
print(f"   🎯 Logistic Regression – Accuracy: {exactitud:.3f} ({exactitud*100:.1f}%)")
print(f"   🎯 Precision: {precision:.3f} ({precision*100:.1f}%)")
print(f"   🎯 Recall: {recall:.3f} ({recall*100:.1f}%)")
print(f"   🎯 F1-Score: {f1:.3f}")

# Mostrar matriz de confusión de forma simple
matriz_confusion = confusion_matrix(y_test_cancer, predicciones_cancer)
print(f"\n🔢 MATRIZ DE CONFUSIÓN:")
print(f"   📊 {matriz_confusion}")
print(f"   📋 [Verdaderos Negativos, Falsos Positivos]")
print(f"   📋 [Falsos Negativos, Verdaderos Positivos]")

# Reporte detallado
print(f"\n📋 REPORTE DETALLADO:")
print(classification_report(y_test_cancer, predicciones_cancer, target_names=['Maligno', 'Benigno']))

# 5. Ver ejemplos específicos
print(f"\n🔍 EJEMPLOS (Real vs Predicho):")
for i in range(5):
    real = "Benigno" if y_test_cancer[i] == 1 else "Maligno"
    predicho = "Benigno" if predicciones_cancer[i] == 1 else "Maligno"
    print(f"   Paciente {i+1}: Real: {real} vs Predicho: {predicho}")
```

**Salida:**

```text
📊 Datos de entrenamiento: 455 pacientes
📊 Datos de prueba: 114 pacientes
✅ Modelo de clasificación entrenado!

📈 MÉTRICAS DE CLASIFICACIÓN:
   🎯 Baseline (most_frequent) – Accuracy: 0.632
   🎯 Logistic Regression – Accuracy: 0.965 (96.5%)
   🎯 Precision: 0.973 (97.3%)
   🎯 Recall: 0.986 (98.6%)
   🎯 F1-Score: 0.979

🔢 MATRIZ DE CONFUSIÓN:
   📊 [[38  2]
       [ 1 73]]
   📋 [Verdaderos Negativos, Falsos Positivos]
   📋 [Falsos Negativos, Verdaderos Positivos]

📋 REPORTE DETALLADO:
              precision    recall  f1-score   support

     Maligno       0.97      0.95      0.96        40
     Benigno       0.97      0.99      0.98        74

    accuracy                           0.96       114
   macro avg       0.97      0.97      0.97       114
weighted avg       0.97      0.96      0.96       114

🔍 EJEMPLOS (Real vs Predicho):
   Paciente 1: Real: Benigno vs Predicho: Benigno
   Paciente 2: Real: Benigno vs Predicho: Benigno
   Paciente 3: Real: Benigno vs Predicho: Benigno
   Paciente 4: Real: Maligno vs Predicho: Maligno
   Paciente 5: Real: Benigno vs Predicho: Benigno
```

## 📝 Parte 3: Actividad Final — Comparar los Dos Modelos

### 🔍 Paso 7: Comparación Simple

| Aspecto              | Regresión Lineal                       | Regresión Logística                             |
| -------------------- | -------------------------------------- | ----------------------------------------------- |
| Qué predice          | Números continuos | Categorías (0/1) a partir de probabilidad   |
| Ejemplo de uso       | Precio de casa                         | Email spam vs no spam; diagnóstico      |
| Rango de salida      | - Infinito, + Infinito                      | Probabilidad en [0, 1]                     |
| Métricas principales | MAE, MSE, RMSE, R²             | Accuracy, Precision, Recall, F1 |

### 🎯 Paso 8: Reflexión Final

#### Respuestas a preguntas planteadas durante la tarea:

* **¿Diferencia principal entre lineal y logística?** La lineal responde cuánto. La logística responde cuál basándose en una probabilidad.
* **¿Por qué separar train/test?** Para no engañarnos: medir en datos nuevos verifica si el modelo generaliza y no memorizó.
* **¿Qué significa 95% de accuracy?** Que acierta 95 de cada 100; si hay desbalance, puede ser engañoso: miro precision/recall/F1 y la matriz de confusión.
* **Error más grave en medicina:** Usualmente el falso negativo (decir “está bien” cuando no), porque demora tratamiento.
* **¿Qué usar para sueldo?** **Regresión lineal** ya que es continuo.
* **¿Y para spam?** **Regresión logística** ya que es binario y además tengo probabilidad para umbrales y priorización.

## Referencias

* [Documentación LinearRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html)
* [Documentación LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
* [Documentación train_test_split](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html)
* Link al proyecto en Colab: [Practica 4.ipynb](https://colab.research.google.com/drive/1PxB1I5wnvImY_kZTkXiVLRABrM85DCYO#scrollTo=XrE_lbsxGdia)

---
