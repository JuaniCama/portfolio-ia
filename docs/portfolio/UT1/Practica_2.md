# Práctica 2 — Feature Engineering simple + Modelo base

## Contexto

En esta práctica se extiende el trabajo sobre Titanic con **feature engineering simple** y un **modelo base**. Se construye un *baseline* con `DummyClassifier` y se entrena una `LogisticRegression`, comparando resultados mediante *accuracy*, *classification\_report* y *confusion\_matrix*.

## Resumen de la investigación (scikit-learn)

* **LogisticRegression**: modelo lineal para clasificación binaria/multiclase. Parámetros útiles: `penalty`, `C`, `solver`, `max_iter`, `class_weight`. Útil cuando la relación entre features y log-odds es aproximadamente lineal.

- **Solvers**: `liblinear` es adecuado para datasets pequeños y L1/L2; `lbfgs` y `saga` escalan mejor y soportan multiclase; `saga` admite L1 de forma eficiente.
* **DummyClassifier**: establece un punto de referencia (*baseline*) con estrategias como `most_frequent`, `stratified`, `uniform`.
* **train\_test\_split**: separa datos; `stratify=y` mantiene la proporción de clases; `random_state` hace reproducible la partición; porcentajes típicos de test: 20–30%.
* **Métricas**: `accuracy` puede ser engañosa con clases desbalanceadas; el `classification_report` resume *precision*, *recall* y *f1* por clase; la `confusion_matrix` muestra los aciertos/errores por tipo.

## Desarrollo

### 1) Setup y organización de carpetas

**Entrada:**

```python
from pathlib import Path
try:
    from google.colab import drive
    drive.mount('/content/drive')
    ROOT = Path('/content/drive/MyDrive/IA-UT1')
except Exception:
    ROOT = Path.cwd() / 'IA-UT1'

DATA_DIR = ROOT / 'data'
RESULTS_DIR = ROOT / 'results'
for d in (DATA_DIR, RESULTS_DIR):
    d.mkdir(parents=True, exist_ok=True)
print('Outputs →', ROOT)
```

**Salida:**

```text
Drive already mounted at /content/drive; to attempt to forcibly...remount, call drive.mount("/content/drive", force_remount=True).
Outputs → /content/drive/MyDrive/IA-UT1
```

### 2) Preprocesamiento y Feature Engineering

**Entrada:**

```python
df = train.copy()

# 🚫 PASO 1: Manejar valores faltantes (imputación)
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])  # Valor más común
df['Fare'] = df['Fare'].fillna(df['Fare'].median())               # Mediana
df['Age'] = df['Age'].fillna(df.groupby(['Sex','Pclass'])['Age'].transform('median'))

# 🆕 PASO 2: Crear nuevas features útiles
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
df['IsAlone'] = (df['FamilySize'] == 1).astype(int)

df['Title'] = df['Name'].str.extract(',\s*([^\.]+)\.')
rare_titles = df['Title'].value_counts()[df['Title'].value_counts() < 10].index
df['Title'] = df['Title'].replace(rare_titles, 'Rare')

# 🔄 PASO 3: Preparar datos para el modelo
features = ['Pclass','Sex','Age','Fare','Embarked','FamilySize','IsAlone','Title','SibSp','Parch']
X = pd.get_dummies(df[features], drop_first=True)
y = df['Survived']

X.shape, y.shape
```

**Salida:**

```text
(891, 14), (891,)
```

### 3) Modelo base, Regresión Logística y evaluación

**Entrada:**

```python
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

dummy = DummyClassifier(strategy='most_frequent', random_state=42)
dummy.fit(X_train, y_train)
baseline_pred = dummy.predict(X_test)

lr = LogisticRegression(max_iter=1000, solver='liblinear', random_state=42)
lr.fit(X_train, y_train)
pred = lr.predict(X_test)

print('Baseline acc:', accuracy_score(y_test, baseline_pred))
print('LogReg acc  :', accuracy_score(y_test, pred))

print('\nClassification report (LogReg):')
print(classification_report(y_test, pred))

print('\nConfusion matrix (LogReg):')
print(confusion_matrix(y_test, pred))
```

**Salida:**

```text
Baseline acc: 0.6145251396648045
LogReg acc  : 0.8156424581005587

Classification report (LogReg):
              precision    recall  f1-score   support

           0       0.82      0.89      0.86       110
           1       0.80      0.70      0.74        69

    accuracy                           0.82       179
   macro avg       0.81      0.79      0.80       179
weighted avg       0.81      0.82      0.81       179


Confusion matrix (LogReg):
[[98 12]
 [21 48]]
```

## Interpretación y respuestas de la consigna

1. **Matriz de confusión** (LogReg): permite ver FP y FN. En esta corrida, hubo 12 FP (predijo que sobrevivían y no) y 21 FN (predijo que no y sí sobrevivían).
2. **Clases atendidas**: la clase 0 (no sobrevivió) muestra mejor recall (0.89) que la clase 1 (0.70).
3. **Comparación con baseline**: `LogReg` (0.816) supera al baseline `most_frequent` (0.615).
4. **Errores más importantes**: si priorizamos identificar sobrevivientes, los FN son más críticos por lo tanto, conviene mejorar recall de la clase 1.
5. **Observaciones generales**: `Sex`, `Pclass`, `Fare` y features derivadas como `FamilySize`, `IsAlone`, `Title` aportan señal. La imputación de `Age` por grupo (`Sex`, `Pclass`) ayuda a estabilizar.

## Referencias

* Link al proyecto en Colab: [Practica 2.ipynb](https://colab.research.google.com/drive/1xqMZ4bgd6l43aCXNUve2H2SSq03NfjMk#scrollTo=aSjn1KqD4Zpo)

---
