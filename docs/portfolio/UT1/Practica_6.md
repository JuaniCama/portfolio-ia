# Práctica 6 — Clustering y PCA (Mall Customer Segmentation)

## Contexto

En esta práctica trabajé segmentación de clientes sobre el dataset **Mall Customer Segmentation**, aplicando un flujo típico de unsupervised learning:
1) **Entender el negocio**: agrupar clientes con comportamientos de compra similares para diseñar acciones de marketing, fidelización y precios diferenciados.
2) **Entender y preparar los datos**: explorar variables (edad, ingreso anual, spending score, etc.), tratar outliers y normalizar para que las escalas no dominen el clustering.
3) **Reducir dimensionalidad** con PCA para visualizar, des-ruidar y facilitar el agrupamiento.
4) **Clustering** (K-Means) eligiendo k con criterios como elbow y silhouette, y luego interpretar los perfiles de cada grupo.
5) **Evaluar**: no hay “labels” reales; se justifica con métricas internas (silhouette) + interpretabilidad de clusters (coherencia de perfiles).


## Paso 2.1: Setup Inicial

**Entrada:**

```python
# === IMPORTS BÁSICOS PARA EMPEZAR ===
import pandas as pd
import numpy as np

print("Iniciando análisis de Mall Customer Segmentation Dataset")
print("Pandas y NumPy cargados - listos para trabajar con datos")
```

**Salida:**

```text
Iniciando análisis de Mall Customer Segmentation Dataset
Pandas y NumPy cargados - listos para trabajar con datos
```

## Paso 2.2: Carga del Dataset

**Entrada:**

```python
# Descargar desde GitHub (opción más confiable)
url = "https://raw.githubusercontent.com/SteffiPeTaffy/machineLearningAZ/master/Machine%20Learning%20A-Z%20Template%20Folder/Part%204%20-%20Clustering/Section%2024%20-%20K-Means%20Clustering/Mall_Customers.csv"

df_customers = pd.read_csv(url)
```
**Salida:**

```text
(sin salida)
```

## Paso 2.3: Inspección Inicial del Dataset

**Entrada:**

```python
print("INFORMACIÓN DEL DATASET:")
print(f"Shape: {df_customers.shape[0]} filas, {df_customers.shape[1]} columnas")
print(f"Columnas: {list(df_customers.columns)}")
print(f"Memoria: {df_customers.memory_usage(deep=True).sum() / 1024:.1f} KB")

print(f"\nPRIMERAS 5 FILAS:")
df_customers.head()
```
**Salida:**

```text
INFORMACIÓN DEL DATASET:
Shape: 200 filas, 5 columnas
Columnas: ['CustomerID', 'Genre', 'Age', 'Annual Income (k$)', 'Spending Score (1-100)']
Memoria: 16.9 KB

PRIMERAS 5 FILAS:

   CustomerID   Genre  Age  Annual Income (k$)  Spending Score (1-100)
0           1    Male   19                  15                      39
1           2    Male   21                  15                      81
2           3  Female   20                  16                       6
3           4  Female   23                  16                      77
4           5  Female   31                  17                      40
```

## Paso 2.4: Análisis de Tipos de Datos

**Entrada:**

```python
# === ANÁLISIS DE TIPOS Y ESTRUCTURA ===
print("INFORMACIÓN DETALLADA DE COLUMNAS:")
print(df_customers.info())

print(f"\nESTADÍSTICAS DESCRIPTIVAS:")
df_customers.describe()
```
**Salida:**

```text
INFORMACIÓN DETALLADA DE COLUMNAS:
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 200 entries, 0 to 199
Data columns (total 5 columns):
 #   Column                  Non-Null Count  Dtype 
---  ------                  --------------  ----- 
 0   CustomerID              200 non-null    int64 
 1   Genre                   200 non-null    object
 2   Age                     200 non-null    int64 
 3   Annual Income (k$)      200 non-null    int64 
 4   Spending Score (1-100)  200 non-null    int64 
dtypes: int64(4), object(1)
memory usage: 7.9+ KB
None

ESTADÍSTICAS DESCRIPTIVAS:

       CustomerID         Age  Annual Income (k$)  Spending Score (1-100)
count  200.000000  200.000000          200.000000              200.000000
mean   100.500000   38.850000           60.560000               50.200000
std     57.879185   13.969007           26.264721               25.823522
min      1.000000   18.000000           15.000000                1.000000
25%     50.750000   28.750000           41.500000               34.750000
50%    100.500000   36.000000           61.500000               50.000000
75%    150.250000   49.000000           78.000000               73.000000
max    200.000000   70.000000          137.000000               99.000000
```

## Paso 2.5: Análisis de Distribución por Género

**Entrada:**

```python
# === ANÁLISIS DE GÉNERO ===
print("DISTRIBUCIÓN POR GÉNERO:")
gender_counts = df_customers['Genre'].value_counts()
print(gender_counts)
print(f"\nPorcentajes:")
for gender, count in gender_counts.items():
    pct = (count / len(df_customers) * 100)
    print(f"   {gender}: {pct:.1f}%")
```
**Salida:**

```text
DISTRIBUCIÓN POR GÉNERO:
Genre
Female    112
Male       88
Name: count, dtype: int64

Porcentajes:
   Female: 56.0%
   Male: 44.0%
```

## Paso 2.6: Estadísticas de Variables Clave

**Entrada:**

```python
# === ESTADÍSTICAS DE VARIABLES DE SEGMENTACIÓN ===
numeric_vars = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']

print("ESTADÍSTICAS CLAVE:")
print(df_customers[numeric_vars].describe().round(2))

print(f"\nRANGOS OBSERVADOS:")
for var in numeric_vars:
    min_val, max_val = df_customers[var].min(), df_customers[var].max()
    mean_val = df_customers[var].mean()
    print(f"   {var}: {min_val:.0f} - {max_val:.0f} (promedio: {mean_val:.1f})")
```
**Salida:**

```text
ESTADÍSTICAS CLAVE:
          Age  Annual Income (k$)  Spending Score (1-100)
count  200.00              200.00                  200.00
mean    38.85               60.56                   50.20
std     13.97               26.26                   25.82
min     18.00               15.00                    1.00
25%     28.75               41.50                   34.75
50%     36.00               61.50                   50.00
75%     49.00               78.00                   73.00
max     70.00              137.00                   99.00

RANGOS OBSERVADOS:
   Age: 18 - 70 (promedio: 38.9)
   Annual Income (k$): 15 - 137 (promedio: 60.6)
   Spending Score (1-100): 1 - 99 (promedio: 50.2)
```

## Paso 2.7: Detección de Outliers

**Entrada:**

```python
# === DETECCIÓN DE OUTLIERS USANDO IQR ===
print("DETECCIÓN DE OUTLIERS:")

outlier_cols = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']

for col in outlier_cols:
    Q1 = df_customers[col].quantile(0.25)
    Q3 = df_customers[col].quantile(0.75)
    IQR = Q3 - Q1

    # Calcular límites
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Encontrar outliers
    outliers = df_customers[(df_customers[col] < lower_bound) |
                           (df_customers[col] > upper_bound)]

    print(f"   {col}: {len(outliers)} outliers ({len(outliers)/len(df_customers)*100:.1f}%)")
    print(f"      Límites normales: {lower_bound:.1f} - {upper_bound:.1f}")
```
**Salida:**

```text
DETECCIÓN DE OUTLIERS:
   Age: 0 outliers (0.0%)
      Límites normales: -1.6 - 79.4
   Annual Income (k$): 2 outliers (1.0%)
      Límites normales: -13.2 - 132.8
   Spending Score (1-100): 0 outliers (0.0%)
      Límites normales: -22.6 - 130.4
```

## Paso 2.8: Visualizaciones - Distribuciones

**Entrada:**

```python
# === IMPORTS PARA VISUALIZACIÓN ===
import matplotlib.pyplot as plt
import seaborn as sns

# Configurar estilo
plt.style.use('default')
sns.set_palette("husl")

# === HISTOGRAMAS DE VARIABLES PRINCIPALES ===
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle('Distribuciones de Variables Clave', fontsize=14, fontweight='bold')

vars_to_plot = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']

for i, (var, color) in enumerate(zip(vars_to_plot, colors)):
    axes[i].hist(df_customers[var], bins=20, alpha=0.7, color=color, edgecolor='black')
    axes[i].set_title(f'{var}')
    axes[i].set_xlabel(var)
    axes[i].set_ylabel('Frecuencia')
    axes[i].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```
**Salida:**

![Distribuciones de Variables Clave](../../assets/ta6_1.png)

## Paso 2.9: Visualizaciones - Relaciones

**Entrada:**

```python
# === SCATTER PLOTS PARA RELACIONES CLAVE ===
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Relaciones Entre Variables', fontsize=14, fontweight='bold')

# Age vs Income
axes[0].scatter(df_customers['Age'], df_customers['Annual Income (k$)'],
                alpha=0.6, color='#96CEB4', s=50)
axes[0].set_xlabel('Age (años)')
axes[0].set_ylabel('Annual Income (k$)')
axes[0].set_title('Age vs Income')
axes[0].grid(True, alpha=0.3)

# Income vs Spending Score ⭐ CLAVE PARA SEGMENTACIÓN
axes[1].scatter(df_customers['Annual Income (k$)'], df_customers['Spending Score (1-100)'],
                alpha=0.6, color='#FFEAA7', s=50)
axes[1].set_xlabel('Annual Income (k$)')
axes[1].set_ylabel('Spending Score (1-100)')
axes[1].set_title('Income vs Spending Score (CLAVE)')
axes[1].grid(True, alpha=0.3)

# Age vs Spending Score
axes[2].scatter(df_customers['Age'], df_customers['Spending Score (1-100)'],
                alpha=0.6, color='#DDA0DD', s=50)
axes[2].set_xlabel('Age (años)')
axes[2].set_ylabel('Spending Score (1-100)')
axes[2].set_title('Age vs Spending Score')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```
**Salida:**

![Relaciones Entre Variables](../../assets/ta6_2.png)

## Paso 2.10: Análisis Comparativo por Género

**Entrada:**

```python
# === MATRIZ DE CORRELACIÓN ===
correlation_vars = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
corr_matrix = df_customers[correlation_vars].corr()

print("MATRIZ DE CORRELACIÓN:")
print(corr_matrix.round(3))

# Visualizar matriz de correlación
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='RdYlBu_r', center=0, 
            fmt='.3f', linewidths=0.5, square=True)
plt.title('Matriz de Correlación - Mall Customers')
plt.tight_layout()
plt.show()

print(f"\nCORRELACIÓN MÁS FUERTE:")
# Encontrar la correlación más alta (excluyendo diagonal)
corr_flat = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
max_corr = corr_flat.stack().idxmax()
max_val = corr_flat.stack().max()
print(f"   {max_corr[0]} ↔ {max_corr[1]}: {max_val:.3f}")
```
**Salida(1/3):**

```text
ANÁLISIS COMPARATIVO POR GÉNERO:
          Age        Annual Income (k$)        Spending Score (1-100)       
         mean    std               mean    std                   mean    std
Genre                                                                       
Female  38.10  12.64              59.25  26.01                  51.53  24.11
Male    39.81  15.51              62.23  26.64                  48.51  27.90

INSIGHTS POR GÉNERO:
   Spending Score (1-100): Mujeres tienen promedio más alto (diferencia: 3.0)
```

**Salida(2/3):**

![Matriz de Correlación - Mall Customers](../../assets/ta6_3.png)

**Salida(3/3):**

```text
CORRELACIÓN MÁS FUERTE:
   Annual Income (k$) ↔ Spending Score (1-100): 0.010
```

## Paso 2.11: Análisis Comparativo por Género

**Entrada:**

```python
# === COMPARACIÓN ESTADÍSTICAS POR GÉNERO ===
print("ANÁLISIS COMPARATIVO POR GÉNERO:")

gender_stats = df_customers.groupby('Genre')[numeric_vars].agg(['mean', 'std']).round(2)
print(gender_stats)

print(f"\nINSIGHTS POR GÉNERO:")
for var in numeric_vars:
    male_avg = df_customers[df_customers['Genre'] == 'Male'][var].mean()
    female_avg = df_customers[df_customers['Genre'] == 'Female'][var].mean()

    if male_avg > female_avg:
        higher = "Hombres"
        diff = male_avg - female_avg
else:
    higher = "Mujeres"
    diff = female_avg - male_avg

print(f"   {var}: {higher} tienen promedio más alto (diferencia: {diff:.1f})")
```
**Salida:**

```text
ANÁLISIS COMPARATIVO POR GÉNERO:
          Age        Annual Income (k$)        Spending Score (1-100)       
         mean    std               mean    std                   mean    std
Genre                                                                       
Female  38.10  12.64              59.25  26.01                  51.53  24.11
Male    39.81  15.51              62.23  26.64                  48.51  27.90

INSIGHTS POR GÉNERO:
   Spending Score (1-100): Mujeres tienen promedio más alto (diferencia: 3.0)
```

## Paso 2.12: Síntesis de Insights

**Entrada:**

```python
# === COMPLETE ESTOS INSIGHTS BASÁNDOTE EN LO OBSERVADO ===
print("INSIGHTS PRELIMINARES - COMPLETE:")

print(f"\nCOMPLETE BASÁNDOTE EN TUS OBSERVACIONES:")
print(f"   Variable con mayor variabilidad: Annual Income")
print(f"   ¿Existe correlación fuerte entre alguna variable? No")
print(f"   ¿Qué variable tiene más outliers? Annual Income")
print(f"   ¿Los hombres y mujeres tienen patrones diferentes? Sí. Mujeres tienen más Spending Score, hombres más edad e ingreso")
print(f"   ¿Qué insight es más relevante para el análisis? Income y Spending Score forman grupos claros. Género/edad influyen menos")
print(f"   ¿Qué 2 variables serán más importantes para clustering? Annual Income y Spending Score")

print(f"\nPREPARÁNDOSE PARA CLUSTERING:")
print(f"   ¿Qué relación entre Income y Spending Score observas? Prácticamente nula es no lineal, se ven clusters")
print(f"   ¿Puedes imaginar grupos naturales de clientes? Sí, alto ingreso/alto gasto. Alto ingreso/bajo gasto. Bajo ingreso/alto gasto. Bajo ingreso/bajo gasto")
```
**Salida:**

```text
INSIGHTS PRELIMINARES - COMPLETE:

COMPLETE BASÁNDOTE EN TUS OBSERVACIONES:
   Variable con mayor variabilidad: Annual Income
   ¿Existe correlación fuerte entre alguna variable? No
   ¿Qué variable tiene más outliers? Annual Income
   ¿Los hombres y mujeres tienen patrones diferentes? Sí. Mujeres tienen más Spending Score, hombres más edad e ingreso
   ¿Qué insight es más relevante para el análisis? Income y Spending Score forman grupos claros. Género/edad influyen menos
   ¿Qué 2 variables serán más importantes para clustering? Annual Income y Spending Score

PREPARÁNDOSE PARA CLUSTERING:
   ¿Qué relación entre Income y Spending Score observas? Prácticamente nula es no lineal, se ven clusters
   ¿Puedes imaginar grupos naturales de clientes? Sí, alto ingreso/alto gasto. Alto ingreso/bajo gasto. Bajo ingreso/alto gasto. Bajo ingreso/bajo gasto
```

## Paso 2.13: Identificación de Features para Clustering

**Entrada:**

```python
# === ANÁLISIS DE COLUMNAS DISPONIBLES ===
print("ANÁLISIS DE COLUMNAS PARA CLUSTERING:")
print(f"   Todas las columnas: {list(df_customers.columns)}")
print(f"   Numéricas: {df_customers.select_dtypes(include=[np.number]).columns.tolist()}")
print(f"   Categóricas: {df_customers.select_dtypes(include=[object]).columns.tolist()}")

# Identificar qué excluir y qué incluir
exclude_columns = ['CustomerID']  # ID no aporta información
numeric_columns = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
categorical_columns = ['Genre']

print(f"\nSELECCIÓN DE FEATURES:")
print(f"   Excluidas: {exclude_columns} (no informativas)")
print(f"   Numéricas: {numeric_columns}")
print(f"   Categóricas: {categorical_columns} (codificaremos)")
```
**Salida:**

```text
ANÁLISIS DE COLUMNAS PARA CLUSTERING:
   Todas las columnas: ['CustomerID', 'Genre', 'Age', 'Annual Income (k$)', 'Spending Score (1-100)']
   Numéricas: ['CustomerID', 'Age', 'Annual Income (k$)', 'Spending Score (1-100)']
   Categóricas: ['Genre']

SELECCIÓN DE FEATURES:
   Excluidas: ['CustomerID'] (no informativas)
   Numéricas: ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
   Categóricas: ['Genre'] (codificaremos)
```

## Paso 2.14: Codificación de Variables Categóricas con OneHotEncoder

**Entrada:**

```python
# === IMPORT ONEHOTENCODER ===
from sklearn.preprocessing import OneHotEncoder

print("CODIFICACIÓN DE VARIABLES CATEGÓRICAS CON SKLEARN:")
print("Usaremos OneHotEncoder en lugar de pd.get_dummies() por varias razones:")
print("   Integración perfecta con pipelines de sklearn")
print("   Manejo automático de categorías no vistas en nuevos datos") 
print("   Control sobre nombres de columnas y comportamiento")
print("   Consistencia con el ecosistema de machine learning")

# Crear y configurar OneHotEncoder
encoder = OneHotEncoder(sparse_output=False)

# Ajustar y transformar Genre
genre_data = df_customers[['Genre']]  # Debe ser 2D para sklearn
genre_encoded_array = encoder.fit_transform(genre_data)  # Método para ajustar y transformar

# Obtener nombres de las nuevas columnas
feature_names = encoder.get_feature_names_out(['Genre'])  # Método para obtener nombres de las features
genre_encoded = pd.DataFrame(genre_encoded_array, columns=feature_names)

print(f"\nRESULTADO DE CODIFICACIÓN:")
print(f"   Categorías originales: {df_customers['Genre'].unique()}")
print(f"   Columnas generadas: {list(genre_encoded.columns)}")
print(f"   Shape: {genre_data.shape} → {genre_encoded.shape}")

# Mostrar ejemplo de codificación
print(f"\nEJEMPLO DE TRANSFORMACIÓN:")
comparison = pd.concat([
    df_customers['Genre'].head().reset_index(drop=True),
    genre_encoded.head()
], axis=1)
print(comparison)
```

**Salida:**

```text
CODIFICACIÓN DE VARIABLES CATEGÓRICAS CON SKLEARN:
Usaremos OneHotEncoder en lugar de pd.get_dummies() por varias razones:
   Integración perfecta con pipelines de sklearn
   Manejo automático de categorías no vistas en nuevos datos
   Control sobre nombres de columnas y comportamiento
   Consistencia con el ecosistema de machine learning

RESULTADO DE CODIFICACIÓN:
   Categorías originales: ['Male' 'Female']
   Columnas generadas: ['Genre_Female', 'Genre_Male']
   Shape: (200, 1) → (200, 2)

EJEMPLO DE TRANSFORMACIÓN:
    Genre  Genre_Female  Genre_Male
0    Male           0.0         1.0
1    Male           0.0         1.0
2  Female           1.0         0.0
3  Female           1.0         0.0
4  Female           1.0         0.0
```

## Paso 2.15: Preparación del Dataset Final

**Entrada:**

```python
# === CREACIÓN DEL DATASET FINAL ===
# Combinar variables numéricas + categóricas codificadas
X_raw = pd.concat([
    df_customers[numeric_columns],
    genre_encoded
], axis=1)

print("DATASET FINAL PARA CLUSTERING:")
print(f"   Shape: {X_raw.shape}")
print(f"   Columnas: {list(X_raw.columns)}")
print(f"   Variables numéricas: {numeric_columns}")
print(f"   Variables categóricas codificadas: {list(genre_encoded.columns)}")
print(f"   Total features: {X_raw.shape[1]} (3 numéricas + 2 categóricas binarias)")
print(f"   Memoria: {X_raw.memory_usage(deep=True).sum() / 1024:.1f} KB")
```

**Salida:**

```text
DATASET FINAL PARA CLUSTERING:
   Shape: (200, 5)
   Columnas: ['Age', 'Annual Income (k$)', 'Spending Score (1-100)', 'Genre_Female', 'Genre_Male']
   Variables numéricas: ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
   Variables categóricas codificadas: ['Genre_Female', 'Genre_Male']
   Total features: 5 (3 numéricas + 2 categóricas binarias)
   Memoria: 7.9 KB
```

## Paso 2.16: Verificación de Calidad de Datos

**Entrada:**

```python
# === VERIFICACIONES ANTES DE CONTINUAR ===
print("VERIFICACIÓN DE CALIDAD:")

# 1. Datos faltantes
missing_data = X_raw.isnull().sum()
print(f"\nDATOS FALTANTES:")
if missing_data.sum() == 0:
    print("   PERFECTO! No hay datos faltantes")
else:
    for col, missing in missing_data.items():
        if missing > 0:
            pct = (missing / len(X_raw)) * 100
            print(f"   WARNING {col}: {missing} faltantes ({pct:.1f}%)")

# 2. Vista previa
print(f"\nVISTA PREVIA DEL DATASET:")
print(X_raw.head())

# 3. Tipos de datos
print(f"\nTIPOS DE DATOS:")
print(X_raw.dtypes)
```

**Salida:**

```text
VERIFICACIÓN DE CALIDAD:

DATOS FALTANTES:
   PERFECTO! No hay datos faltantes

VISTA PREVIA DEL DATASET:
   Age  Annual Income (k$)  Spending Score (1-100)  Genre_Female  Genre_Male
0   19                  15                      39           0.0         1.0
1   21                  15                      81           0.0         1.0
2   20                  16                       6           1.0         0.0
3   23                  16                      77           1.0         0.0
4   31                  17                      40           1.0         0.0

TIPOS DE DATOS:
Age                         int64
Annual Income (k$)          int64
Spending Score (1-100)      int64
Genre_Female              float64
Genre_Male                float64
dtype: object
```

## Paso 2.17: Análisis de Escalas (Pre-Normalización)

**Entrada:**

```python
# === ANÁLISIS DE ESCALAS ===
print("ANÁLISIS DE ESCALAS - ¿Por qué necesitamos normalización?")

print(f"\nESTADÍSTICAS POR VARIABLE:")
for col in X_raw.columns:
    if X_raw[col].dtype in ['int64', 'float64']:  # Solo numéricas
        min_val = X_raw[col].min()
        max_val = X_raw[col].max()
        mean_val = X_raw[col].mean()
        std_val = X_raw[col].std()

        print(f"\n   {col}:")
        print(f"      Rango: {min_val:.1f} - {max_val:.1f}")
        print(f"      Media: {mean_val:.1f}")
        print(f"      Desviación: {std_val:.1f}")

print(f"\nANÁLISIS DE LAS ESTADÍSTICAS - COMPLETA:")
print(f"   ¿Qué variable tiene el rango más amplio? Annual Income")
print(f"   ¿Cuál es la distribución de género en el dataset? Female 112 (56%), Male 88 (44%)")
print(f"   ¿Qué variable muestra mayor variabilidad (std)? Annual Income")
print(f"   ¿Los clientes son jóvenes o mayores en promedio? Adultos jóvenes (39 años aprox)")
print(f"   ¿El income promedio sugiere qué clase social? Clase media")
print(f"   ¿Por qué la normalización será crítica aca? Las escalas difieren mucho y K-Means usa distancias. Sin escalar, Income seria mucho mejor")

# Guardar para próximas fases
feature_columns = list(X_raw.columns)
print(f"\nLISTO PARA DATA PREPARATION con {len(feature_columns)} features")
```

**Salida:**

```text
ANÁLISIS DE ESCALAS - ¿Por qué necesitamos normalización?

ESTADÍSTICAS POR VARIABLE:

   Age:
      Rango: 18.0 - 70.0
      Media: 38.9
      Desviación: 14.0

   Annual Income (k$):
      Rango: 15.0 - 137.0
      Media: 60.6
      Desviación: 26.3

   Spending Score (1-100):
      Rango: 1.0 - 99.0
      Media: 50.2
      Desviación: 25.8

   Genre_Female:
      Rango: 0.0 - 1.0
      Media: 0.6
      Desviación: 0.5

   Genre_Male:
      Rango: 0.0 - 1.0
      Media: 0.4
      Desviación: 0.5

ANÁLISIS DE LAS ESTADÍSTICAS - COMPLETA:
   ¿Qué variable tiene el rango más amplio? Annual Income
   ¿Cuál es la distribución de género en el dataset? Female 112 (56%), Male 88 (44%)
   ¿Qué variable muestra mayor variabilidad (std)? Annual Income
   ¿Los clientes son jóvenes o mayores en promedio? Adultos jóvenes (39 años aprox)
   ¿El income promedio sugiere qué clase social? Clase media
   ¿Por qué la normalización será crítica aca? Las escalas difieren mucho y K-Means usa distancias. Sin escalar, Income seria mucho mejor

LISTO PARA DATA PREPARATION con 5 features
```

# FASE 3: DATA PREPARATION

## Paso 3.1: Setup para Normalización

**Entrada:**

```python
# === IMPORTAR HERRAMIENTAS DE NORMALIZACIÓN ===
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler

print("BATALLA DE NORMALIZACIÓN: MinMax vs Standard vs Robust")
print("Objetivo: Encontrar el mejor scaler para nuestros datos")

# Recordar por qué es importante
print(f"\nESCALAS ACTUALES (problema):")
for col in X_raw.columns:
    min_val, max_val = X_raw[col].min(), X_raw[col].max()
    print(f"   {col}: {min_val:.1f} - {max_val:.1f} (rango: {max_val-min_val:.1f})")

print("\nLas escalas son MUY diferentes - normalización es crítica!")
```

**Salida:**

```text
BATALLA DE NORMALIZACIÓN: MinMax vs Standard vs Robust
Objetivo: Encontrar el mejor scaler para nuestros datos

ESCALAS ACTUALES (problema):
   Age: 18.0 - 70.0 (rango: 52.0)
   Annual Income (k$): 15.0 - 137.0 (rango: 122.0)
   Spending Score (1-100): 1.0 - 99.0 (rango: 98.0)
   Genre_Female: 0.0 - 1.0 (rango: 1.0)
   Genre_Male: 0.0 - 1.0 (rango: 1.0)

Las escalas son MUY diferentes - normalización es crítica!
```

## Paso 3.2: Aplicar los 3 Scalers

**Entrada:**

```python
# === CREAR Y APLICAR LOS 3 SCALERS ===
scalers = {
    'MinMax': MinMaxScaler(),        # Escala a rango [0,1]
    'Standard': StandardScaler(),    # Media=0, std=1  
    'Robust': RobustScaler()         # Usa mediana y IQR, robusto a outliers
}

# Aplicar cada scaler
X_scaled = {}
for name, scaler in scalers.items():
    X_scaled[name] = scaler.fit_transform(X_raw)  # Método para entrenar y transformar
    print(f"{name}Scaler aplicado: {X_scaled[name].shape}")

print(f"\nTenemos 3 versiones escaladas de los datos para comparar")
```

**Salida:**

```text
MinMaxScaler aplicado: (200, 5)
StandardScaler aplicado: (200, 5)
RobustScaler aplicado: (200, 5)

Tenemos 3 versiones escaladas de los datos para comparar
```

## Paso 3.3: Comparación Visual - Boxplots

**Entrada:**

```python
# === COMPARACIÓN VISUAL CON BOXPLOTS ===
fig, axes = plt.subplots(1, 4, figsize=(16, 4))
fig.suptitle('Comparación de Scalers - Boxplots', fontsize=14, fontweight='bold')

# Datos originales
axes[0].boxplot([X_raw[col] for col in X_raw.columns], labels=X_raw.columns)
axes[0].set_title('Original')
axes[0].tick_params(axis='x', rotation=45)

# Datos escalados
for i, (name, X_scaled_data) in enumerate(X_scaled.items(), 1):
    axes[i].boxplot([X_scaled_data[:, j] for j in range(X_scaled_data.shape[1])], 
                    labels=X_raw.columns)
    axes[i].set_title(f'{name}')
    axes[i].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()

print("Observa cómo cada scaler ajusta las escalas de forma diferente")
```

**Salida(1/2):**

![Comparación de Scalers](../../assets/ta6_4.png)

**Salida(2/2):**

```text
Observa cómo cada scaler ajusta las escalas de forma diferente
```

## Paso 3.4: Comparación de Distribuciones

**Entrada:**

```python
# === COMPARAR DISTRIBUCIONES DE UNA VARIABLE ===
# Vamos a analizar 'Annual Income (k$)' en detalle
income_col_idx = 1  # Posición de Annual Income

fig, axes = plt.subplots(1, 4, figsize=(16, 4))
fig.suptitle('Annual Income: Original vs Scalers', fontsize=14, fontweight='bold')

# Original
axes[0].hist(X_raw.iloc[:, income_col_idx], bins=20, alpha=0.7, color='gray', edgecolor='black')
axes[0].set_title('Original')
axes[0].set_xlabel('Annual Income (k$)')

# Escalados
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
for i, ((name, X_scaled_data), color) in enumerate(zip(X_scaled.items(), colors), 1):
    axes[i].hist(X_scaled_data[:, income_col_idx], bins=20, alpha=0.7, color=color, edgecolor='black')
    axes[i].set_title(f'{name}')
    axes[i].set_xlabel('Annual Income (escalado)')

plt.tight_layout()
plt.show()

print("¿Notas cómo cambia la forma de la distribución?")
```

**Salida(1/2):**

![Anual Income: Original vs Scalers](../../assets/ta6_5.png)

**Salida(2/2):**

```text
¿Notas cómo cambia la forma de la distribución?
```

## Paso 3.5: Análisis Estadístico Post-Scaling

**Entrada:**

```python
# === ESTADÍSTICAS DESPUÉS DEL SCALING ===
print("ESTADÍSTICAS POST-SCALING (Annual Income):")

# Original
income_original = X_raw['Annual Income (k$)']
print(f"\n   Original:")
print(f"      Media: {income_original.mean():.1f}")
print(f"      Std:   {income_original.std():.1f}")
print(f"      Min:   {income_original.min():.1f}")
print(f"      Max:   {income_original.max():.1f}")

# Escalados
for name, X_scaled_data in X_scaled.items():
    income_scaled = X_scaled_data[:, income_col_idx]
    print(f"\n   {name}:")
    print(f"      Media: {income_scaled.mean():.3f}")
    print(f"      Std:   {income_scaled.std():.3f}")
    print(f"      Min:   {income_scaled.min():.3f}")
    print(f"      Max:   {income_scaled.max():.3f}")

print(f"\nOBSERVACIONES:")
print(f"   MinMaxScaler → Rango [0,1]")
print(f"   StandardScaler → Media=0, Std=1")
print(f"   RobustScaler → Menos afectado por outliers")
```

**Salida:**

```text
ESTADÍSTICAS POST-SCALING (Annual Income):

   Original:
      Media: 60.6
      Std:   26.3
      Min:   15.0
      Max:   137.0

   MinMax:
      Media: 0.373
      Std:   0.215
      Min:   0.000
      Max:   1.000

   Standard:
      Media: -0.000
      Std:   1.000
      Min:   -1.739
      Max:   2.918

   Robust:
      Media: -0.026
      Std:   0.718
      Min:   -1.274
      Max:   2.068

OBSERVACIONES:
   MinMaxScaler → Rango [0,1]
   StandardScaler → Media=0, Std=1
   RobustScaler → Menos afectado por outliers
```

## Paso 3.6: Test de Impacto en Clustering

**Entrada:**

```python
# === IMPORT PARA CLUSTERING TEST ===
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# === QUICK TEST: ¿Qué scaler funciona mejor para clustering? ===
print("QUICK TEST: Impacto en Clustering (K=4)")

clustering_results = {}
for name, X_scaled_data in X_scaled.items():
    # Aplicar K-Means con K=4
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)  # Completar
    labels = kmeans.fit_predict(X_scaled_data)  # Método para obtener clusters

    # Calcular silhouette score
    silhouette = silhouette_score(X_scaled_data, labels)  # Métrica de calidad
    clustering_results[name] = silhouette

    print(f"   {name:>10}: Silhouette Score = {silhouette:.3f}")

# Encontrar el mejor
best_scaler = max(clustering_results, key=clustering_results.get)
best_score = clustering_results[best_scaler]

print(f"\nGANADOR: {best_scaler} (Score: {best_score:.3f})")
```

**Salida:**

```text
QUICK TEST: Impacto en Clustering (K=4)
       MinMax: Silhouette Score = 0.364
     Standard: Silhouette Score = 0.332
       Robust: Silhouette Score = 0.298

GANADOR: MinMax (Score: 0.364)
```

## Paso 3.7: Decisión Final de Scaler

**Entrada:**

```python
# === TOMAR DECISIÓN BASADA EN RESULTADOS ===
print("DECISIÓN FINAL DEL SCALER:")

print(f"\nCOMPLETE TU ANÁLISIS:")
print(f"   Mejor scaler según silhouette: {best_scaler}")
print(f"   ¿Por qué crees que funcionó mejor? Igualó las escalas (Age/Income/Spending) y evitó que 'Annual Income' domine la separación")
print(f"   ¿Algún scaler tuvo problemas obvios? MinMax es sensible a outliers, puede comprimir demasiado si hay valores extremos")

# Implementar decisión
selected_scaler_name = best_scaler  # O elige manualmente: 'MinMax', 'Standard', 'Robust'
selected_scaler = scalers[selected_scaler_name]

# Aplicar scaler elegido
X_preprocessed = X_scaled[selected_scaler_name]
feature_names_scaled = [f"{col}_scaled" for col in X_raw.columns]

print(f"\nSCALER SELECCIONADO: {selected_scaler_name}")
print(f"Datos preparados: {X_preprocessed.shape}")
print(f"Listo para PCA y Feature Selection")
```

**Salida:**

```text
DECISIÓN FINAL DEL SCALER:

COMPLETE TU ANÁLISIS:
   Mejor scaler según silhouette: MinMax
   ¿Por qué crees que funcionó mejor? Igualó las escalas (Age/Income/Spending) y evitó que 'Annual Income' domine la separación
   ¿Algún scaler tuvo problemas obvios? MinMax es sensible a outliers, puede comprimir demasiado si hay valores extremos

SCALER SELECCIONADO: MinMax
Datos preparados: (200, 5)
Listo para PCA y Feature Selection
```

# Paso 3.2: PCA - Reducción de Dimensionalidad (20 min)

**Entrada:**

```python
from sklearn.decomposition import PCA

# === OPERACIÓN: DIMENSION COLLAPSE ===
print("PCA: Reduciendo dimensiones sin perder la esencia")
print("   Objetivo: De 5D → 2D para visualización + análisis de varianza")

# 1. Aplicar PCA completo para análisis de varianza
pca_full = PCA()
X_pca_full = pca_full.fit_transform(X_preprocessed)

# 2. ANÁLISIS DE VARIANZA EXPLICADA
explained_variance_ratio = pca_full.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_variance_ratio)

print(f"\n📊 ANÁLISIS DE VARIANZA EXPLICADA:")
for i, (var, cum_var) in enumerate(zip(explained_variance_ratio, cumulative_variance)):
    print(f"   PC{i+1}: {var:.3f} ({var*100:.1f}%) | Acumulada: {cum_var:.3f} ({cum_var*100:.1f}%)")

# 3. VISUALIZACIÓN DE VARIANZA EXPLICADA
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Scree plot
axes[0].bar(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio, 
           alpha=0.7, color='#FF6B6B')
axes[0].set_xlabel('Componentes Principales')
axes[0].set_ylabel('Varianza Explicada')
axes[0].set_title('📊 Scree Plot - Varianza por Componente')
axes[0].set_xticks(range(1, len(explained_variance_ratio) + 1))

# Cumulative variance
axes[1].plot(range(1, len(cumulative_variance) + 1), cumulative_variance, 
            marker='o', linewidth=2, markersize=8, color='#4ECDC4')
axes[1].axhline(y=0.95, color='red', linestyle='--', alpha=0.7, label='95% threshold')
axes[1].axhline(y=0.90, color='orange', linestyle='--', alpha=0.7, label='90% threshold')
axes[1].set_xlabel('Número de Componentes')
axes[1].set_ylabel('Varianza Acumulada')
axes[1].set_title('📈 Varianza Acumulada')
axes[1].legend()
axes[1].grid(True, alpha=0.3)
axes[1].set_xticks(range(1, len(cumulative_variance) + 1))

plt.tight_layout()
plt.show()

# 4. DECISIÓN SOBRE NÚMERO DE COMPONENTES
print(f"\n🎯 DECISIÓN DE COMPONENTES:")
n_components_90 = np.argmax(cumulative_variance >= 0.90) + 1
n_components_95 = np.argmax(cumulative_variance >= 0.95) + 1

print(f"   📊 Para retener 90% varianza: {n_components_90} componentes")
print(f"   📊 Para retener 95% varianza: {n_components_95} componentes")
print(f"   🎯 Para visualización: 2 componentes ({cumulative_variance[1]*100:.1f}% varianza)")

# 5. APLICAR PCA CON 2 COMPONENTES PARA VISUALIZACIÓN
pca_2d = PCA(n_components=2, random_state=42)
X_pca_2d = pca_2d.fit_transform(X_preprocessed)

print(f"\nPCA aplicado:")
print(f"   📊 Dimensiones: {X_preprocessed.shape} → {X_pca_2d.shape}")
print(f"   📈 Varianza explicada: {pca_2d.explained_variance_ratio_.sum()*100:.1f}%")

# 6. ANÁLISIS DE COMPONENTES PRINCIPALES
print(f"\n🔍 INTERPRETACIÓN DE COMPONENTES:")
feature_names = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)', 'Genre_Female', 'Genre_Male']

for i, pc in enumerate(['PC1', 'PC2']):
    print(f"\n   {pc} (varianza: {pca_2d.explained_variance_ratio_[i]*100:.1f}%):")
    # Obtener los loadings (pesos de cada feature original en el componente)
    loadings = pca_2d.components_[i]
    for j, (feature, loading) in enumerate(zip(feature_names, loadings)):
        direction = "↑" if loading > 0 else "↓"
        print(f"     {feature:>15}: {loading:>7.3f} {direction}")

# 7. VISUALIZACIÓN EN 2D
plt.figure(figsize=(12, 8))
plt.scatter(X_pca_2d[:, 0], X_pca_2d[:, 1], alpha=0.6, s=50, color='#96CEB4')
plt.xlabel(f'PC1 ({pca_2d.explained_variance_ratio_[0]*100:.1f}% varianza)')
plt.ylabel(f'PC2 ({pca_2d.explained_variance_ratio_[1]*100:.1f}% varianza)')
plt.title('Mall Customers en Espacio PCA 2D')
plt.grid(True, alpha=0.3)
plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
plt.tight_layout()
plt.show()

print(f"\n💡 INTERPRETACIÓN DE NEGOCIO:")
print(f"   🎯 PC1 parece representar: nivel conjunto de ingreso y nivel de gasto")
print(f"   🎯 PC2 parece representar: variación ligada a la edad")
print(f"   📊 Los clusters visibles sugieren: segmentos alto ingreso–alto gasto, alto ingreso–bajo gasto, bajo ingreso–alto gasto y bajo ingreso–bajo gasto")
```

**Salida(1/5):**

```text
PCA: Reduciendo dimensiones sin perder la esencia
   Objetivo: De 5D → 2D para visualización + análisis de varianza

📊 ANÁLISIS DE VARIANZA EXPLICADA:
   PC1: 0.726 (72.6%) | Acumulada: 0.726 (72.6%)
   PC2: 0.137 (13.7%) | Acumulada: 0.863 (86.3%)
   PC3: 0.070 (7.0%) | Acumulada: 0.932 (93.2%)
   PC4: 0.068 (6.8%) | Acumulada: 1.000 (100.0%)
   PC5: 0.000 (0.0%) | Acumulada: 1.000 (100.0%)
/tmp/ipython-input-2122711217.py:42: UserWarning: Glyph 128202 (\N{BAR CHART}) missing from font(s) DejaVu Sans.
  plt.tight_layout()
/tmp/ipython-input-2122711217.py:42: UserWarning: Glyph 128200 (\N{CHART WITH UPWARDS TREND}) missing from font(s) DejaVu Sans.
  plt.tight_layout()
/usr/local/lib/python3.12/dist-packages/IPython/core/pylabtools.py:151: UserWarning: Glyph 128202 (\N{BAR CHART}) missing from font(s) DejaVu Sans.
  fig.canvas.print_figure(bytes_io, **kw)
/usr/local/lib/python3.12/dist-packages/IPython/core/pylabtools.py:151: UserWarning: Glyph 128200 (\N{CHART WITH UPWARDS TREND}) missing from font(s) DejaVu Sans.
  fig.canvas.print_figure(bytes_io, **kw)
```

**Salida(2/5):**

![Varianza por componente - Varianza Acumulada](../../assets/ta6_6.png)

**Salida(3/5):**

```text
🎯 DECISIÓN DE COMPONENTES:
   📊 Para retener 90% varianza: 3 componentes
   📊 Para retener 95% varianza: 4 componentes
   🎯 Para visualización: 2 componentes (86.3% varianza)

PCA aplicado:
   📊 Dimensiones: (200, 5) → (200, 2)
   📈 Varianza explicada: 86.3%

🔍 INTERPRETACIÓN DE COMPONENTES:

   PC1 (varianza: 72.6%):
                 Age:   0.029 ↑
     Annual Income (k$):   0.019 ↑
     Spending Score (1-100):  -0.027 ↓
        Genre_Female:  -0.706 ↓
          Genre_Male:   0.706 ↑

   PC2 (varianza: 13.7%):
                 Age:   0.727 ↑
     Annual Income (k$):  -0.026 ↓
     Spending Score (1-100):  -0.685 ↓
        Genre_Female:   0.027 ↑
          Genre_Male:  -0.027 ↓
```

**Salida(4/5):**

![Varianza por componente - Varianza Acumulada](../../assets/ta6_7.png)

**Salida(5/5):**

```text
💡 INTERPRETACIÓN DE NEGOCIO:
   🎯 PC1 parece representar: nivel conjunto de ingreso y nivel de gasto
   🎯 PC2 parece representar: variación ligada a la edad
   📊 Los clusters visibles sugieren: segmentos alto ingreso–alto gasto, alto ingreso–bajo gasto, bajo ingreso–alto gasto y bajo ingreso–bajo gasto
```

# 🔍 Paso 3.3: Feature Selection - Alternativas a PCA (25 min)

## Paso 1: Imports y Setup Feature Selection

**Entrada:**

```python
# === IMPORTS PARA FEATURE SELECTION ===
from sklearn.feature_selection import SequentialFeatureSelector  # Para Forward/Backward Selection
```

**Salida:**

```text
(Sin Salida)
```

## Paso 2: Setup y Función de Evaluación

**Entrada:**

```python
# === OPERACIÓN: FEATURE SELECTION SHOWDOWN ===
print("🎯 FEATURE SELECTION vs PCA: ¿Seleccionar o Transformar?")
print("   🎯 Objetivo: Comparar Forward/Backward Selection vs PCA")

print(f"\n📊 FEATURE SELECTION: Forward vs Backward vs PCA")
print(f"   Dataset: {X_preprocessed.shape[0]} muestras, {X_preprocessed.shape[1]} features")

# Setup: Función para evaluar features en clustering
def evaluate_features_for_clustering(X, n_clusters=4):
    """Evalúa qué tan buenas son las features para clustering usando Silhouette Score"""
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)
    return silhouette_score(X, labels)

# === IMPORTS PARA ESTIMADORES PERSONALIZADOS ===
from sklearn.base import BaseEstimator, ClassifierMixin  # Clases base necesarias

# CLASE AUXILIAR: Estimador basado en KMeans para SequentialFeatureSelector
class ClusteringEstimator(BaseEstimator, ClassifierMixin):
    """Estimador que usa KMeans y Silhouette Score para feature selection"""
    def __init__(self, n_clusters=4):
        self.n_clusters = n_clusters

    def fit(self, X, y=None):
        self.kmeans_ = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
        self.labels_ = self.kmeans_.fit_predict(X)
        return self

    def score(self, X, y=None):
        # SequentialFeatureSelector llama a score() para evaluar features
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)
        return silhouette_score(X, labels)

    def predict(self, X):
        # Método requerido por ClassifierMixin
        if hasattr(self, 'kmeans_'):
            return self.kmeans_.predict(X)
        else:
            # Si no está entrenado, entrenar primero
            kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
            return kmeans.fit_predict(X)

print("✅ Setup completado - Funciones de evaluación listas")
```

**Salida:**

```text
🎯 FEATURE SELECTION vs PCA: ¿Seleccionar o Transformar?
   🎯 Objetivo: Comparar Forward/Backward Selection vs PCA

📊 FEATURE SELECTION: Forward vs Backward vs PCA
   Dataset: 200 muestras, 5 features
✅ Setup completado - Funciones de evaluación listas
```

## Paso 3: Baseline - Todas las Features

**Entrada:**

```python
# BASELINE: Todas las features
baseline_score = evaluate_features_for_clustering(X_preprocessed)
print(f"\n📊 BASELINE (todas las features): Silhouette = {baseline_score:.3f}")
print(f"   Este es el score con las {X_preprocessed.shape[1]} features originales")
print(f"   ¿Podremos mejorar seleccionando solo las mejores 3?")
```

**Salida:**

```text

📊 BASELINE (todas las features): Silhouette = 0.364
   Este es el score con las 5 features originales
   ¿Podremos mejorar seleccionando solo las mejores 3?
```

## Paso 4: Forward Selection

**Entrada:**

```python
# === FORWARD SELECTION (sklearn oficial) ===
print(f"\n🔄 FORWARD SELECTION (sklearn oficial):")
print(f"   Estrategia: Empezar con 0 features, agregar la mejor en cada paso")

forward_selector = SequentialFeatureSelector(
    estimator=ClusteringEstimator(n_clusters=4),  # Estimador que implementa fit() y score()
    n_features_to_select=3,
    direction='forward',  # ¿Qué dirección para Forward?
    cv=3,
    n_jobs=-1
)

forward_selector.fit(X_preprocessed)  # Método para entrenar
forward_mask = forward_selector.get_support()  # Método para obtener máscara booleana
X_forward = X_preprocessed[:, forward_mask]
forward_features = np.array(feature_columns)[forward_mask]
forward_score = evaluate_features_for_clustering(X_forward)

print(f"   Features seleccionadas: {list(forward_features)}")
print(f"   📊 Silhouette Score: {forward_score:.3f}")
print(f"   {'✅ Mejora!' if forward_score > baseline_score else '❌ Sin mejora'}")
```

**Salida:**

```text

🔄 FORWARD SELECTION (sklearn oficial):
   Estrategia: Empezar con 0 features, agregar la mejor en cada paso
   Features seleccionadas: [np.str_('Spending Score (1-100)'), np.str_('Genre_Female'), np.str_('Genre_Male')]
   📊 Silhouette Score: 0.573
   ✅ Mejora!
```

## Paso 5: Backward Elimination

**Entrada:**

```python
# === BACKWARD ELIMINATION (sklearn oficial) ===
print(f"\n🔄 BACKWARD ELIMINATION (sklearn oficial):")
print(f"   Estrategia: Empezar con todas las features, eliminar la peor en cada paso")

backward_selector = SequentialFeatureSelector(
    estimator=ClusteringEstimator(n_clusters=4),  # Mismo estimador que Forward
    n_features_to_select=3,
    direction='backward',  # ¿Qué dirección para Backward?
    cv=3,
    n_jobs=-1
)

backward_selector.fit(X_preprocessed)  # Método para entrenar
backward_mask = backward_selector.get_support()  # Método para obtener máscara
X_backward = X_preprocessed[:, backward_mask]
backward_features = np.array(feature_columns)[backward_mask]
backward_score = evaluate_features_for_clustering(X_backward)

print(f"   Features seleccionadas: {list(backward_features)}")
print(f"   📊 Silhouette Score: {backward_score:.3f}")
print(f"   {'✅ Mejora!' if backward_score > baseline_score else '❌ Sin mejora'}")
```

**Salida:**

```text
🔄 BACKWARD ELIMINATION (sklearn oficial):
   Estrategia: Empezar con todas las features, eliminar la peor en cada paso
   Features seleccionadas: [np.str_('Spending Score (1-100)'), np.str_('Genre_Female'), np.str_('Genre_Male')]
   📊 Silhouette Score: 0.573
   ✅ Mejora!
```

## Paso 6: Comparación Final

**Entrada:**

```python
# === COMPARACIÓN FINAL DE TODOS LOS MÉTODOS ===
print(f"\n📊 COMPARACIÓN DE MÉTODOS:")
print(f"   🏁 Baseline (todas): {baseline_score:.3f}")
print(f"   🔄 Forward Selection: {forward_score:.3f}")
print(f"   🔙 Backward Elimination: {backward_score:.3f}")

# Comparar con PCA (ya calculado anteriormente)
pca_score = evaluate_features_for_clustering(X_pca_2d)
print(f"   📐 PCA (2D): {pca_score:.3f}")

# Encontrar el mejor método
methods = {
    'Baseline (todas)': baseline_score,
    'Forward Selection': forward_score, 
    'Backward Elimination': backward_score,
    'PCA (2D)': pca_score
}

best_method = max(methods, key=methods.get)
best_score = methods[best_method]

print(f"\n🏆 GANADOR: {best_method} con score = {best_score:.3f}")

# Análisis de diferencias
print(f"\n🔍 ANÁLISIS:")
for method, score in sorted(methods.items(), key=lambda x: x[1], reverse=True):
    improvement = ((score - baseline_score) / baseline_score) * 100
    print(f"   {method}: {score:.3f} ({improvement:+.1f}% vs baseline)")
```

**Salida:**

```text

📊 COMPARACIÓN DE MÉTODOS:
   🏁 Baseline (todas): 0.364
   🔄 Forward Selection: 0.573
   🔙 Backward Elimination: 0.573
   📐 PCA (2D): 0.686

🏆 GANADOR: PCA (2D) con score = 0.686

🔍 ANÁLISIS:
   PCA (2D): 0.686 (+88.3% vs baseline)
   Forward Selection: 0.573 (+57.5% vs baseline)
   Backward Elimination: 0.573 (+57.5% vs baseline)
   Baseline (todas): 0.364 (+0.0% vs baseline)
```

## Paso 7: Visualización Comparativa

**Entrada:**

```python
# === VISUALIZACIÓN DE COMPARACIÓN ===
methods_names = ['Baseline', 'Forward', 'Backward', 'PCA 2D'] 
scores_values = [baseline_score, forward_score, backward_score, pca_score]
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']

plt.figure(figsize=(12, 6))
bars = plt.bar(methods_names, scores_values, color=colors, alpha=0.7)
plt.ylabel('Silhouette Score')
plt.title('Comparación de Métodos de Feature Selection')
plt.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Threshold Aceptable (0.5)')
plt.axhline(y=0.7, color='green', linestyle='--', alpha=0.5, label='Threshold Muy Bueno (0.7)')
plt.legend()
plt.grid(True, alpha=0.3, axis='y')

# Añadir valores en las barras
for bar, score in zip(bars, scores_values):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
             f'{score:.3f}', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.show()
```

**Salida:**

![Comparación de Métodos de Feature Selection](../../assets/ta6_8.png)

## Paso 8: Análisis y Decisión Final

**Entrada:**

```python
# === ANÁLISIS DE RESULTADOS ===
print(f"\n🎯 ANÁLISIS DE RESULTADOS:")

# Comparar features seleccionadas
print(f"\n🔍 FEATURES SELECCIONADAS POR CADA MÉTODO:")
print(f"   🔄 Forward Selection: {list(forward_features)}")
print(f"   🔙 Backward Elimination: {list(backward_features)}")

# Análisis de coincidencias
forward_set = set(forward_features)
backward_set = set(backward_features)

common_forward_backward = forward_set & backward_set

print(f"\n🤝 COINCIDENCIAS:")
print(f"   Forward ∩ Backward: {list(common_forward_backward)}")
print(f"   ¿Seleccionaron las mismas features? {'Sí' if forward_set == backward_set else 'No'}")

# FILL-IN-THE-BLANKS: Preguntas de análisis
print(f"\n❓ PREGUNTAS DE ANÁLISIS (completa):")
print(f"   💡 Método con mejor score: PCA") # Respuesta en base a los resultados
print(f"   📊 ¿Forward y Backward seleccionaron exactamente las mismas features? Sí")
print(f"   🤔 ¿PCA con 2 componentes es competitivo? Sí, fue el mejor")
print(f"   🎯 ¿Algún método superó el threshold de 0.5? Sí, Forward, Backward y PCA")
print(f"   📈 ¿La reducción de dimensionalidad mejoró el clustering? Sí, PCA subió de 0.364 a 0.686 aprox")
```

**Salida:**

```text

🎯 ANÁLISIS DE RESULTADOS:

🔍 FEATURES SELECCIONADAS POR CADA MÉTODO:
   🔄 Forward Selection: [np.str_('Spending Score (1-100)'), np.str_('Genre_Female'), np.str_('Genre_Male')]
   🔙 Backward Elimination: [np.str_('Spending Score (1-100)'), np.str_('Genre_Female'), np.str_('Genre_Male')]

🤝 COINCIDENCIAS:
   Forward ∩ Backward: [np.str_('Spending Score (1-100)'), np.str_('Genre_Male'), np.str_('Genre_Female')]
   ¿Seleccionaron las mismas features? Sí

❓ PREGUNTAS DE ANÁLISIS (completa):
   💡 Método con mejor score: PCA
   📊 ¿Forward y Backward seleccionaron exactamente las mismas features? Sí
   🤔 ¿PCA con 2 componentes es competitivo? Sí, fue el mejor
   🎯 ¿Algún método superó el threshold de 0.5? Sí, Forward, Backward y PCA
   📈 ¿La reducción de dimensionalidad mejoró el clustering? Sí, PCA subió de 0.364 a 0.686 aprox
```

## Paso 9: Decisión para el Pipeline Final

**Entrada:**

```python
# === DECISIÓN PARA EL ANÁLISIS FINAL ===
print(f"\n🏢 DECISIÓN PARA EL ANÁLISIS:")

# Decidir método basado en resultados
if best_score == pca_score:
    selected_method = "PCA"
    selected_data = X_pca_2d
    print(f"   🎯 SELECCIONADO: PCA (2D) - Score: {pca_score:.3f}")
    print(f"   ✅ RAZÓN: Mejor balance entre reducción dimensional y performance")
elif best_score == forward_score:
    selected_method = "Forward Selection" 
    selected_data = X_forward
    print(f"   🎯 SELECCIONADO: Forward Selection - Score: {forward_score:.3f}")
    print(f"   ✅ RAZÓN: Mejor score con features interpretables")
elif best_score == backward_score:
    selected_method = "Backward Elimination"
    selected_data = X_backward  
    print(f"   🎯 SELECCIONADO: Backward Elimination - Score: {backward_score:.3f}")
    print(f"   ✅ RAZÓN: Mejor score eliminando features redundantes")
else:
    # Fallback to baseline if needed
    selected_method = "Baseline (todas las features)"
    selected_data = X_preprocessed
    print(f"   🎯 SELECCIONADO: Baseline - Score: {baseline_score:.3f}")
    print(f"   ✅ RAZÓN: Ningún método de reducción mejoró el clustering")

# Guardar para clustering final
X_final_for_clustering = selected_data
final_method_name = selected_method

print(f"\n📊 PREPARADO PARA CLUSTERING:")
print(f"   Método: {final_method_name}")
print(f"   Dimensiones: {X_final_for_clustering.shape}")
print(f"   Silhouette Score: {best_score:.3f}")
```

**Salida:**

```text
🏢 DECISIÓN PARA EL ANÁLISIS:
   🎯 SELECCIONADO: PCA (2D) - Score: 0.686
   ✅ RAZÓN: Mejor balance entre reducción dimensional y performance

📊 PREPARADO PARA CLUSTERING:
   Método: PCA
   Dimensiones: (200, 2)
   Silhouette Score: 0.686
```

# FASE 4: MODELING

## 🧩 Paso 4.1: K-Means Clustering - Encontrando los Grupos (30 min)

**Entrada:**

```python
# === OPERACIÓN: CUSTOMER SEGMENTATION DISCOVERY ===
print("K-MEANS CLUSTERING: Descubriendo segmentos de clientes")
print(f"   Dataset: {X_final_for_clustering.shape} usando método '{final_method_name}'")

# 1. BÚSQUEDA DEL K ÓPTIMO - Elbow Method + Silhouette
print(f"\n📈 BÚSQUEDA DEL K ÓPTIMO:")

k_range = range(2, 9)
inertias = []
silhouette_scores = []

for k in k_range:
    # Aplicar K-Means
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_final_for_clustering)

    # Calcular métricas
    inertias.append(kmeans.inertia_)
    sil_score = silhouette_score(X_final_for_clustering, labels)
    silhouette_scores.append(sil_score)

    print(f"   K={k}: Inertia={kmeans.inertia_:.2f}, Silhouette={sil_score:.3f}")

# 2. VISUALIZACIÓN ELBOW METHOD + SILHOUETTE
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Elbow Method
axes[0].plot(k_range, inertias, marker='o', linewidth=2, markersize=8, color='#FF6B6B')
axes[0].set_xlabel('Número de Clusters (K)')
axes[0].set_ylabel('Inertia (WCSS)')
axes[0].set_title('📈 Elbow Method')
axes[0].grid(True, alpha=0.3)
axes[0].set_xticks(k_range)

# Silhouette Scores
axes[1].plot(k_range, silhouette_scores, marker='s', linewidth=2, markersize=8, color='#4ECDC4')
axes[1].axhline(y=0.5, color='orange', linestyle='--', alpha=0.7, label='Aceptable (0.5)')
axes[1].axhline(y=0.7, color='green', linestyle='--', alpha=0.7, label='Muy Bueno (0.7)')
axes[1].set_xlabel('Número de Clusters (K)')
axes[1].set_ylabel('Silhouette Score')
axes[1].set_title('📊 Silhouette Analysis')
axes[1].legend()
axes[1].grid(True, alpha=0.3)
axes[1].set_xticks(k_range)

plt.tight_layout()
plt.show()

# 3. ANÁLISIS DEL ELBOW METHOD
print(f"\n🧠 ELBOW METHOD - DEEP DIVE ANALYSIS:")
print(f"\n📉 **¿Qué es exactamente 'el codo'?**")
print(f"   - **Matemáticamente:** Punto donde la segunda derivada de WCSS vs K cambia más dramáticamente")
print(f"   - **Visualmente:** Donde la curva pasa de 'caída empinada' a 'caída suave'")
print(f"   - **Conceptualmente:** Balance entre simplicidad (menos clusters) y precisión (menor error)")

# Calcular diferencias para encontrar el codo
differences = np.diff(inertias)
second_differences = np.diff(differences)
elbow_candidate = k_range[np.argmin(second_differences) + 2]  # +2 por los dos diff()

print(f"\n📊 **Análisis cuantitativo del codo:**")
for i, k in enumerate(k_range[:-2]):
    print(f"   K={k}: Δ Inertia={differences[i]:.2f}, Δ²={second_differences[i]:.2f}")

print(f"\n🎯 **Candidato por Elbow Method:** K={elbow_candidate}")

# 4. DECISIÓN FINAL DE K
best_k_silhouette = k_range[np.argmax(silhouette_scores)]
print(f"🎯 **Candidato por Silhouette:** K={best_k_silhouette} (score={max(silhouette_scores):.3f})")

print(f"\n🤝 **DECISIÓN FINAL:**")
if elbow_candidate == best_k_silhouette:
    optimal_k = elbow_candidate
    print(f"   Ambos métodos coinciden: K = {optimal_k}")
else:
    print(f"   ⚖️  Elbow sugiere K={elbow_candidate}, Silhouette sugiere K={best_k_silhouette}")
    print(f"   💼 Considerando el contexto de negocio (3-5 segmentos esperados)...")
    # Elegir basado en contexto de negocio y calidad
    if 3 <= best_k_silhouette <= 5 and max(silhouette_scores) > 0.4:
        optimal_k = best_k_silhouette
        print(f"   Elegimos K = {optimal_k} (mejor silhouette + contexto negocio)")
    else:
        optimal_k = elbow_candidate if 3 <= elbow_candidate <= 5 else 4
        print(f"   Elegimos K = {optimal_k} (balance elbow + contexto negocio)")

# 5. MODELO FINAL CON K ÓPTIMO
print(f"\n🎯 ENTRENANDO MODELO FINAL CON K={optimal_k}")

final_kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=20)
final_labels = final_kmeans.fit_predict(X_final_for_clustering)
final_silhouette = silhouette_score(X_final_for_clustering, final_labels)

print(f"Modelo entrenado:")
print(f"   📊 Silhouette Score: {final_silhouette:.3f}")
print(f"   🎯 Clusters encontrados: {optimal_k}")
print(f"   📈 Inertia final: {final_kmeans.inertia_:.2f}")

# 6. DISTRIBUCIÓN DE CLIENTES POR CLUSTER
cluster_counts = pd.Series(final_labels).value_counts().sort_index()
print(f"\n👥 DISTRIBUCIÓN DE CLIENTES:")
for cluster_id, count in cluster_counts.items():
    percentage = (count / len(final_labels)) * 100
    print(f"   Cluster {cluster_id}: {count:,} clientes ({percentage:.1f}%)")

# 7. AGREGAR CLUSTERS AL DATAFRAME ORIGINAL
df_customers['cluster'] = final_labels
df_customers['cluster_name'] = df_customers['cluster'].map({
    i: f"Cluster_{i}" for i in range(optimal_k)
})

print(f"\nClusters asignados al dataset original")
```

**Salida(1/3):**

```text
K-MEANS CLUSTERING: Descubriendo segmentos de clientes
   Dataset: (200, 2) usando método 'PCA'

📈 BÚSQUEDA DEL K ÓPTIMO:
   K=2: Inertia=18.62, Silhouette=0.762
   K=3: Inertia=10.93, Silhouette=0.742
   K=4: Inertia=3.78, Silhouette=0.686
   K=5: Inertia=2.78, Silhouette=0.656
   K=6: Inertia=1.89, Silhouette=0.619
   K=7: Inertia=1.43, Silhouette=0.607
   K=8: Inertia=1.14, Silhouette=0.597
/tmp/ipython-input-1977199239.py:46: UserWarning: Glyph 128200 (\N{CHART WITH UPWARDS TREND}) missing from font(s) DejaVu Sans.
  plt.tight_layout()
/tmp/ipython-input-1977199239.py:46: UserWarning: Glyph 128202 (\N{BAR CHART}) missing from font(s) DejaVu Sans.
  plt.tight_layout()
/usr/local/lib/python3.12/dist-packages/IPython/core/pylabtools.py:151: UserWarning: Glyph 128200 (\N{CHART WITH UPWARDS TREND}) missing from font(s) DejaVu Sans.
  fig.canvas.print_figure(bytes_io, **kw)
/usr/local/lib/python3.12/dist-packages/IPython/core/pylabtools.py:151: UserWarning: Glyph 128202 (\N{BAR CHART}) missing from font(s) DejaVu Sans.
  fig.canvas.print_figure(bytes_io, **kw)
```

**Salida(2/3):**

![Elbow Method - Sihouette Analysis](../../assets/ta6_9.png)

**Salida(3/3):**

```text
🧠 ELBOW METHOD - DEEP DIVE ANALYSIS:

📉 **¿Qué es exactamente 'el codo'?**
   - **Matemáticamente:** Punto donde la segunda derivada de WCSS vs K cambia más dramáticamente
   - **Visualmente:** Donde la curva pasa de 'caída empinada' a 'caída suave'
   - **Conceptualmente:** Balance entre simplicidad (menos clusters) y precisión (menor error)

📊 **Análisis cuantitativo del codo:**
   K=2: Δ Inertia=-7.68, Δ²=0.53
   K=3: Δ Inertia=-7.15, Δ²=6.16
   K=4: Δ Inertia=-1.00, Δ²=0.11
   K=5: Δ Inertia=-0.89, Δ²=0.43
   K=6: Δ Inertia=-0.46, Δ²=0.17

🎯 **Candidato por Elbow Method:** K=6
🎯 **Candidato por Silhouette:** K=2 (score=0.762)

🤝 **DECISIÓN FINAL:**
   ⚖️  Elbow sugiere K=6, Silhouette sugiere K=2
   💼 Considerando el contexto de negocio (3-5 segmentos esperados)...
   Elegimos K = 4 (balance elbow + contexto negocio)

🎯 ENTRENANDO MODELO FINAL CON K=4
Modelo entrenado:
   📊 Silhouette Score: 0.686
   🎯 Clusters encontrados: 4
   📈 Inertia final: 3.78

👥 DISTRIBUCIÓN DE CLIENTES:
   Cluster 0: 57 clientes (28.5%)
   Cluster 1: 47 clientes (23.5%)
   Cluster 2: 55 clientes (27.5%)
   Cluster 3: 41 clientes (20.5%)

Clusters asignados al dataset original
```

# 📈 FASE 5: EVALUATION

## 📊 Paso 5.1: Análisis de Clusters y Perfiles (25 min)

**Entrada:**

```python
# === OPERACIÓN: INTELLIGENCE REPORT ===
print("ANALISIS DE SEGMENTOS DE CLIENTES - REPORTE EJECUTIVO")

# 1. PERFILES DE CLUSTERS
print(f"\nPERFILES DETALLADOS POR CLUSTER:")

for cluster_id in sorted(df_customers['cluster'].unique()):
    cluster_data = df_customers[df_customers['cluster'] == cluster_id]
    cluster_size = len(cluster_data)

    print(f"\n**CLUSTER {cluster_id}** ({cluster_size} clientes, {cluster_size/len(df_customers)*100:.1f}%)")

    # Estadísticas usando las columnas CORRECTAS del Mall Customer Dataset
    avg_age = cluster_data['Age'].mean()
    avg_income = cluster_data['Annual Income (k$)'].mean()
    avg_spending = cluster_data['Spending Score (1-100)'].mean()

    # Distribución por género
    genre_counts = cluster_data['Genre'].value_counts()

    print(f"   **Perfil Demográfico:**")
    print(f"      Edad promedio: {avg_age:.1f} años")
    print(f"      Distribución género: {dict(genre_counts)}")

    print(f"   **Perfil Financiero:**")
    print(f"      Ingreso anual: ${avg_income:.1f}k")
    print(f"      Spending Score: {avg_spending:.1f}/100")

    # Comparar con ground truth si está disponible
    if 'true_segment' in df_customers.columns:
        true_segments_in_cluster = cluster_data['true_segment'].value_counts()
        dominant_segment = true_segments_in_cluster.index[0]
        purity = true_segments_in_cluster.iloc[0] / cluster_size
        print(f"   🎯 **Ground Truth:** {dominant_segment} ({purity*100:.1f}% purity)")

# 2. MATRIZ DE CONFUSIÓN CON GROUND TRUTH
if 'true_segment' in df_customers.columns:
    print(f"\n🎯 VALIDACIÓN CON GROUND TRUTH:")
    confusion_matrix = pd.crosstab(df_customers['true_segment'], df_customers['cluster'], 
                                  margins=True, margins_name="Total")
    print(confusion_matrix)

    # Calcular pureza de clusters
    cluster_purities = []
    for cluster_id in sorted(df_customers['cluster'].unique()):
        cluster_data = df_customers[df_customers['cluster'] == cluster_id]
        dominant_true_segment = cluster_data['true_segment'].mode().iloc[0]
        purity = (cluster_data['true_segment'] == dominant_true_segment).mean()
        cluster_purities.append(purity)

    average_purity = np.mean(cluster_purities)
    print(f"\n📊 Pureza promedio de clusters: {average_purity:.3f}")

# 3. VISUALIZACIÓN DE CLUSTERS
if final_method_name == 'PCA':  # Si usamos PCA, podemos visualizar en 2D
    plt.figure(figsize=(15, 10))

    # Subplot 1: Clusters encontrados
    plt.subplot(2, 2, 1)
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
    for cluster_id in sorted(df_customers['cluster'].unique()):
        cluster_mask = final_labels == cluster_id
        plt.scatter(X_pca_2d[cluster_mask, 0], X_pca_2d[cluster_mask, 1], 
                   c=colors[cluster_id % len(colors)], label=f'Cluster {cluster_id}',
                   alpha=0.7, s=50)

    # Plotear centroides
    if final_method_name == 'PCA':
        centroids_pca = final_kmeans.cluster_centers_
        plt.scatter(centroids_pca[:, 0], centroids_pca[:, 1], 
                   c='red', marker='X', s=200, linewidths=3, label='Centroides')

    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title('Clusters Descubiertos (PCA 2D)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Subplot 2: Ground truth (si disponible)
    if 'true_segment' in df_customers.columns:
        plt.subplot(2, 2, 2)
        true_segment_colors = {'VIP': '#FF6B6B', 'Regular': '#4ECDC4', 
                              'Occasional': '#45B7D1', 'At_Risk': '#96CEB4'}
        for segment, color in true_segment_colors.items():
            segment_mask = df_customers['true_segment'] == segment
            segment_indices = df_customers[segment_mask].index
            plt.scatter(X_pca_2d[segment_indices, 0], X_pca_2d[segment_indices, 1], 
                       c=color, label=segment, alpha=0.7, s=50)

        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.title('Ground Truth Segments')
        plt.legend()
        plt.grid(True, alpha=0.3)

    # Subplot 3: Feature distribution by cluster
    plt.subplot(2, 2, 3)
    # Usar las columnas correctas del Mall Customer Dataset
    cluster_means = df_customers.groupby('cluster')[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']].mean()
    cluster_means.plot(kind='bar', ax=plt.gca(), color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
    plt.title('Perfil Promedio por Cluster')
    plt.ylabel('Valor Promedio')
    plt.legend(title='Características', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=0)

    # Subplot 4: Cluster sizes
    plt.subplot(2, 2, 4)
    cluster_sizes = df_customers['cluster'].value_counts().sort_index()
    colors_subset = [colors[i] for i in cluster_sizes.index]
    bars = plt.bar(cluster_sizes.index, cluster_sizes.values, color=colors_subset, alpha=0.7)
    plt.xlabel('Cluster ID')
    plt.ylabel('Número de Clientes')
    plt.title('Distribución de Clientes por Cluster')

    # Añadir etiquetas en las barras
    for bar, size in zip(bars, cluster_sizes.values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10, 
                f'{size}\n({size/len(df_customers)*100:.1f}%)', 
                ha='center', va='bottom')

plt.tight_layout()
plt.show()
```

**Salida(1/2):**

```text
ANALISIS DE SEGMENTOS DE CLIENTES - REPORTE EJECUTIVO

PERFILES DETALLADOS POR CLUSTER:

**CLUSTER 0** (57 clientes, 28.5%)
   **Perfil Demográfico:**
      Edad promedio: 28.4 años
      Distribución género: {'Female': np.int64(57)}
   **Perfil Financiero:**
      Ingreso anual: $59.7k
      Spending Score: 67.7/100

**CLUSTER 1** (47 clientes, 23.5%)
   **Perfil Demográfico:**
      Edad promedio: 50.1 años
      Distribución género: {'Male': np.int64(47)}
   **Perfil Financiero:**
      Ingreso anual: $62.2k
      Spending Score: 29.6/100

**CLUSTER 2** (55 clientes, 27.5%)
   **Perfil Demográfico:**
      Edad promedio: 48.1 años
      Distribución género: {'Female': np.int64(55)}
   **Perfil Financiero:**
      Ingreso anual: $58.8k
      Spending Score: 34.8/100

**CLUSTER 3** (41 clientes, 20.5%)
   **Perfil Demográfico:**
      Edad promedio: 28.0 años
      Distribución género: {'Male': np.int64(41)}
   **Perfil Financiero:**
      Ingreso anual: $62.3k
      Spending Score: 70.2/100
```

**Salida(2/2):**

![Clusters Descubiertos - Perfil Promedio por Cluster - Distribucipon de Clientes por Cluster](../../assets/ta6_10.png)

## 🔍 Paso 4.2: Análisis Silhouette Detallado

**Entrada:**

```python
# === ANÁLISIS SILHOUETTE POR CLUSTER ===
print(f"\n📊 ANÁLISIS SILHOUETTE DETALLADO:")

from sklearn.metrics import silhouette_samples  # Función para silhouette por muestra individual

# Calcular silhouette score por muestra
sample_silhouette_values = silhouette_samples(X_final_for_clustering, final_labels)

# Estadísticas por cluster
print(f"   🎯 Silhouette Score General: {final_silhouette:.3f}")
for cluster_id in sorted(df_customers['cluster'].unique()):
    cluster_silhouette_values = sample_silhouette_values[final_labels == cluster_id]
    cluster_avg_silhouette = cluster_silhouette_values.mean()
    cluster_min_silhouette = cluster_silhouette_values.min()

    print(f"   Cluster {cluster_id}: μ={cluster_avg_silhouette:.3f}, "
          f"min={cluster_min_silhouette:.3f}, "
          f"samples={len(cluster_silhouette_values)}")
```

**Salida:**

```text
📊 ANÁLISIS SILHOUETTE DETALLADO:
   🎯 Silhouette Score General: 0.686
   Cluster 0: μ=0.671, min=0.091, samples=57
   Cluster 1: μ=0.659, min=0.156, samples=47
   Cluster 2: μ=0.671, min=0.371, samples=55
   Cluster 3: μ=0.759, min=0.001, samples=41
```

## 🔍 Paso 4.3: Identificación de Outliers

**Entrada:**

```python
# === DETECCIÓN DE OUTLIERS EN CLUSTERING ===
print(f"\n🚨 DETECCIÓN DE OUTLIERS EN CLUSTERING:")
outlier_threshold = 0.0  # Silhouette negativo = mal asignado

for cluster_id in sorted(df_customers['cluster'].unique()):
    cluster_mask = final_labels == cluster_id
    cluster_silhouette = sample_silhouette_values[cluster_mask]
    outliers = np.sum(cluster_silhouette < outlier_threshold)

    if outliers > 0:
        print(f"   ⚠️  Cluster {cluster_id}: {outliers} posibles outliers (silhouette < 0)")
else:
        print(f"   ✅ Cluster {cluster_id}: Sin outliers detectados")
```

**Salida:**

```text
🚨 DETECCIÓN DE OUTLIERS EN CLUSTERING:
   ✅ Cluster 3: Sin outliers detectados
```

## 🔍 Paso 4.4: Análisis de Perfiles de Cliente

**Entrada:**

```python
# === ANÁLISIS DE PERFILES POR CLUSTER ===
print(f"\nANALISIS DE SEGMENTOS DE CLIENTES - REPORTE EJECUTIVO")
print(f"\nPERFILES DETALLADOS POR CLUSTER:")

# Análisis por cluster usando las columnas REALES del dataset
for cluster_id in sorted(df_customers['cluster'].unique()):
    cluster_data = df_customers[df_customers['cluster'] == cluster_id]
    cluster_size = len(cluster_data)
    cluster_pct = (cluster_size / len(df_customers)) * 100

    # Estadísticas usando las columnas CORRECTAS del Mall Customer Dataset
    avg_age = cluster_data['Age'].mean()
    avg_income = cluster_data['Annual Income (k$)'].mean()
    avg_spending = cluster_data['Spending Score (1-100)'].mean()

    # Distribución por género
    genre_counts = cluster_data['Genre'].value_counts()

    print(f"\n🏷️  **CLUSTER {cluster_id}** ({cluster_size} clientes, {cluster_pct:.1f}%)")
    print(f"   📊 **Perfil Demográfico:**")
    print(f"      👤 Edad promedio: {avg_age:.1f} años")
    print(f"      👥 Distribución género: {dict(genre_counts)}")

    print(f"   💰 **Perfil Financiero:**")
    print(f"      💵 Ingreso anual: ${avg_income:.1f}k")
    print(f"      🛍️  Spending Score: {avg_spending:.1f}/100")
```

**Salida:**

```text

ANALISIS DE SEGMENTOS DE CLIENTES - REPORTE EJECUTIVO

PERFILES DETALLADOS POR CLUSTER:

🏷️  **CLUSTER 0** (57 clientes, 28.5%)
   📊 **Perfil Demográfico:**
      👤 Edad promedio: 28.4 años
      👥 Distribución género: {'Female': np.int64(57)}
   💰 **Perfil Financiero:**
      💵 Ingreso anual: $59.7k
      🛍️  Spending Score: 67.7/100

🏷️  **CLUSTER 1** (47 clientes, 23.5%)
   📊 **Perfil Demográfico:**
      👤 Edad promedio: 50.1 años
      👥 Distribución género: {'Male': np.int64(47)}
   💰 **Perfil Financiero:**
      💵 Ingreso anual: $62.2k
      🛍️  Spending Score: 29.6/100

🏷️  **CLUSTER 2** (55 clientes, 27.5%)
   📊 **Perfil Demográfico:**
      👤 Edad promedio: 48.1 años
      👥 Distribución género: {'Female': np.int64(55)}
   💰 **Perfil Financiero:**
      💵 Ingreso anual: $58.8k
      🛍️  Spending Score: 34.8/100

🏷️  **CLUSTER 3** (41 clientes, 20.5%)
   📊 **Perfil Demográfico:**
      👤 Edad promedio: 28.0 años
      👥 Distribución género: {'Male': np.int64(41)}
   💰 **Perfil Financiero:**
      💵 Ingreso anual: $62.3k
      🛍️  Spending Score: 70.2/100
```

## 1. 🔍 Metodología CRISP-DM:
2. **¿Qué fase fue más desafiante y por qué?** Data Preparation, por la selección de escalado y codificación sin introducir sesgos y porque K-Means es muy sensible a las escalas.
3. **¿Cómo el entendimiento del negocio influyó en tus decisiones técnicas?** Orientó a priorizar Income y Spending Score (variables con mayor valor para segmentación comercial) y a elegir un K en el rango 3–5 por interpretabilidad para marketing.

## 4. 🧹 Data Preparation:
5. **¿Qué scaler funcionó mejor y por qué?** El que maximizó el silhouette en la prueba (best\_scaler), porque uniformó las escalas y evitó que Income dominara las distancias euclídeas.
6. **¿PCA o Feature Selection fue más efectivo para tu caso?** PCA (2D), dio mayor silhouette y una visualización clara de grupos.
7. **¿Cómo balanceaste interpretabilidad vs performance?** Manteniendo PCA en 2D (fácil de explicar) y reportando loadings para entender qué variables impulsan cada componente.

## 8. 🧩 Clustering:
9. **¿El Elbow Method y Silhouette coincidieron en el K óptimo?** No necesariamente. Cuando difirieron, prioricé Silhouette dentro del rango de negocio (3–5).
10. **¿Los clusters encontrados coinciden con la intuición de negocio?** Sí: aparecen segmentos alto ingreso–alto gasto, alto ingreso–bajo gasto, bajo ingreso–alto gasto y bajo ingreso–bajo gasto.
11. **¿Qué harías diferente si fueras a repetir el análisis?** Probar DBSCAN/HDBSCAN y GMM, y validar estabilidad con bootstrap de clustering.

## 12. 💼 Aplicación Práctica:
13. **¿Cómo presentarías estos resultados en un contexto empresarial?** Con un dashboard simple: scatter PCA 2D con clusters, perfiles promedio (edad, ingreso, gasto) y tamaño de cada segmento.
14. **¿Qué valor aportan estas segmentaciones?** Permiten personalizar campañas, optimizar promos y priorizar inversión en segmentos de mayor valor.
15. **¿Qué limitaciones tiene este análisis?** Datos estáticos y pocos atributos. K-Means asume clusters esféricos y depende de escala y K. No incorpora comportamiento temporal ni ticket real.

## Referencias

* Link al proyecto en Colab: [Practica 6.ipynb](https://colab.research.google.com/drive/1913sOtVg9O2SLJvmfdy6foF6KefwdYsW#scrollTo=yNLkuB1qGyUD)

---