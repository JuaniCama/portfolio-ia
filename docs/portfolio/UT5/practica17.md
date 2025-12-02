# Vertex AI Pipelines: Qwik Start


## Contexto

Este lab de Google Cloud Skills Boost introduce el uso de **Vertex AI Pipelines** para automatizar y reproducir flujos de trabajo de aprendizaje automático (AA) sobre Google Cloud.

Se trabaja en un entorno de Qwiklabs con **credenciales temporales** y un proyecto de Google Cloud aprovisionado para el lab. Desde una instancia de **Vertex AI Workbench** (JupyterLab), se construyen y ejecutan dos canalizaciones:

1. Una pipeline introductoria en tres pasos que produce una frase con un emoji.
2. Una pipeline de **AutoML Tabular** de extremo a extremo sobre el dataset *Dry Beans*, que crea el dataset, entrena el modelo, evalúa métricas y decide si desplegar el modelo en un endpoint de Vertex AI.

## Objetivos

- Acceder a Vertex AI Workbench y trabajar desde un notebook de Python en JupyterLab.
- Configurar el entorno de trabajo para Vertex AI Pipelines:
  - Instalar `google-cloud-aiplatform`, `kfp` y `google_cloud_pipeline_components`.
  - Definir `PROJECT_ID`, `BUCKET_NAME`, `REGION` y `PIPELINE_ROOT`.
- Diseñar, compilar y ejecutar una **canalización simple** con tres componentes personalizados (texto + emoji).
- Diseñar, compilar y ejecutar una **canalización de AA de extremo a extremo** con AutoML Tabular, evaluación de métricas y despliegue condicional del modelo.
- Explorar los artefactos, el linaje y los metadatos de ejecución de las canalizaciones en Vertex AI.

## Actividades (con tiempos estimados)

- Lectura de la consigna y preparación del entorno de Qwiklabs — 10 min  
- Tarea 1: apertura de Workbench y JupyterLab — 10 min  
- Tarea 2: instalación de paquetes, configuración de proyecto/bucket y constantes — 20 min  
- Tarea 3: definición, compilación y ejecución de la pipeline de emoji — 20 min  
- Tarea 4: definición, compilación y lanzamiento de la pipeline de AutoML Tabular — 25 min  
- Exploración de artefactos, linaje y métricas + cierre del lab — 5 min  

## Desarrollo

### Acceso al entorno y apertura de JupyterLab

Tras iniciar el lab y obtener las credenciales temporales:

- Se accedió a la consola de Google Cloud con el usuario `student-...@qwiklabs.net`.
- Desde el menú de navegación se ingresó a **Vertex AI → Workbench**, donde ya existía una instancia de notebook.
- Se abrió **JupyterLab** mediante el botón **Open JupyterLab**.  
  En caso de problemas, el lab prevé un *reset* de la instancia y nuevo intento de apertura.

En JupyterLab se creó un notebook de Python 3 y se le asignó un nombre significativo (por ejemplo, `vertex-pipelines-intro.ipynb`).

### Configuración de Vertex AI Pipelines

En el notebook se realizó la configuración necesaria:

1. **Instalación de paquetes**

   Usando `%pip` y la bandera `--user`:

   - `google-cloud-aiplatform==1.59.0`
   - `kfp` y `google-cloud-pipeline-components==0.1.1`
   - Ajustes de versiones de `shapely`, `pygeos` y `geopandas` para evitar incompatibilidades.
   - Reinstalación de `google-cloud-pipeline-components`.

   Después se ejecutó un pequeño bloque que reinicia el kernel para asegurar que los paquetes recién instalados estén disponibles. Finalmente, se verificaron las versiones de `kfp` y `google_cloud_pipeline_components` mediante comandos `!python3 -c ...`.

2. **Configuración de proyecto y bucket**

   - Se obtuvo el `PROJECT_ID` con `gcloud config list --format 'value(core.project)'`.
   - Se definió `BUCKET_NAME` como `gs://<PROJECT_ID>-labconfig-bucket`.
   - Se ajustó la variable de entorno `PATH` para incluir el directorio local de `pip`.

3. **Importación de librerías y constantes**

   - Se importaron `kfp`, `dsl`, `compiler`, `AIPlatformClient`, `aiplatform` y `google_cloud_pipeline_components.aiplatform` (alias `gcc_aip`), entre otras.
   - Se definió la región (`REGION`, por ejemplo `us-central1`) y el `PIPELINE_ROOT`:

     ```text
     PIPELINE_ROOT = gs://<PROJECT_ID>-labconfig-bucket/pipeline_root/
     ```

   Este directorio en Cloud Storage es el lugar donde se guardan los artefactos generados por las canalizaciones (modelos, datasets, métricas, endpoints, etc.).

### Primera canalización: “hello world” con emoji

El objetivo fue construir una canalización sencilla en tres componentes para entender el flujo de KFP:

1. **Componentes personalizados**

   - `product_name(text: str) -> str`  
     Retorna el texto de entrada. Se definió con el decorador `@component`, especificando la imagen base (`python:3.9`) y el archivo YAML de salida.

   - `emoji(text: str) -> NamedTuple("Outputs", [("emoji_text", str), ("emoji", str)])`  
     Usa la librería `emoji` (instalada vía `packages_to_install`) para convertir una descripción textual, como `"sparkles"`, en el emoji ✨, y devuelve tanto el texto como el emoji.

   - `build_sentence(product: str, emoji: str, emojitext: str) -> str`  
     Combina los resultados anteriores para construir una frase final del estilo:

     ```text
     "<product> is <emoji|emojitext>"
     ```

2. **Definición de la pipeline**

   Se definió la función `intro_pipeline` decorada con `@dsl.pipeline`, utilizando `PIPELINE_ROOT` como ruta raíz y parámetros por defecto:

   - `text = "Vertex AI Pipelines"`
   - `emoji_str = "sparkles"`

   Dentro de la pipeline:

   - `product_task = product_name(text)`
   - `emoji_task = emoji(emoji_str)`
   - `build_sentence` consume las salidas de los dos componentes anteriores:
     - `product_task.output`
     - `emoji_task.outputs["emoji"]`
     - `emoji_task.outputs["emoji_text"]`

3. **Compilación y ejecución**

   - Se compiló la pipeline a `intro_pipeline_job.json` mediante `compiler.Compiler().compile(...)`.
   - Se creó un `AIPlatformClient` con el `PROJECT_ID` y `REGION`.
   - Se ejecutó `create_run_from_job_spec`, lo que lanzó una ejecución en Vertex AI Pipelines.

En la consola de Vertex AI se pudo ver el grafo de la pipeline con tres nodos, y tras unos minutos la ejecución terminó en estado **Succeeded**, mostrando como salida final la frase con el emoji. El botón de “Revisar mi progreso” del lab validó esta tarea.

### Canalización de AA de extremo a extremo con AutoML (Dry Beans)

La segunda parte del lab consistió en diseñar una canalización más compleja que:

- Usa el dataset de *Dry Beans* almacenado en BigQuery.
- Crea un dataset tabular en Vertex AI.
- Entrena un modelo de clasificación con AutoML Tabular.
- Evalúa las métricas del modelo.
- Decide, de forma **condicional**, si desplegar el modelo.
- Implementa el modelo en un endpoint de predicción.

1. **Componente personalizado de evaluación**

   Se definió `classif_model_eval_metrics` como componente personalizado que:

   - Usa `google-cloud-aiplatform` para consultar las evaluaciones del modelo entrenado.
   - Extrae métricas de clasificación (ROC, matriz de confusión, otros indicadores) y las registra en los artefactos `metrics` y `metricsc` para que aparezcan en la UI de Vertex AI Pipelines.
   - Lee un JSON con umbrales (`thresholds_dict_str`, por defecto `{"auRoc": 0.95}`) y compara la métrica `auRoc` contra ese valor.
   - Devuelve un string `dep_decision` igual a `"true"` si el modelo supera el umbral, o `"false"` en caso contrario.

2. **Pipeline de entrenamiento y despliegue**

   Se definió la pipeline `automl-tab-beans-training-v2` con parámetros:

   - `bq_source` = `"bq://aju-dev-demos.beans.beans1"`  
   - `display_name` = `DISPLAY_NAME` (nombre único con timestamp).
   - `project` = `PROJECT_ID`.
   - `gcp_region` y `api_endpoint` ajustados a la región usada (por ejemplo `us-central1`).
   - `thresholds_dict_str` = `'{"auRoc": 0.95}'`.

   Componentes utilizados:

   - `TabularDatasetCreateOp` (predefinido)  
     Crea un dataset tabular en Vertex AI a partir de la tabla de BigQuery.

   - `AutoMLTabularTrainingJobRunOp` (predefinido)  
     Lanza un trabajo de AutoML Tabular de clasificación con:
     - Transformaciones de columnas numéricas y categóricas.
     - `target_column = "Class"`.
     - Presupuesto de entrenamiento (`budget_milli_node_hours`).

   - `classif_model_eval_metrics` (personalizado)  
     Recibe el modelo entrenado y produce el indicador `dep_decision`.

   - `ModelDeployOp` (predefinido)  
     Despliega el modelo en un endpoint de predicción solo si la condición:

     ```text
     model_eval_task.outputs["dep_decision"] == "true"
     ```

     se cumple, gracias al uso de `dsl.Condition`.

3. **Compilación, ejecución y seguimiento**

   - Se compiló la pipeline en `tab_classif_pipeline.json`.
   - Se lanzó la ejecución con `create_run_from_job_spec`, indicando `PIPELINE_ROOT` y parámetros (`project`, `display_name`).

La ejecución de la pipeline apareció en Vertex AI Pipelines, mostrando la creación del dataset, el entrenamiento con AutoML y la etapa de evaluación. El entrenamiento completo toma aproximadamente una hora, por lo que el lab solo exige que el trabajo de entrenamiento **haya comenzado**; no es necesario esperar al final para obtener el crédito en “Revisar mi progreso”.

Además, se exploraron:

- Los **artefactos** generados (dataset, modelo, endpoint, métricas).
- El **linaje** asociado a los artefactos (por ejemplo, desde el dataset hasta los modelos que lo usan).
- Opcionalmente, se utilizó `aiplatform.get_pipeline_df("automl-tab-beans-training-v2")` para obtener un `DataFrame` de Pandas con los metadatos de varias ejecuciones y comparar métricas entre runs.


## Reflexión

- **Qué aprendí**
  - La motivación de usar **canalizaciones de AA**: separar pasos en contenedores reutilizables, permitir reproducibilidad, trazabilidad y colaboración (MLOps).
  - Cómo diseñar componentes personalizados con KFP (`@component`) y cómo encadenar sus salidas dentro de una pipeline (`@dsl.pipeline`).
  - El uso de **componentes predefinidos** de `google_cloud_pipeline_components` para interactuar con servicios de Vertex AI (datasets, AutoML, deploy).
  - Cómo integrar lógica de negocio en la pipeline (umbral de `auRoc` para decidir el despliegue).
  - Cómo inspeccionar artefactos, métricas y linaje desde Vertex AI Pipelines.

- **Qué mejoraría**
  - Automatizar parte de la configuración (por ejemplo, derivar `REGION` y `api_endpoint` a partir de una sola variable para evitar errores de placeholder).
  - Enriquecer el componente de evaluación con más métricas específicas (precisión, recall por clase) y reglas de decisión más completas.
  - Añadir tests unitarios ligeros para componentes clave antes de usarlos en la pipeline.

- **Próximos pasos**
  - Replicar una versión simplificada de esta pipeline en un proyecto propio de Google Cloud, usando un dataset diferente.
  - Explorar la integración de **Cloud Scheduler** para ejecutar la pipeline de forma periódica o gatillada por nuevos datos.
  - Investigar cómo versionar y administrar plantillas de pipeline en repositorios de código (Git) y cómo integrarlas en un flujo de CI/CD.

## Referencias

- Lab de Google Cloud Skills Boost: **Vertex AI Pipelines: Qwik Start (GSP965)**.  
- Documentación oficial de:
  - [Vertex AI Pipelines](https://cloud.google.com/vertex-ai/docs/pipelines)
  - [Kubeflow Pipelines SDK (KFP)](https://www.kubeflow.org/docs/components/pipelines/)
  - [google-cloud-pipeline-components](https://cloud.google.com/vertex-ai/docs/pipelines/google-cloud-pipeline-components)
- Artículo del dataset: *Dry Beans* (KOKLU, M. y OZKAN, I.A., 2020).
