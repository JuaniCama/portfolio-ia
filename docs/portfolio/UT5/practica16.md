# Recorrido por los labs prácticos de Google Cloud


## Contexto

Este lab introductorio de Google Cloud Skills Boost (“Recorrido por los labs prácticos de Google Cloud”, GSP282) tuvo como objetivo conocer el entorno de labs de Qwiklabs y dar los primeros pasos en la consola de Google Cloud.

Durante 45 minutos trabajé con un proyecto temporal de Google Cloud, utilizando credenciales de estudiante generadas para el lab. Desde allí exploré la consola, la estructura de proyectos, los roles de IAM y la habilitación de APIs.

## Objetivos

- Acceder a la consola de Google Cloud usando credenciales temporales del lab.
- Identificar los componentes principales de la interfaz de un lab (tiempo, créditos, tracking, botón Start/Finish lab).
- Explorar proyectos de Google Cloud y entender la diferencia entre el proyecto del lab y el proyecto compartido **Qwiklabs Resources**.
- Revisar y modificar roles básicos de IAM (viewer, editor, owner) para cuentas de estudiante.
- Habilitar una API específica (Dialogflow) desde la biblioteca de APIs.
- Completar correctamente los “Check my progress” para registrar la finalización del lab.

## Actividades (con tiempos estimados)

- Lectura rápida de la descripción del lab y preguntas iniciales de opción múltiple — 5 min  
- Tarea 1: acceso a la consola de Cloud con credenciales de estudiante — 5 min  
- Tarea 2: exploración de proyectos y verificación de **Qwiklabs Resources** — 5 min  
- Tarea 3: revisión de IAM, asignación de rol *Viewer* a una segunda cuenta de estudiante — 10 min  
- Tarea 4: habilitación de la API de Dialogflow y revisión en la documentación — 10 min  
- Tarea 5: cierre del lab, envío y registro de la actividad — 5 min  
- Tiempo restante: resolución de dudas con la interfaz y repaso conceptual — ~5 min  

## Desarrollo

### Acceso al entorno del lab

1. En la página de Google Cloud Skills Boost inicié el lab con el botón **Start lab**.  
2. El sistema aprovisionó un proyecto temporal de Google Cloud y mostró, en el panel izquierdo, las credenciales:
   - **Username** de estudiante (`student-...@qwiklabs.net`)
   - **Password**
   - **Project ID** (`qwiklabs-gcp-...`)
3. Desde ese mismo panel usé el botón **Open Google Cloud Console** para abrir la consola ya vinculada al proyecto del lab.  
4. Inicié sesión con el usuario de estudiante, acepté los términos de servicio y confirmé el acceso a la consola.

Un error inicial fue intentar entrar desde la página genérica de Google Cloud (“Comenzar gratis”), lo que lleva a la cuenta personal. La corrección fue entrar siempre desde el botón del lab, que abre la consola con las credenciales temporales.

### Exploración de proyectos

En la barra superior de la consola, desplegué el selector de proyecto y usé el filtro **All**. Allí confirmé:

- El proyecto temporal del lab (donde debía trabajar).
- El proyecto compartido **Qwiklabs Resources**, que es de solo lectura y se usa como repositorio de datos/imágenes para otros labs.

La consigna pedía verificar este proyecto pero sin cambiar el proyecto activo, por lo que cerré el diálogo con **Cancel**.

### Revisión y modificación de roles de IAM

1. Desde el menú de navegación abrí **IAM & Admin → IAM**.  
2. Localicé mi usuario de estudiante (`student-04-0a1f794cabba@qwiklabs.net`) con rol **Editor**, lo que coincide con la descripción del lab (permisos de lectura y de modificación de recursos, pero sin administración completa de IAM ni facturación).  
3. Para la parte práctica, el lab pedía otorgar el rol **Viewer** a una segunda cuenta de estudiante:
   - Abrí **Grant access / Otorgar acceso**.
   - En **Add principals / Agregar principales** ingresé el email indicado en la consigna (`student-04-e15f5849cd93@qwiklabs.net`).
   - En **Select a role** seleccioné **Viewer**.
   - Guardé los cambios y verifiqué que la nueva principal apareciera en la lista con el rol correcto.  
4. De vuelta en la página del lab, usé **Check my progress** para que el sistema de tracking registrara la tarea como completada.

Durante esta parte apareció inicialmente un error al intentar usar “User 2” como texto literal en lugar del email completo; se solucionó leyendo con más detalle la consigna, donde figuraba la dirección exacta a utilizar.

### Habilitación de la API de Dialogflow

1. Abrí **APIs & Services → Library** desde el menú de navegación.  
2. Utilicé la barra de búsqueda para encontrar **Dialogflow API**.  
3. En la página de detalle de la API, hice clic en **Enable** para habilitarla en el proyecto del lab.  
4. Verifiqué que la API figurara como “Enabled” y probé el enlace **Try this API**, que abre la documentación y el explorador de APIs en otra pestaña.  
5. Regresé a la consola y luego al lab para marcar el objetivo como logrado con **Check my progress**.

### Cierre del lab

Al completar todas las tareas, regresé a la página del lab y usé el botón **Finish lab**. Al confirmar, el entorno temporal se eliminó (proyecto, credenciales y recursos), y la consola cerró la sesión automáticamente. Finalmente califiqué la experiencia del lab.

## Evidencias

- No pude sacar evidencias del lab ya que una vez finalizado no tenia más acceso a lo realizado, y por cuestiones de tiempo, no pude realizarlo nuevamente

## Reflexión

- **Qué aprendí**
  - Cómo funciona el entorno de labs de Google Cloud Skills Boost, incluyendo el uso de proyectos temporales y credenciales efímeras.
  - La estructura básica de un proyecto de Google Cloud (nombre, ID, número) y cómo se selecciona el proyecto activo desde la consola.
  - La diferencia entre roles básicos de IAM (Viewer, Editor, Owner) y su impacto sobre las acciones que se pueden realizar en un proyecto.
  - El flujo para habilitar APIs específicas en un proyecto (en este caso, Dialogflow) y dónde consultar su documentación.

- **Qué mejoraría**
  - Leer con más calma los pasos de la consigna para evitar errores como intentar usar “User 2” en lugar del email de la segunda cuenta.
  - Organizar desde el inicio las capturas de pantalla para documentar el lab de forma más sistemática en el portafolio.

- **Próximos pasos**
  - Realizar labs más avanzados recomendados al final de la actividad:
    - “Cómo crear una máquina virtual”.
    - “Cómo comenzar a utilizar Cloud Shell y gcloud”.
  - Profundizar en el uso de IAM en escenarios más complejos (roles personalizados, políticas a nivel de recurso).
  - Explorar con más detalle la API de Dialogflow en un proyecto propio, integrándola con aplicaciones web o bots.

## Referencias

- Google Cloud Skills Boost — *Recorrido por los labs prácticos de Google Cloud (GSP282)*.  
- Documentación de Google Cloud:
  - **Consola de Google Cloud** – descripción general y navegación.
  - **Cloud Identity and Access Management (IAM)** – roles básicos y administración de permisos.
  - **Dialogflow API** – descripción y guía de uso.
  - Consigna: [`Recorrido por los labs prácticos de Google Cloud`](https://www.skills.google/focuses/2794?catalog_rank=%7B%22rank%22%3A3%2C%22num_filters%22%3A2%2C%22has_search%22%3Atrue%7D&parent=catalog&search_id=60924676)

