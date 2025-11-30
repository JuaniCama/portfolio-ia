# Prompting y Parametros de LLM con LangChain (UT4-14)

## 1) Resumen de práctica
- Objetivo: probar llamadas a LLM via `langchain-openai` (modelo `gpt-5-mini`) y observar el efecto de hiperparametros de decodificacion (`temperature`, `max_tokens`, `top_p`).
- Metodologia: prompts simples (definiciones, tuit, bullets) ejecutados en loops con distintas temperaturas; un ejemplo de haiku y un set de "combos" de parametros.
- Hallazgos: con `temperature=0` las respuestas son casi deterministas; al subir a `0.5` y `0.9` aumentan variacion y creatividad sin perder estructura; `max_tokens` controla longitud de manera efectiva.
- Incidentes: advertencias de autenticacion LangSmith (401) y un `UnicodeEncodeError` al imprimir en entorno local; no afectan la logica central pero sugieren revisar config/regional.

## 2) Setup
- Instalacion (Colab): `langchain`, `langchain-openai`, `langsmith` y extras (`faiss-cpu`, `chromadb`, `tavily-python`, `duckduckgo-search`, `langchain-text-splitters`).
- Modelo usado: `ChatOpenAI(model="gpt-5-mini", temperature=<var>)`.
- Parametros variando: `temperature` en `{0.0, 0.5, 0.9}`; en combos se fijan tambien `max_tokens=80`, `top_p=1.0`.

## 3) Ejercicios de prompting
- Definicion de "Transformer" (temp=0): respuesta precisa sobre self-attention, paralelismo y dependencia a largo plazo.
- Identidad del modelo: indica ser "ChatGPT (modelo GPT-4), conocimiento hasta junio 2024".
- Haiku (temp=0): produce terceto coherente sobre evaluacion de modelos.
- Variacion por temperatura (dos prompts: tuit corto y 3 bullets de ventajas de Transformers):
  - Temp 0.0: salidas concisas, casi identicas entre runs; estructura estricta.
  - Temp 0.5: mantiene forma pero introduce sinonimos y cambios leves en orden.
  - Temp 0.9: mas creatividad y adjetivos, sin romper el formato pedido.
- Combo de parametros (temp 0.0, max_tokens 80, top_p 1.0): tuit celebratorio y bullets tecnicos; confirma control de longitud y estilo.

## 4) Observaciones y issues
- Warnings recurrentes: `LangSmithAuthError 401 Unauthorized` al intentar loguear corridas; se resolvian ignorando el logging (no afecto outputs).
- Error final: `UnicodeEncodeError: 'charmap'` al imprimir en entorno Windows con cp1252; en Colab no se reproduce. Mitigacion: forzar `encoding='utf-8'` en stdout o sanitizar caracteres no ASCII.

## 5) Conclusiones y siguientes pasos
- El pipeline `ChatOpenAI` responde bien a prompts breves y se observa claramente el efecto de `temperature` en creatividad vs determinismo.
- Para un flujo productivo: configurar credenciales LangSmith o desactivar tracking; estandarizar encoding UTF-8 para evitar fallos locales.
- Sugerencias de extension: probar `top_p` < 1 para mayor control, agregar evaluacion automatica (p.ej., longitud, presencia de bullets) y comparar con otro modelo (ej. `gpt-4o-mini`).

## Referencias

* Link al proyecto en Colab: [Practica 14.ipynb](https://colab.research.google.com/drive/1I6IfUsfqm8bDc6ypDsFGg5s85-H7aP_o?usp=drive_open#scrollTo=_kxNX-srmBXN)

---