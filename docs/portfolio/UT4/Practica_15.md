# Agentes con LangGraph y Tools (UT4-15)

## 1) Resumen de práctica

* Objetivo: construir un agente con **LangGraph** que use herramientas (RAG simple y lookup de pedidos) y probar el flujo de mensajes con `ChatOpenAI` (`gpt-5-mini`).
* Componentes clave: estado tipado con `messages` y `summary`, nodo `assistant` que llama al LLM con tools, nodo `tools` que ejecuta funciones, y routing condicional (`route_from_assistant`).
* Resultados: el agente responde sobre LangGraph y, con RAG, explica “¿qué es RAG?” usando el corpus vectorizado; el streaming muestra la secuencia human → ai.
  La extensión de memoria no se integró en el grafo compilado (quedó opcional).
* Incidentes: conflicto de `requests` (Colab pide 2.32.4, pip instala 2.32.5), warnings de `graph.compile` tras agregar memoria, y avisos de Gradio sobre `share=True` y `type` de chatbot.

## 2) Setup e instalación

* Instalado: `langgraph>=0.2`, `langchain>=0.2.11`, `langchain-openai`, `langchain-community`, `langchain-text-splitters`, `faiss-cpu`.
* Credenciales: variables de entorno para OpenAI y LangSmith (definidas en el notebook).
  Configurar en local sin exponer claves en código.

## 3) Estado y grafo

* Estado (`AgentState`): `messages: list` (acumula `HumanMessage`/`AIMessage`), `summary: Optional[str]` (reservado para memoria).
* Grafo base:

  * Nodo `assistant`: LLM `gpt-5-mini` bind con tools.
  * Nodo `tools`: ejecuta llamadas a tools cuando el LLM las solicita.
  * Routing: `START -> assistant`; si el último mensaje del LLM tiene `tool_calls`, se va a `tools`, luego vuelve a `assistant`; si no, `END`.

## 4) Tools implementadas

* `rag_search(question)`: usa un índice FAISS sobre un mini corpus (LangGraph, RAG, embeddings) construido con `RecursiveCharacterTextSplitter` + `OpenAIEmbeddings`. Devuelve texto relevante.
* `get_order_status(order_id)`: lookup en diccionario ficticio (`ABC123`, `XYZ999`).
* `get_utc_time()`: hora UTC (definida en el notebook para el set de tools).

## 5) Ejecuciones y salidas principales

* **Pregunta 1:** “Hola, ¿qué es LangGraph en pocas palabras?” → Respuesta resume LangGraph como framework de orquestación por nodos/edges para agentes y pipelines, integrado con modelos y tools, útil para chatbots y RAG.
* **Pregunta 2 (con RAG):** “Usá tu base de conocimiento y decime qué es RAG.” → Usa `rag_search` y devuelve definición completa: retrieve→context→generate, ventajas (factualidad, actualización), limitaciones (latencia, calidad de recuperación, ventana de contexto) y casos de uso.
* **Streaming:** muestra orden de mensajes: human → ai con explicación breve de RAG.

## 6) Parte opcional (memoria + UI)

* Se añadió `memory_node` que resume últimos 4 mensajes con el LLM, pero se agregó después de compilar el grafo → warnings y sin efecto en el grafo ya compilado.
* Se bosquejó UI en Gradio (`Chatbot`, `launch(share=True)`); avisos de deprecación de `type` y nota de Colab sobre compartir URL pública.

## 7) Issues y mitigaciones

* `pip` instala `requests 2.32.5` y Colab espera 2.32.4 → si surge error, fijar versión `requests==2.32.4` o reinstalar tras el notebook.
* Warnings de `graph.compile` al agregar nodos/edges luego de compilar → reconstruir grafo antes de compilar si se suma memoria.
* `share=True` automático en Gradio en Colab; ajustar `share`/`debug` según necesidad.

## 8) Conclusión

* El grafo base funciona: enruta llamadas a tools y responde con y sin RAG.
  La memoria queda pendiente de integración limpia (recompilar) y la UI de Gradio es un prototipo.
  Próximo paso: recompilar el grafo con `memory_node` incluido desde el inicio y probar interacciones completas por la interfaz.

## Referencias

* Link al proyecto en Colab: [Practica 15.ipynb](https://colab.research.google.com/drive/1DyCe0VDUmPfCVyhvzDg_zN61B3Q39k3F?usp=drive_open#scrollTo=0zSWfo8Y5csn)

---
