import os
import streamlit as st
import faiss
import pickle
import numpy as np
import pandas as pd
import json
from sentence_transformers import SentenceTransformer
from groq import Groq
from openai import OpenAI

# Configuración inicial
os.environ["GROG_API_KEY"] = "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"
client = Groq(api_key=os.environ.get("GROG_API_KEY"))





# Cargar embeddings y datos
with open('embeddings3.pkl', 'rb') as f:
    embeddings = pickle.load(f)

df = pd.read_csv('DATA3.csv')  # Archivo CSV con los fragmentos
texts = df['contenido_fragmento'].tolist()

# Cargar el índice FAISS previamente guardado
index = faiss.read_index('faiss_index3.index')  # Carga el índice guardado

# Cargar modelo de embeddings para consultas
model = SentenceTransformer('all-mpnet-base-v2')

# Archivo para guardar el historial
HISTORIAL_PATH = "chat_history.json"


# Función para obtener el contexto relevante del índice FAISS
def get_context_from_faiss(query, k=3):
    query_embedding = model.encode([query])[0]  # Generar embedding de la consulta
    query_embedding = np.array([query_embedding], dtype=np.float32)
    _, indices = index.search(query_embedding, k)  # Buscar los k fragmentos más relevantes
    return [texts[i] for i in indices[0]]  # Recuperar los textos correspondientes

# Función para obtener el sistema de prompt con contexto y lenguaje
def get_system_prompt(context: str, language: str = "Spanish"):
    return f""" Eres Lio, un asistente virtual altamente especializado en auditoría financiera, creado para brindar apoyo a los auditores con la mayor amabilidad y profesionalismo. Tu misión es ayudar en todo lo relacionado con la auditoría financiera, proporcionando respuestas claras, precisas y siempre alentadoras. Tu conocimiento está basado en las NIIF, las NIA, las leyes tributarias y laborales de Ecuador, y las mejores prácticas en el campo de la auditoría.

    Tu propósito es apoyar de manera experta en áreas clave de auditoría financiera, incluyendo:

    - *Porcentaje del 2025 del IVA es 15%, manteniendo las tarifas 0% en productos de canasta basica y servicios basicos, 8% en feriados y 5% en materiales de construccion
    - *Aplicación de Normas Internacionales de Auditoría (NIA)*, especialmente NIA 320, NIA 315 y el marco COSO.
    - *Eres un experto en la Ley de Regimen Tributario Interno de Ecuador
    - *Brindar información acerca de la nueva norma, NIIF 18 Presentación e información a revelar en los estados financieros, la cual entrará en vigencia a partir del 1 de enero del 2027
    - *Análisis de normativas de sostenibilidad* e impacto ambiental en auditoría financiera.
    - *Recomendaciones basadas en casos emblemáticos*, como el Caso Enron, Caso Coopera y la Ley Sarbanes-Oxley (SOX).
    - *Prevención de lavado de activos* y evaluación de riesgos mediante Normas UAFFE.

    Recuerda que tu rol es mantener un tono amable y profesional en todo momento. Evitarás responder preguntas fuera del ámbito de la auditoría financiera, contabilidad, normativa fiscal ecuatoriana y leyes laborales. ¡Tu trabajo es ser una fuente confiable y amigable de conocimiento para los auditores!

    Este es el contexto para tu tarea:
    '''
    {context}
    '''
    """
with st.expander("💡 Tips para obtener mejores resultados:"):
    st.markdown("""
    - Sé claro y específico en tu pregunta.
    - Proporciona contexto relevante.
    - Haz preguntas concretas y estructuradas.
    - Indica el formato de respuesta que prefieres.
    - Usa términos técnicos del área.
    - Solicita ejemplos o casos prácticos si es necesario.
    - Aunque trato de hacerlo lo mejor posible, a veces puedo cometer errores. 
      ¡No te preocupes! Si algo no queda claro, solo vuelve a preguntar o proporciona un poco más de contexto. 
      ¡Siempre estoy aquí para ayudarte!
    """)

# Título de la aplicación
st.title("Lio Assistant")

# Mostrar el mensaje de bienvenida
st.write("¡Hola! Soy Lio, tu asistente virtual experto en auditoría financiera.")

mensaje_inicial={
    "role": "system",
    "content": """✨ ¡Feliz Año Nuevo! ✨
Que este año esté lleno de éxitos, alegría y nuevas oportunidades. 🎉
🌟 Estoy aquí para ayudarte con lo que necesites. ¿Por dónde empezamos hoy? 😊"""
}


# Cargar historial guardado
def load_history():
    if os.path.exists(HISTORIAL_PATH):
        with open(HISTORIAL_PATH, "r") as f:
            return json.load(f)
    return []

# Guardar historial actual
def save_history(messages):
    with open(HISTORIAL_PATH, "w") as f:
        json.dump(messages, f)



# Inicializa el historial del chat
if "messages" not in st.session_state:
    st.session_state.messages = [mensaje_inicial]

# Mostrar los mensajes anteriores en el chat (tanto los del usuario como los del asistente)
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Entrada de texto para la pregunta del usuario
if prompt := st.chat_input("¿En qué puedo ayudarte?"):
    # Obtener el contexto relevante del índice FAISS
    context = get_context_from_faiss(prompt, k=3)

    # Crear el prompt con el contexto actualizado solo para el sistema
    system_message = get_system_prompt(context, "Spanish")

    # Agregar la pregunta del usuario al historial de la conversación
    user_message = {
        "role": "user",
        "content": prompt
    }
    st.session_state.messages.append(user_message)

    # Mostrar el mensaje del usuario en el chat
    with st.chat_message("user"):
        st.markdown(prompt)  # Mostrar pregunta del usuario

    llm_response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": system_message}
        ] + [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
    )

    assistant_reply = llm_response.choices[0].message.content
    # Mostrar la respuesta del asistente en el chat
    with st.chat_message("assistant"):
        st.markdown(assistant_reply)


    # Agregar la respuesta del asistente al historial de la conversación
    st.session_state.messages.append({
        "role": "assistant",
        "content": assistant_reply
    })



# import os
# import streamlit as st
# import faiss
# import pickle
# import numpy as np
# import pandas as pd
# from sentence_transformers import SentenceTransformer
# from groq import Groq

# # Configuración inicial
# os.environ["GROG_API_KEY"] = "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"
# client = Groq(api_key=os.environ.get("GROG_API_KEY"))

# # Cargar embeddings y datos
# with open(r'C:\Users\Usuario\Desktop\TESIS\TESIS DATA\PRIMERA VERSION\embeddings4.pkl','rb') as f:
#     embeddings = pickle.load(f)

# df = pd.read_csv(r'C:\Users\Usuario\Desktop\TESIS\TESIS DATA\PRIMERA VERSION\fragmentos_dos.csv')  # Archivo CSV con los fragmentos
# texts = df['contenido_fragmento'].tolist()

# # Cargar el índice FAISS previamente guardado
# index = faiss.read_index(r'C:\Users\Usuario\Desktop\TESIS\TESIS DATA\PRIMERA VERSION\faiss_index.index')  # Carga el índice guardado

# # Cargar modelo de embeddings para consultas
# model = SentenceTransformer('all-mpnet-base-v2')

# # Función para obtener el contexto relevante del índice FAISS
# def get_context_from_faiss(query, k=2):
#     query_embedding = model.encode([query])[0]  # Generar embedding de la consulta
#     query_embedding = np.array([query_embedding], dtype=np.float32)
#     _, indices = index.search(query_embedding, k)  # Buscar los k fragmentos más relevantes
#     return [texts[i] for i in indices[0]]  # Recuperar los textos correspondientes

# # Función para obtener el sistema de prompt con contexto y lenguaje
# def get_system_prompt(context: str, language: str = "Spanish"):
#     return f"""Te llamas Lio, eres un asistente virtual experto en auditoría financiera...
# {context}
# """

# # Título de la aplicación
# st.title("Lio Assistant")

# # Mensaje inicial
# mensaje_inicial = {
#     "role": "system",
#     "content": "¿En qué te puedo ayudar hoy?"
# }

# # Inicializa el historial del chat
# if "messages" not in st.session_state:
#     st.session_state.messages = [mensaje_inicial]

# # Mostrar historial en la barra lateral
# st.sidebar.header("Historial de Conversaciones")
# if "chat_history" not in st.session_state:
#     st.session_state.chat_history = []

# for idx, chat in enumerate(st.session_state.chat_history):
#     st.sidebar.write(f"Chat {idx + 1}: {chat['title']}")

# if st.sidebar.button("Nuevo Chat"):
#     st.session_state.messages = [mensaje_inicial]
#     st.session_state.chat_history.append({
#         "title": f"Chat {len(st.session_state.chat_history) + 1}",
#         "messages": []
#     })
#     st.experimental_rerun()

# # Mostrar mensajes en el cuerpo principal
# for message in st.session_state.messages:
#     with st.chat_message(message["role"]):
#         st.markdown(message["content"])

# # Entrada de texto
# if prompt := st.chat_input("¿En qué puedo ayudarte?"):
#     # Obtener el contexto relevante del índice FAISS
#     context = get_context_from_faiss(prompt, k=2)
#     context_str = " ".join(context)

#     # Crear el prompt con el contexto
#     system_message = get_system_prompt(context_str, "Spanish")

#     # Agregar el mensaje del usuario al historial
#     user_message = {
#         "role": "user",
#         "content": prompt
#     }
#     st.session_state.messages.append(user_message)

#     with st.chat_message("user"):
#         st.markdown(prompt)

#     llm_response = client.chat.completions.create(
#         model="llama-3.1-70b-versatile",
#         messages=[
#             {"role": "system", "content": system_message}
#         ] + [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
#     )

#     assistant_reply = llm_response.choices[0].message.content

#     with st.chat_message("assistant"):
#         st.markdown(assistant_reply)

#     st.session_state.messages.append({
#         "role": "assistant",
#         "content": assistant_reply
#     })

#     # Agregar al historial de conversaciones
#     if len(st.session_state.chat_history) > 0:
#         st.session_state.chat_history[-1]["messages"] = st.session_state.messages

