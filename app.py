import streamlit as st
from rag import RAGSystem  # Angenommen, Sie haben diese Datei

st.title("Saalbach Tourismus Chatbot")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Chatbot-Logik hier
# ...

# Beispiel für eine einfache Chat-Oberfläche
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Was möchten Sie über Saalbach wissen?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Hier würden Sie Ihre RAG-Logik aufrufen
    response = "Dies ist eine Beispielantwort. Implementieren Sie hier Ihre tatsächliche Chatbot-Logik."
    
    st.session_state.messages.append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.markdown(response)
