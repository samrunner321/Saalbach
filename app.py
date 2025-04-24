"""
Minimal-Version des Saalbach Tourismus Chatbots, die nur im Fallback-Modus läuft.
Diese Version benötigt keine ChromaDB oder andere komplexe Abhängigkeiten.
"""

import os
import sys
import streamlit as st
import json
import re
import traceback

# WICHTIG: st.set_page_config() MUSS der erste Streamlit-Befehl sein
st.set_page_config(
    page_title="Saalbach-Hinterglemm Chatbot",
    page_icon="🏔️",
    layout="wide"
)

# Haupt-UI anzeigen
st.title("🏔️ Saalbach-Hinterglemm Tourismusberater")

# Globale Variablen
KNOWLEDGE_DIR = "knowledge"
CONFIG_FILE = "config.json"

# Klassen-Definitionen für den minimalen Betrieb
class SimpleConfig:
    """Einfache Konfigurationsverwaltung ohne externe Abhängigkeiten."""
    
    def __init__(self):
        """Initialisiert die Konfigurationsverwaltung."""
        self.config = self._load_config()
        
    def _load_config(self):
        """Lädt Konfiguration aus Session-State oder erstellt eine neue."""
        if "config" in st.session_state:
            return st.session_state.config
            
        # Standard-Konfiguration
        default_config = {
            "api_keys": {"openai": ""},
            "settings": {"model": "gpt-3.5-turbo"},
            "rag_settings": {"n_results": 3}
        }
        
        # Prüfen auf Streamlit Secrets
        if 'openai' in st.secrets and 'api_key' in st.secrets['openai']:
            default_config["api_keys"]["openai"] = st.secrets['openai']['api_key']
            
        # In Session-State speichern
        st.session_state.config = default_config
        return default_config
    
    def get_api_key(self, provider="openai"):
        """API-Key aus Konfiguration lesen."""
        # Zuerst Streamlit Secrets prüfen
        if provider in st.secrets and "api_key" in st.secrets[provider]:
            return st.secrets[provider]["api_key"]
            
        return self.config.get("api_keys", {}).get(provider, "")
    
    def set_api_key(self, key, provider="openai"):
        """API-Key in Konfiguration speichern."""
        if "api_keys" not in self.config:
            self.config["api_keys"] = {}
        
        self.config["api_keys"][provider] = key
        st.session_state.config = self.config
    
    def get_setting(self, key, default=None):
        """Einstellung aus Konfiguration lesen."""
        return self.config.get("settings", {}).get(key, default)
    
    def set_setting(self, key, value):
        """Einstellung in Konfiguration speichern."""
        if "settings" not in self.config:
            self.config["settings"] = {}
        
        self.config["settings"][key] = value
        st.session_state.config = self.config

# Sehr einfache RAG-Implementierung ohne externe Abhängigkeiten
class VerySimpleRAG:
    """
    Extrem einfache RAG-Implementierung ohne externe Abhängigkeiten.
    Verwendet nur Standard-Python-Bibliotheken.
    """
    
    def __init__(self, openai_api_key=None, model="gpt-3.5-turbo"):
        """Initialisiert das einfache RAG-System."""
        self.api_key = openai_api_key
        self.model = model
        self.knowledge_base = self._load_knowledge_base()
        
        # Basis-Prompt
        self.base_system_prompt = """
Du bist ein freundlicher, persönlicher Tourismus-Assistent für die Region Saalbach-Hinterglemm. 
Du sprichst wie ein echter Einheimischer, der die Region liebt und alle Insider-Tipps kennt.
Du nutzt dein umfangreiches eigenes Wissen über Saalbach-Hinterglemm und den Skicircus, einschließlich Unterkünfte, Restaurants, Wanderwege, Skigebiete, Biketouren und Sehenswürdigkeiten.

Halte dich an diese Kommunikationsregeln:
- Beginne deine Antworten immer mit einer herzlichen Begrüßung wie "Servus!", "Grüß dich!" oder "Hallo!"
- Duze die Gäste immer - das ist in Saalbach üblich und persönlicher
- Verwende einen begeisterten, lebendigen Gesprächsstil mit österreichischer Färbung
- Verwende Emojis, um deinen Antworten Persönlichkeit zu verleihen 😊 🏔️ 🚵‍♂️
- Sei proaktiv und gib konkrete, detaillierte Empfehlungen statt allgemeiner Aussagen
- Beende längere Antworten mit einer Frage, um das Gespräch fortzuführen
"""
        
    def _load_knowledge_base(self):
        """Lädt das Wissen aus den Markdown-Dateien."""
        knowledge = []
        
        if not os.path.exists(KNOWLEDGE_DIR):
            st.warning(f"Verzeichnis '{KNOWLEDGE_DIR}' nicht gefunden.")
            return knowledge
            
        # Alle Markdown-Dateien im Verzeichnis laden
        for filename in os.listdir(KNOWLEDGE_DIR):
            if filename.endswith(".md"):
                try:
                    file_path = os.path.join(KNOWLEDGE_DIR, filename)
                    with open(file_path, 'r', encoding='utf-8') as file:
                        content = file.read()
                        knowledge.append({
                            "theme": os.path.splitext(filename)[0],
                            "content": content
                        })
                except Exception as e:
                    st.warning(f"Fehler beim Laden von {filename}: {str(e)}")
        
        return knowledge
    
    def _simple_search(self, query, n_results=3):
        """Sehr einfache Textsuche ohne externe Bibliotheken."""
        results = []
        
        if not self.knowledge_base:
            return results
            
        # Query in Wörter aufteilen
        query_words = set(re.findall(r'\w+', query.lower()))
        
        # Jedes Dokument bewerten
        for doc in self.knowledge_base:
            content = doc["content"].lower()
            score = 0
            
            for word in query_words:
                if word in content:
                    score += 1
            
            if score > 0:
                results.append({
                    "document": doc,
                    "score": score
                })
        
        # Nach Score sortieren
        results.sort(key=lambda x: x["score"], reverse=True)
        
        # Begrenzte Anzahl zurückgeben
        return [r["document"] for r in results[:n_results]]
    
    def answer_query(self, query, chat_history=None):
        """Beantwortet eine Anfrage mit einfacher Textsuche und OpenAI."""
        try:
            if not self.api_key:
                return "Servus! Ich brauche einen API-Schlüssel, um dir helfen zu können. Bitte gib einen OpenAI API-Schlüssel in den Einstellungen ein. Danke! 😊"
            
            # Relevante Dokumente suchen
            relevant_docs = self._simple_search(query, n_results=2)
            
            # Kontext erstellen
            context = ""
            if relevant_docs:
                context = "\n\n".join([doc["content"][:2000] for doc in relevant_docs])
            else:
                context = "Keine spezifischen Informationen verfügbar. Nutze dein eigenes Wissen."
            
            # System-Prompt
            system_prompt = f"{self.base_system_prompt}\n\nZUSÄTZLICHE INFORMATION:\n{context[:4000]}"
            
            # OpenAI API aufrufen
            import openai
            client = openai.OpenAI(api_key=self.api_key)
            
            # Nachrichten vorbereiten
            messages = [{"role": "system", "content": system_prompt}]
            
            # Chat-Verlauf hinzufügen (maximal 3 Nachrichten)
            if chat_history:
                recent_history = chat_history[-3:] if len(chat_history) > 3 else chat_history
                messages.extend(recent_history)
            
            # Anfrage
            messages.append({"role": "user", "content": query})
            
            # OpenAI API aufrufen
            response = client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.7,
                max_tokens=800
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            st.error(f"Fehler: {str(e)}")
            return f"Servus! Es tut mir leid, aber ich habe gerade ein technisches Problem. Magst du es in ein paar Minuten nochmal versuchen? Danke für dein Verständnis! 😊"

# Funktion zum Testen der OpenAI API
def test_openai_connection(api_key):
    """Testet die Verbindung zur OpenAI API."""
    if not api_key or api_key.strip() == "":
        return False, "Kein API-Schlüssel angegeben."
    
    try:
        import openai
        client = openai.OpenAI(api_key=api_key)
        
        # Test-Anfrage
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Test"}],
            max_tokens=5
        )
        
        return True, f"API-Verbindung erfolgreich (Modell: {response.model})"
    
    except Exception as e:
        error_message = str(e)
        
        if "API key" in error_message.lower():
            return False, "Ungültiger API-Schlüssel."
        elif "quota" in error_message.lower() or "billing" in error_message.lower():
            return False, "Kontingent erschöpft oder Zahlungsproblem."
        else:
            return False, f"Fehler: {error_message}"

# Initialisierung des Session-State
def initialize_session():
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "rag_system" not in st.session_state:
        st.session_state.rag_system = None
    if "show_api_key" not in st.session_state:
        st.session_state.show_api_key = True

# Sidebar mit Informationen
with st.sidebar:
    st.image("https://via.placeholder.com/150x80?text=Saalbach", width=150)
    st.title("Saalbach-Hinterglemm")
    st.subheader("Ihr virtueller Reiseberater")
    
    st.markdown("""
    Stellen Sie Fragen zu:
    - Skigebieten und Pisten
    - Mountainbike-Strecken
    - Unterkünften und Hotels
    - Restaurants und Kulinarik
    - Events und Veranstaltungen
    - Joker Card und Angeboten
    """)
    
    # API-Key Verwaltung
    st.markdown("### API-Konfiguration")
    
    # Konfiguration laden
    config = SimpleConfig()
    saved_api_key = config.get_api_key()
    
    # API-Key Feld
    if "show_api_key" not in st.session_state:
        st.session_state.show_api_key = not saved_api_key
    
    if st.session_state.show_api_key:
        openai_api_key = st.text_input(
            "OpenAI API Key", 
            value=saved_api_key,
            type="password",
            help="Der API-Schlüssel wird für diese Sitzung gespeichert."
        )
        
        if st.button("API-Schlüssel speichern und testen"):
            with st.spinner("Teste Verbindung..."):
                success, message = test_openai_connection(openai_api_key)
                
                if success:
                    config.set_api_key(openai_api_key)
                    st.success(f"{message}")
                    st.session_state.show_api_key = False
                    if "rag_system" in st.session_state:
                        st.session_state.rag_system = None
                else:
                    st.error(message)
    else:
        st.success("✅ API-Schlüssel ist gespeichert")
        if st.button("API-Schlüssel ändern"):
            st.session_state.show_api_key = True
    
    # Modellauswahl
    with st.expander("Modell-Einstellungen"):
        model_options = ["gpt-3.5-turbo", "gpt-4"]
        default_model = config.get_setting("model", "gpt-3.5-turbo")
        model_index = 0
        if default_model in model_options:
            model_index = model_options.index(default_model)
        
        model = st.selectbox(
            "LLM-Modell",
            model_options,
            index=model_index
        )
        
        if model != default_model:
            config.set_setting("model", model)
            if "rag_system" in st.session_state:
                st.session_state.rag_system = None

# Session State initialisieren
initialize_session()

# Chatbot-UI
for message in st.session_state.chat_history:
    avatar = "🧑‍💻" if message["role"] == "user" else "🤖"
    with st.chat_message(message["role"], avatar=avatar):
        st.write(message["content"])

# Benutzereingabe
prompt = st.chat_input("Wie kann ich dir mit deiner Reise nach Saalbach-Hinterglemm helfen?")
if prompt:
    # API-Key überprüfen
    api_key = config.get_api_key()
    
    if not api_key:
        with st.chat_message("assistant", avatar="🤖"):
            st.warning("Servus! Ich brauche einen API-Schlüssel, um dir helfen zu können. Bitte gib einen OpenAI API-Schlüssel in den Einstellungen links ein. Dann kann ich dich richtig beraten! 😊")
        st.stop()
    
    # Benutzernachricht anzeigen
    with st.chat_message("user", avatar="🧑‍💻"):
        st.write(prompt)
    
    # Nachricht zur Chat-History hinzufügen
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    
    # RAG-System initialisieren, falls nötig
    if st.session_state.rag_system is None:
        with st.spinner("Initialisiere Tourismusberater..."):
            model = config.get_setting("model", "gpt-3.5-turbo")
            st.session_state.rag_system = VerySimpleRAG(api_key, model)
    
    # Antwort generieren
    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("Denke nach..."):
            try:
                # Chat-History für den Kontext vorbereiten
                chat_context = [
                    {"role": msg["role"], "content": msg["content"]}
                    for msg in st.session_state.chat_history[:-1]
                ]
                
                # Antwort generieren
                response = st.session_state.rag_system.answer_query(
                    query=prompt,
                    chat_history=chat_context
                )
                
                # Antwort anzeigen
                st.write(response)
                
            except Exception as e:
                error_message = f"Fehler bei der Verarbeitung: {str(e)}"
                st.error(error_message)
                response = "Servus! Entschuldige bitte, ich habe gerade ein technisches Problem. Magst du es in ein paar Minuten nochmal versuchen? Danke für dein Verständnis! 😊"
                st.write(response)
        
        # Antwort zur Chat-History hinzufügen
        st.session_state.chat_history.append({"role": "assistant", "content": response})

# Hinweis zur Verwendung am Ende
st.markdown("---")
st.caption("Dies ist ein KI-gestützter Chatbot. Bitte beachten Sie, dass sich Informationen ändern können.")
st.caption("Diese Version läuft im Fallback-Modus mit reduzierter Funktionalität und minimal benötigten Abhängigkeiten.")
