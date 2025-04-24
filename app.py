"""
Hauptanwendung für den Saalbach Tourismus Chatbot.
Implementiert die Streamlit-Oberfläche und verknüpft alle Komponenten.
Mit Fallback-Mechanismus für ChromaDB-Probleme.
"""

import os
import sys
import streamlit as st
import traceback

# WICHTIG: st.set_page_config() muss der erste Streamlit-Befehl sein
st.set_page_config(
    page_title="Saalbach-Hinterglemm Chatbot",
    page_icon="🏔️",
    layout="wide"
)

# Globale Variablen für den Fallback-Mechanismus
USE_FALLBACK = False
IMPORT_ERROR = None

# Funktion zum Anzeigen von Debug-Informationen
def show_debug_info():
    with st.expander("Debug-Informationen (nur während der Entwicklung)", expanded=False):
        st.write(f"Python-Version: {sys.version}")
        st.write(f"Aktuelles Verzeichnis: {os.getcwd()}")
        st.write(f"Dateien im aktuellen Verzeichnis: {os.listdir('.')}")
        
        # Prüfen, ob modules-Verzeichnis existiert
        if os.path.exists("modules"):
            st.write(f"Dateien im modules-Verzeichnis: {os.listdir('modules')}")
            # Prüfe, ob __init__.py existiert
            if not os.path.exists(os.path.join("modules", "__init__.py")):
                st.warning("__init__.py fehlt im modules-Verzeichnis!")
        else:
            st.error("modules-Verzeichnis nicht gefunden!")

# Funktion zum Testen der OpenAI API
def test_openai_connection(api_key):
    """
    Testet, ob der OpenAI API-Schlüssel gültig ist.
    
    Args:
        api_key: Der zu testende API-Schlüssel
        
    Returns:
        (bool, str): Erfolgsstatus und Nachricht
    """
    if not api_key or api_key.strip() == "":
        return False, "Kein API-Schlüssel angegeben."
    
    try:
        # OpenAI-Client konfigurieren
        client = openai.OpenAI(api_key=api_key)
        
        # Kurze Testanfrage
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Test"}],
            max_tokens=5
        )
        
        # Erfolgreiche Verbindung
        return True, f"API-Verbindung erfolgreich (Modell: {response.model})"
    
    except Exception as e:
        # Fehlerbehandlung
        error_message = str(e)
        
        if "API key" in error_message.lower():
            return False, "Ungültiger API-Schlüssel. Bitte überprüfen Sie Ihre Eingabe."
        elif "quota" in error_message.lower() or "billing" in error_message.lower():
            return False, "Kontingent erschöpft oder Zahlungsproblem mit dem OpenAI-Konto."
        else:
            return False, f"Fehler bei der API-Verbindung: {error_message}"

# Funktion zum Initialisieren des Session State
def initialize_session_state():
    """Initialisiert die Session-State-Variablen."""
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "rag_system" not in st.session_state:
        st.session_state.rag_system = None
    if "show_api_key" not in st.session_state:
        st.session_state.show_api_key = True

# Hauptfunktion für die App-Logik
def main():
    global USE_FALLBACK, IMPORT_ERROR
    
    # Debug-Informationen anzeigen
    show_debug_info()

    # Versuche, erforderliche Module zu importieren
    try:
        import openai
        st.success("✅ OpenAI erfolgreich importiert")
    except ImportError as e:
        st.error(f"❌ Fehler beim Import von OpenAI: {str(e)}")
        st.stop()

    # Versuche, das RAG-System zu importieren
    try:
        from modules.rag import RAGSystem
        st.success("✅ RAGSystem erfolgreich importiert")
        
        # Teste, ob ChromaDB importierbar ist
        try:
            import chromadb
            st.success("✅ ChromaDB erfolgreich importiert")
        except Exception as e:
            st.warning(f"⚠️ ChromaDB konnte nicht importiert werden: {str(e)}")
            st.info("Der Chatbot wird im Fallback-Modus mit reduzierter Funktionalität ausgeführt.")
            USE_FALLBACK = True
            IMPORT_ERROR = str(e)
    except ImportError as e:
        st.warning(f"⚠️ RAGSystem konnte nicht importiert werden: {str(e)}")
        st.info("Der Chatbot wird im Fallback-Modus mit reduzierter Funktionalität ausgeführt.")
        USE_FALLBACK = True
        IMPORT_ERROR = str(e)

    # Fallback-Import, wenn das reguläre System nicht funktioniert
    if USE_FALLBACK:
        try:
            from modules.simple_rag import SimpleRAG
            st.success("✅ SimpleRAG (Fallback) erfolgreich importiert")
        except ImportError as e:
            st.error(f"❌ Fehler beim Import des Fallback-Systems: {str(e)}")
            st.stop()

    # Versuche, die restlichen Module zu importieren
    try:
        from modules.config_handler import ConfigHandler
        st.success("✅ ConfigHandler erfolgreich importiert")
        
        # Konfigurationsmanager initialisieren
        config = ConfigHandler()
    except ImportError as e:
        st.error(f"❌ Fehler beim Import des Config-Handlers: {str(e)}")
        st.code(traceback.format_exc())
        st.stop()  # App beenden, wenn der Config-Handler nicht verfügbar ist

    # Nachdem alle Debug-Infos angezeigt wurden, können wir die Hauptapp rendern
    render_main_app(config, USE_FALLBACK, IMPORT_ERROR)

def render_main_app(config, use_fallback, import_error):
    """
    Rendert die Hauptanwendung, nachdem alle Importe abgeschlossen sind.
    
    Args:
        config: ConfigHandler-Instanz
        use_fallback: Ob der Fallback-Modus verwendet werden soll
        import_error: Fehlermeldung, falls vorhanden
    """
    # Titel und Layout
    st.title("🏔️ Saalbach-Hinterglemm Tourismusberater")
    
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
        
        # Prüfen auf Streamlit Secrets
        has_secrets = False
        if 'openai' in st.secrets and 'api_key' in st.secrets['openai']:
            saved_api_key = st.secrets['openai']['api_key']
            has_secrets = True
            st.success("✅ API-Schlüssel aus Streamlit-Secrets geladen")
        else:
            # Gespeicherten API-Key laden
            saved_api_key = config.get_api_key()
        
        # Session-State für API-Key-Anzeige initialisieren
        if "show_api_key" not in st.session_state:
            st.session_state.show_api_key = not saved_api_key
        
        # API-Key Feld
        if st.session_state.show_api_key:
            openai_api_key = st.text_input(
                "OpenAI API Key", 
                value=saved_api_key,
                type="password",
                help="Der API-Schlüssel wird lokal gespeichert und muss nicht bei jedem Start neu eingegeben werden."
            )
            
            if st.button("API-Schlüssel speichern und testen"):
                with st.spinner("Teste Verbindung..."):
                    success, message = test_openai_connection(openai_api_key)
                    
                    if success:
                        # API-Schlüssel speichern
                        config.set_api_key(openai_api_key)
                        st.success(f"{message} - Schlüssel wurde gespeichert.")
                        st.session_state.show_api_key = False
                        # RAG-System mit neuem Schlüssel neu initialisieren
                        if "rag_system" in st.session_state:
                            del st.session_state.rag_system
                    else:
                        st.error(message)
        else:
            if has_secrets:
                st.success("✅ API-Schlüssel ist in Streamlit Secrets konfiguriert")
            else:
                st.success("✅ API-Schlüssel ist gespeichert")
                if st.button("API-Schlüssel ändern"):
                    st.session_state.show_api_key = True
        
        # Fortgeschrittene Einstellungen
        with st.expander("Fortgeschrittene Einstellungen", expanded=False):
            # Modellauswahl
            model_options = ["gpt-3.5-turbo", "gpt-4"]
            default_model = config.get_setting("model", "gpt-3.5-turbo")
            model_index = model_options.index(default_model) if default_model in model_options else 0
            
            model = st.selectbox(
                "LLM-Modell",
                model_options,
                index=model_index
            )
            
            if model != default_model:
                config.set_setting("model", model)
                if "rag_system" in st.session_state:
                    del st.session_state.rag_system
            
            # RAG-Einstellungen (nur anzeigen, wenn kein Fallback-Modus)
            if not use_fallback:
                use_own_knowledge = st.checkbox(
                    "Eigenes Wissen priorisieren", 
                    value=config.get_rag_setting("use_own_knowledge_first", True),
                    help="Wenn aktiviert, nutzt der Bot primär sein eigenes Wissen und ergänzt es mit spezifischen Informationen aus der Wissensdatenbank."
                )
                
                if use_own_knowledge != config.get_rag_setting("use_own_knowledge_first", True):
                    config.set_rag_setting("use_own_knowledge_first", use_own_knowledge)
                    if "rag_system" in st.session_state:
                        del st.session_state.rag_system
                
                n_results = st.slider(
                    "Anzahl der Informationsquellen", 
                    min_value=1, 
                    max_value=10, 
                    value=config.get_rag_setting("n_results", 5),
                    help="Anzahl der Informationsquellen aus der Wissensdatenbank, die für die Antwort genutzt werden."
                )
                
                if n_results != config.get_rag_setting("n_results", 5):
                    config.set_rag_setting("n_results", n_results)
                    if "rag_system" in st.session_state:
                        del st.session_state.rag_system
            else:
                st.info("Erweiterte RAG-Einstellungen sind im Fallback-Modus nicht verfügbar.")
                
            # Bibliotheksinformationen
            st.caption("OpenAI Version:")
            try:
                import pkg_resources
                openai_version = pkg_resources.get_distribution("openai").version
                st.code(f"openai=={openai_version}")
            except:
                st.code("Version nicht ermittelbar")
        
        # Status-Information anzeigen, wenn im Fallback-Modus
        if use_fallback:
            st.warning("⚠️ Der Chatbot läuft im Fallback-Modus mit reduzierter Funktionalität.")
            with st.expander("Details zum Fallback-Modus"):
                st.write("Der Fallback-Modus verwendet eine einfachere Textsuche anstelle der ChromaDB-Vektordatenbank.")
                st.write("Die Qualität der Antworten kann dadurch beeinträchtigt sein.")
                st.write("Fehlermeldung:")
                st.code(import_error)

    # Session State initialisieren
    initialize_session_state()

    # Chatbot-UI
    for message in st.session_state.chat_history:
        avatar = "🧑‍💻" if message["role"] == "user" else "🤖"
        with st.chat_message(message["role"], avatar=avatar):
            st.write(message["content"])

    # Benutzereingabe
    prompt = st.chat_input("Wie kann ich dir mit deiner Reise nach Saalbach-Hinterglemm helfen?")
    if prompt:
        # API-Key überprüfen
        api_key = None
        if 'openai' in st.secrets and 'api_key' in st.secrets['openai']:
            api_key = st.secrets['openai']['api_key']
        else:
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

        # RAG-System initialisieren oder neu laden, wenn es noch nicht existiert
        if st.session_state.rag_system is None:
            with st.spinner("Initialisiere Tourismusberater..."):
                try:
                    model = config.get_setting("model", "gpt-3.5-turbo")
                    
                    # Je nach Modus das passende System initialisieren
                    if not use_fallback:
                        from modules.rag import RAGSystem
                        st.session_state.rag_system = RAGSystem(api_key, model)
                    else:
                        from modules.simple_rag import SimpleRAG
                        st.session_state.rag_system = SimpleRAG(api_key, model)
                except Exception as e:
                    st.error(f"Fehler bei der Initialisierung des RAG-Systems: {str(e)}")
                    st.code(traceback.format_exc())
                    st.session_state.rag_system = None  # Setze auf None, damit es beim nächsten Versuch erneut initialisiert wird
        
        # Antwort mit Fortschrittsindikator generieren
        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("Denke nach..."):
                try:
                    # Chat-History für den Kontext vorbereiten (ohne Systemnachrichten)
                    chat_context = [
                        {"role": msg["role"], "content": msg["content"]}
                        for msg in st.session_state.chat_history[:-1]  # Letzte Nachricht ausschließen, wird separat hinzugefügt
                    ]
                    
                    # Antwort generieren
                    response = st.session_state.rag_system.answer_query(
                        query=prompt,
                        chat_history=chat_context
                    )
                    
                    # Antwort anzeigen
                    st.write(response)
                    
                    # Fehlerbehandlung für Fallback-Antwort
                    if "technisches Problem" in response or "überfordert" in response:
                        st.error("Es gab ein Problem bei der Verarbeitung deiner Anfrage. Bitte überprüfe die API-Einstellungen oder versuche es später erneut.")
                        
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

    # Status-Informationen (nur im normalen Modus)
    if not use_fallback:
        try:
            from modules.knowledge_base import KnowledgeBase
            # Status der Wissensbasis anzeigen
            with st.expander("Status der Wissensbasis", expanded=False):
                try:
                    kb = KnowledgeBase()
                    stats = kb.get_knowledge_statistics()
                    
                    if stats["total_documents"] > 0:
                        st.success(f"Wissensbasis aktiv: {stats['total_documents']} Dokumente in {len(stats['themes'])} Themen")
                        
                        # Themen anzeigen
                        themes_str = ", ".join(stats["themes"])
                        st.write(f"Verfügbare Themen: {themes_str}")
                    else:
                        st.warning("Keine Dokumente in der Wissensbasis gefunden. Bitte importieren Sie Daten über das Admin-Tool.")
                        
                except Exception as e:
                    st.error(f"Fehler beim Abrufen der Wissensbasis-Statistik: {str(e)}")
        except ImportError:
            pass

# App starten
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"Ein unerwarteter Fehler ist aufgetreten: {str(e)}")
        st.code(traceback.format_exc())
