"""
Minimale Version des Saalbach Tourismus Chatbots für initiales Deployment.
Diese Version ist auf Kompatibilität mit Streamlit Cloud optimiert.
"""

import streamlit as st

# WICHTIG: page_config MUSS der erste Streamlit-Befehl sein
st.set_page_config(
    page_title="Saalbach-Hinterglemm Chatbot",
    page_icon="🏔️",
    layout="wide"
)

# Haupttitel anzeigen
st.title("🏔️ Saalbach-Hinterglemm Tourismusberater")
st.subheader("Initialisierung...")

# Grundlegende Statusmeldung
st.info("Der Chatbot wird initialisiert. Diese minimale Version dient dazu, die Streamlit Cloud Deployment-Probleme zu beheben.")

try:
    # Jetzt versuchen wir, die Module zu importieren
    st.text("Versuche Module zu importieren...")
    
    # OpenAI importieren
    import openai
    st.success("✅ OpenAI erfolgreich importiert")
    
    # Python-Version anzeigen
    import sys
    st.text(f"Python-Version: {sys.version}")
    
    # Verzeichnisstruktur anzeigen
    import os
    st.text(f"Aktuelles Verzeichnis: {os.getcwd()}")
    st.text(f"Dateien im aktuellen Verzeichnis: {os.listdir('.')}")
    
    if os.path.exists("modules"):
        st.text(f"Dateien im modules-Verzeichnis: {os.listdir('modules')}")
        
        # Prüfen, ob __init__.py existiert
        init_path = os.path.join("modules", "__init__.py")
        if os.path.exists(init_path):
            st.success("✅ __init__.py existiert im modules-Verzeichnis")
            
            # Jetzt versuchen wir, die Module nacheinander zu importieren
            try:
                from modules.config_handler import ConfigHandler
                st.success("✅ ConfigHandler erfolgreich importiert")
                
                try:
                    from modules.chroma_manager import ChromaManager
                    st.success("✅ ChromaManager erfolgreich importiert") 
                except ImportError as e:
                    st.warning(f"⚠️ ChromaManager konnte nicht importiert werden: {e}")
                
                try:
                    from modules.knowledge_base import KnowledgeBase
                    st.success("✅ KnowledgeBase erfolgreich importiert")
                except ImportError as e:
                    st.warning(f"⚠️ KnowledgeBase konnte nicht importiert werden: {e}")
                
                try:
                    from modules.rag import RAGSystem
                    st.success("✅ RAGSystem erfolgreich importiert")
                except ImportError as e:
                    st.warning(f"⚠️ RAGSystem konnte nicht importiert werden: {e}")
                    
                try:
                    from modules.simple_rag import SimpleRAG
                    st.success("✅ SimpleRAG erfolgreich importiert")
                except ImportError as e:
                    st.warning(f"⚠️ SimpleRAG konnte nicht importiert werden: {e}")
                    
            except ImportError as e:
                st.error(f"❌ Fehler beim Import: {e}")
        else:
            st.error("❌ __init__.py fehlt im modules-Verzeichnis!")
    else:
        st.error("❌ modules-Verzeichnis nicht gefunden!")
        
    # ChromaDB direkt testen
    try:
        import chromadb
        st.success("✅ ChromaDB erfolgreich importiert")
    except ImportError as e:
        st.error(f"❌ ChromaDB konnte nicht importiert werden: {e}")
        
except Exception as e:
    st.error(f"Ein Fehler ist aufgetreten: {e}")
    
# Weitere Informationen
st.markdown("---")
with st.expander("Nächste Schritte"):
    st.markdown("""
    1. Prüfe, ob die Streamlit Cloud Konfiguration korrekt ist (.streamlit/config.toml)
    2. Stelle sicher, dass alle Module korrekt installiert sind (requirements.txt)
    3. Nach erfolgreicher Initialisierung kann die vollständige App implementiert werden
    """)
