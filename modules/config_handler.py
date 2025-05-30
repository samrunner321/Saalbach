"""
Konfigurationsmanager für den Saalbach Tourismus Chatbot.
Verwaltet API-Keys und andere Einstellungen persistant.
Streamlit Cloud-kompatibel mit tempfile-Unterstützung.
"""

import os
import json
from typing import Dict, Any, Optional
import tempfile
import streamlit as st

class ConfigHandler:
    """Verwaltet die Konfiguration und API-Keys für den Chatbot."""
    
    def __init__(self, config_dir: str = None):
        """
        Initialisiert den ConfigHandler.
        
        Args:
            config_dir: Verzeichnis für die Konfigurationsdatei
        """
        # Prüfen, ob wir in Streamlit Cloud sind
        self.using_streamlit_cloud = self._check_streamlit_cloud()
        
        if config_dir is None:
            if self.using_streamlit_cloud:
                # Auf Streamlit Cloud ein temporäres Verzeichnis verwenden
                config_dir = tempfile.gettempdir()
            else:
                # Lokal im Projektverzeichnis speichern
                config_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        self.config_path = os.path.join(config_dir, "saalbach_config.json")
        self.config = self._load_config()
    
    def _check_streamlit_cloud(self) -> bool:
        """
        Prüft, ob wir auf Streamlit Cloud laufen.
        
        Returns:
            True wenn auf Streamlit Cloud, sonst False
        """
        # Streamlit Cloud setzt spezielle Umgebungsvariablen
        return "STREAMLIT_SHARING" in os.environ or "STREAMLIT_RUN_TARGET" in os.environ
    
    def _load_config(self) -> Dict[str, Any]:
        """
        Lädt die Konfiguration aus der JSON-Datei.
        
        Returns:
            Die geladene Konfiguration
        """
        # Bei Streamlit Cloud zuerst Secrets prüfen
        if self.using_streamlit_cloud and "openai" in st.secrets:
            config = self._get_default_config()
            
            # API-Key aus Secrets übernehmen, falls vorhanden
            if "api_keys" not in config:
                config["api_keys"] = {}
                
            if "openai" in st.secrets and "api_key" in st.secrets["openai"]:
                config["api_keys"]["openai"] = st.secrets["openai"]["api_key"]
                
            return config
            
        # Normale Konfigurationsdatei laden
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r', encoding='utf-8') as file:
                    return json.load(file)
            except Exception as e:
                print(f"Fehler beim Laden der Konfiguration: {str(e)}")
                return self._get_default_config()
        else:
            return self._get_default_config()
    
    def _save_config(self) -> None:
        """Speichert die aktuelle Konfiguration in der JSON-Datei."""
        # Auf Streamlit Cloud können wir nicht in die Projektstruktur schreiben
        if self.using_streamlit_cloud and "openai" in st.secrets:
            # Bei vorhandenen Secrets nichts speichern, werden ohnehin bevorzugt
            return
            
        try:
            with open(self.config_path, 'w', encoding='utf-8') as file:
                json.dump(self.config, file, indent=2)
            
            # Datei nur für den Besitzer lesbar machen (für API-Key-Sicherheit)
            try:
                os.chmod(self.config_path, 0o600)
            except:
                # Unter Windows funktioniert das möglicherweise nicht
                pass
                
        except Exception as e:
            print(f"Fehler beim Speichern der Konfiguration: {str(e)}")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """
        Erstellt eine Standardkonfiguration.
        
        Returns:
            Die Standardkonfiguration
        """
        return {
            "api_keys": {
                "openai": ""
            },
            "settings": {
                "model": "gpt-3.5-turbo",
                "temperature": 0.7,
                "max_tokens": 1000,
                "language": "de"
            },
            "rag_settings": {
                "use_own_knowledge_first": True,
                "n_results": 5
            }
        }
    
    def get_api_key(self, provider: str = "openai") -> str:
        """
        Gibt den API-Key für den angegebenen Provider zurück.
        Prüft zuerst Streamlit Secrets, dann gespeicherte Konfiguration.
        
        Args:
            provider: Der Name des API-Providers
            
        Returns:
            Der API-Key oder ein leerer String, wenn nicht vorhanden
        """
        # Zuerst Streamlit Secrets prüfen
        if self.using_streamlit_cloud:
            if provider in st.secrets and "api_key" in st.secrets[provider]:
                return st.secrets[provider]["api_key"]
                
        # Dann lokale Konfiguration prüfen
        return self.config.get("api_keys", {}).get(provider, "")
    
    def set_api_key(self, key: str, provider: str = "openai") -> None:
        """
        Setzt den API-Key für den angegebenen Provider.
        
        Args:
            key: Der zu speichernde API-Key
            provider: Der Name des API-Providers
        """
        # Bei Streamlit Cloud mit Secrets nichts tun
        if self.using_streamlit_cloud and provider in st.secrets and "api_key" in st.secrets[provider]:
            return
            
        if "api_keys" not in self.config:
            self.config["api_keys"] = {}
        
        self.config["api_keys"][provider] = key
        self._save_config()
    
    def get_setting(self, key: str, default: Any = None) -> Any:
        """
        Gibt eine Einstellung zurück.
        
        Args:
            key: Der Name der Einstellung
            default: Standardwert, falls die Einstellung nicht existiert
            
        Returns:
            Der Wert der Einstellung oder der Standardwert
        """
        # Zuerst Streamlit Secrets prüfen
        if self.using_streamlit_cloud and "settings" in st.secrets and key in st.secrets["settings"]:
            return st.secrets["settings"][key]
            
        return self.config.get("settings", {}).get(key, default)
    
    def set_setting(self, key: str, value: Any) -> None:
        """
        Setzt eine Einstellung.
        
        Args:
            key: Der Name der Einstellung
            value: Der zu speichernde Wert
        """
        if "settings" not in self.config:
            self.config["settings"] = {}
        
        self.config["settings"][key] = value
        self._save_config()
    
    def get_rag_setting(self, key: str, default: Any = None) -> Any:
        """
        Gibt eine RAG-Einstellung zurück.
        
        Args:
            key: Der Name der RAG-Einstellung
            default: Standardwert, falls die Einstellung nicht existiert
            
        Returns:
            Der Wert der RAG-Einstellung oder der Standardwert
        """
        # Zuerst Streamlit Secrets prüfen
        if self.using_streamlit_cloud and "rag_settings" in st.secrets and key in st.secrets["rag_settings"]:
            return st.secrets["rag_settings"][key]
            
        return self.config.get("rag_settings", {}).get(key, default)
    
    def set_rag_setting(self, key: str, value: Any) -> None:
        """
        Setzt eine RAG-Einstellung.
        
        Args:
            key: Der Name der RAG-Einstellung
            value: Der zu speichernde Wert
        """
        if "rag_settings" not in self.config:
            self.config["rag_settings"] = {}
        
        self.config["rag_settings"][key] = value
        self._save_config()
