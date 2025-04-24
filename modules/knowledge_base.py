"""
Wissensdatenbank-Verwaltung für den Saalbach Tourismus Chatbot.
Lädt und verarbeitet Markdown-Dateien und speichert sie in ChromaDB.
Streamlit Cloud-kompatibel mit verbesserter Fehlerbehandlung.
"""

import os
import re
import tempfile
import streamlit as st
from typing import List, Dict, Any, Tuple, Optional, Union
from modules.chroma_manager import ChromaManager

class KnowledgeBase:
    """
    Verwaltet die Wissensbasis des Saalbach Tourismus Chatbots.
    Lädt Markdown-Dateien und speichert sie in ChromaDB.
    """
    
    def __init__(self, knowledge_dir: str = None):
        """
        Initialisiert die Wissensbasis.
        
        Args:
            knowledge_dir: Verzeichnis mit den Markdown-Wissensquellen
        """
        try:
            # Prüfen, ob wir in Streamlit Cloud sind
            self.using_streamlit_cloud = "STREAMLIT_SHARING" in os.environ or "STREAMLIT_RUN_TARGET" in os.environ
            
            # Standard-Verzeichnis, wenn nicht angegeben
            if knowledge_dir is None:
                project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                knowledge_dir = os.path.join(project_root, "knowledge")
            
            self.knowledge_dir = knowledge_dir
            
            # Prüfen, ob das Verzeichnis existiert
            if not os.path.exists(self.knowledge_dir):
                print(f"Warnung: Wissensverzeichnis '{self.knowledge_dir}' existiert nicht.")
                
                # Alternativen ausprobieren
                alt_knowledge_dir = os.path.join(os.getcwd(), "knowledge")
                if os.path.exists(alt_knowledge_dir):
                    self.knowledge_dir = alt_knowledge_dir
                    print(f"Verwende alternatives Wissensverzeichnis: {alt_knowledge_dir}")
            
            # ChromaDB Manager initialisieren
            self.chroma_manager = ChromaManager()
            
            # Vorhandene Wissensdateien auflisten
            self.available_files = self._list_knowledge_files()
            
            print(f"Wissensbasis initialisiert. Verfügbare Dateien: {len(self.available_files)}")
            
        except Exception as e:
            print(f"Fehler bei der Initialisierung der Wissensbasis: {str(e)}")
            import traceback
            print(traceback.format_exc())
            self.knowledge_dir = None
            self.available_files = []
    
    def _list_knowledge_files(self) -> List[str]:
        """
        Listet alle verfügbaren Markdown-Dateien im Wissensverzeichnis auf.
        
        Returns:
            Liste der Dateipfade
        """
        if not os.path.exists(self.knowledge_dir):
            print(f"Warnung: Wissensverzeichnis '{self.knowledge_dir}' existiert nicht.")
            return []
            
        files = []
        for file in os.listdir(self.knowledge_dir):
            if file.endswith(".md"):
                files.append(os.path.join(self.knowledge_dir, file))
        
        return files
    
    def _split_markdown_into_chunks(self, content: str, max_chunk_size: int = 1000) -> List[Tuple[str, Dict[str, Any]]]:
        """
        Teilt einen Markdown-Text in sinnvolle Chunks für die Vektordatenbank.
        
        Args:
            content: Der Markdown-Inhalt
            max_chunk_size: Maximale Anzahl an Zeichen pro Chunk
            
        Returns:
            Liste von Tuples (chunk_text, metadata)
        """
        # Überschriften als Trennpunkte verwenden
        chunks = []
        current_section = ""
        current_heading = "Allgemein"
        current_subheading = ""  # Leerer String statt None
        
        lines = content.split('\n')
        
        for line in lines:
            # Hauptüberschrift erkennen (# Titel)
            if line.startswith('# '):
                # Vorherigen Abschnitt speichern, wenn vorhanden
                if current_section.strip():
                    chunks.append((
                        current_section.strip(),
                        {
                            "heading": current_heading,
                            "subheading": current_subheading
                        }
                    ))
                
                current_heading = line.lstrip('# ').strip()
                current_subheading = ""  # Leerer String statt None
                current_section = ""
            
            # Unterüberschrift erkennen (## Titel)
            elif line.startswith('## '):
                # Vorherigen Abschnitt speichern, wenn vorhanden
                if current_section.strip():
                    chunks.append((
                        current_section.strip(),
                        {
                            "heading": current_heading,
                            "subheading": current_subheading
                        }
                    ))
                
                current_subheading = line.lstrip('## ').strip()
                current_section = ""
            
            # Inhalt zum aktuellen Abschnitt hinzufügen
            else:
                current_section += line + '\n'
                
                # Chunk splitten, wenn er zu groß wird
                if len(current_section) > max_chunk_size:
                    chunks.append((
                        current_section.strip(),
                        {
                            "heading": current_heading,
                            "subheading": current_subheading
                        }
                    ))
                    current_section = ""
        
        # Letzten Abschnitt speichern, wenn vorhanden
        if current_section.strip():
            chunks.append((
                current_section.strip(),
                {
                    "heading": current_heading,
                    "subheading": current_subheading
                }
            ))
        
        return chunks
    
    def load_markdown_file(self, file_path: str) -> Dict[str, Any]:
        """
        Lädt eine Markdown-Datei und extrahiert relevante Informationen.
        
        Args:
            file_path: Pfad zur Markdown-Datei
            
        Returns:
            Informationen aus der Markdown-Datei
        """
        try:
            file_name = os.path.basename(file_path)
            theme = os.path.splitext(file_name)[0]  # Theme aus Dateiname ableiten
            
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
            
            chunks = self._split_markdown_into_chunks(content)
            
            result = {
                "theme": theme,
                "chunks": chunks,
                "file_path": file_path
            }
            
            return result
        except Exception as e:
            print(f"Fehler beim Laden der Markdown-Datei {file_path}: {str(e)}")
            return {
                "theme": "error",
                "chunks": [],
                "file_path": file_path,
                "error": str(e)
            }
    
    def import_markdown_to_chroma(self, file_path: str) -> List[str]:
        """
        Importiert eine Markdown-Datei in ChromaDB.
        
        Args:
            file_path: Pfad zur Markdown-Datei
            
        Returns:
            Liste der erstellten Dokument-IDs
        """
        try:
            markdown_data = self.load_markdown_file(file_path)
            
            if "error" in markdown_data:
                print(f"Fehler beim Importieren von {file_path}: {markdown_data['error']}")
                return []
                
            theme = markdown_data["theme"]
            
            texts = []
            metadatas = []
            
            for chunk_text, chunk_metadata in markdown_data["chunks"]:
                texts.append(chunk_text)
                
                # Metadaten zusammenstellen und sicherstellen, dass keine None-Werte enthalten sind
                metadata = {
                    "theme": theme,
                    "source_file": os.path.basename(file_path),
                    "heading": chunk_metadata.get("heading", "Allgemein"),
                    "subheading": chunk_metadata.get("subheading", "")  # Leerer String als Fallback
                }
                
                # Sicherstellen, dass alle Metadaten-Werte gültige Typen sind
                for key, value in metadata.items():
                    if value is None:
                        metadata[key] = ""  # None-Werte durch leere Strings ersetzen
                
                metadatas.append(metadata)
            
            # Batch-Import in ChromaDB, nur wenn wirklich Daten vorhanden sind
            if texts:
                doc_ids = self.chroma_manager.add_documents_batch(texts, metadatas)
                print(f"{len(texts)} Dokumente aus {file_path} zur Collection hinzugefügt.")
                return doc_ids
            else:
                print(f"Keine Dokumente in {file_path} gefunden.")
                return []
                
        except Exception as e:
            print(f"Fehler beim Importieren von {file_path} in ChromaDB: {str(e)}")
            return []
    
    def import_all_knowledge(self) -> Dict[str, int]:
        """
        Importiert alle verfügbaren Markdown-Dateien in ChromaDB.
        
        Returns:
            Ergebnisse des Imports (Dateiname -> Anzahl importierter Dokumente)
        """
        results = {}
        
        for file_path in self.available_files:
            try:
                doc_ids = self.import_markdown_to_chroma(file_path)
                results[os.path.basename(file_path)] = len(doc_ids)
            except Exception as e:
                print(f"Fehler beim Importieren von {file_path}: {str(e)}")
                results[os.path.basename(file_path)] = 0
        
        return results
    
    def get_knowledge_statistics(self) -> Dict[str, Any]:
        """
        Gibt Statistiken über die Wissensbasis zurück.
        
        Returns:
            Statistiken zur Wissensbasis
        """
        try:
            all_docs = self.chroma_manager.get_all_documents()
            
            # Wenn keine Dokumente vorhanden sind, importiere alle verfügbaren
            if not all_docs.get("metadatas", []) and self.available_files:
                print("Keine Dokumente in der Wissensbasis gefunden. Importiere verfügbare Dateien...")
                self.import_all_knowledge()
                all_docs = self.chroma_manager.get_all_documents()
            
            # Anzahl der Dokumente nach Thema gruppieren
            themes = {}
            for metadata in all_docs.get("metadatas", []):
                theme = metadata.get("theme", "Unbekannt")
                if theme not in themes:
                    themes[theme] = 0
                themes[theme] += 1
            
            # Gesamtanzahl der Dokumente
            total_docs = self.chroma_manager.get_document_count()
            
            return {
                "total_documents": total_docs,
                "documents_by_theme": themes,
                "themes": list(themes.keys()),
                "available_files": [os.path.basename(f) for f in self.available_files]
            }
        except Exception as e:
            print(f"Fehler beim Abrufen der Wissensbasis-Statistik: {str(e)}")
            return {
                "total_documents": 0,
                "documents_by_theme": {},
                "themes": [],
                "available_files": [os.path.basename(f) for f in self.available_files],
                "error": str(e)
            }
