"""
Wissensdatenbank-Verwaltung für den Saalbach Tourismus Chatbot.
Lädt und verarbeitet Markdown-Dateien und speichert sie in ChromaDB.
"""

import os
import re
from typing import List, Dict, Any, Tuple
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
        # Standard-Verzeichnis, wenn nicht angegeben
        if knowledge_dir is None:
            knowledge_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "knowledge")
        
        self.knowledge_dir = knowledge_dir
        self.chroma_manager = ChromaManager()
    
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
    
    def import_markdown_to_chroma(self, file_path: str) -> List[str]:
        """
        Importiert eine Markdown-Datei in ChromaDB.
        
        Args:
            file_path: Pfad zur Markdown-Datei
            
        Returns:
            Liste der erstellten Dokument-IDs
        """
        markdown_data = self.load_markdown_file(file_path)
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
        
        # Batch-Import in ChromaDB
        doc_ids = self.chroma_manager.add_documents_batch(texts, metadatas)
        
        return doc_ids
    
    def get_knowledge_statistics(self) -> Dict[str, Any]:
        """
        Gibt Statistiken über die Wissensbasis zurück.
        
        Returns:
            Statistiken zur Wissensbasis
        """
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
            "themes": list(themes.keys())
        }