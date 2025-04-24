"""
ChromaDB Manager für den Saalbach Tourismus Chatbot.
Verwaltet die Vektordatenbank für das RAG-System.
"""

import os
import chromadb
from chromadb.utils import embedding_functions
import uuid
from typing import List, Dict, Any, Optional, Union

# Konstanten für die ChromaDB-Konfiguration
import tempfile
DB_DIRECTORY = os.path.join(tempfile.gettempdir(), "saalbach_db")
COLLECTION_NAME = "saalbach_knowledge"

class ChromaManager:
    """Verwaltet die ChromaDB für das RAG-System des Saalbach-Chatbots."""
    
    def __init__(self, embedding_model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialisiert den ChromaDB Manager.
        
        Args:
            embedding_model_name: Name des zu verwendenden Embedding-Modells
        """
        # Sicherstellen, dass das DB-Verzeichnis existiert
        os.makedirs(DB_DIRECTORY, exist_ok=True)
        
        # ChromaDB Client initialisieren
        self.client = chromadb.PersistentClient(path=DB_DIRECTORY)
        
        # Embedding-Funktion definieren
        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=embedding_model_name
        )
        
        # Collection erstellen oder laden
        try:
            self.collection = self.client.get_collection(
                name=COLLECTION_NAME,
                embedding_function=self.embedding_function
            )
            print(f"Collection '{COLLECTION_NAME}' geladen.")
        except:
            self.collection = self.client.create_collection(
                name=COLLECTION_NAME,
                embedding_function=self.embedding_function
            )
            print(f"Collection '{COLLECTION_NAME}' neu erstellt.")
    
    def add_document(self, 
                    text: str, 
                    metadata: Dict[str, Any],
                    doc_id: Optional[str] = None) -> str:
        """
        Fügt ein Dokument zur Vektordatenbank hinzu.
        
        Args:
            text: Der Text des Dokuments
            metadata: Metadaten zum Dokument (Thema, Quelle, etc.)
            doc_id: Optional, ID des Dokuments. Wird automatisch generiert, wenn nicht angegeben.
            
        Returns:
            Die ID des hinzugefügten Dokuments
        """
        if not doc_id:
            doc_id = str(uuid.uuid4())
            
        self.collection.add(
            documents=[text],
            metadatas=[metadata],
            ids=[doc_id]
        )
        
        return doc_id
    
    def add_documents_batch(self, 
                          texts: List[str], 
                          metadatas: List[Dict[str, Any]],
                          ids: Optional[List[str]] = None) -> List[str]:
        """
        Fügt mehrere Dokumente gleichzeitig zur Vektordatenbank hinzu.
        
        Args:
            texts: Liste der Dokumententexte
            metadatas: Liste der Metadaten für die Dokumente
            ids: Optional, Liste der Dokument-IDs
            
        Returns:
            Liste der IDs der hinzugefügten Dokumente
        """
        if not ids:
            ids = [str(uuid.uuid4()) for _ in range(len(texts))]
            
        self.collection.add(
            documents=texts,
            metadatas=metadatas,
            ids=ids
        )
        
        return ids
    
    def search(self, 
              query: str, 
              n_results: int = 3, 
              filter_criteria: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Durchsucht die Vektordatenbank nach ähnlichen Dokumenten.
        
        Args:
            query: Die Suchanfrage
            n_results: Anzahl der zurückzugebenden Ergebnisse
            filter_criteria: Optional, Filterkriterien für die Suche
            
        Returns:
            Die Suchergebnisse
        """
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results,
            where=filter_criteria
        )
        
        return results
    
    def update_document(self, doc_id: str, text: str, metadata: Dict[str, Any]) -> None:
        """
        Aktualisiert ein vorhandenes Dokument.
        
        Args:
            doc_id: Die ID des zu aktualisierenden Dokuments
            text: Der neue Text
            metadata: Die neuen Metadaten
        """
        self.collection.update(
            ids=[doc_id],
            documents=[text],
            metadatas=[metadata]
        )
    
    def delete_document(self, doc_id: str) -> None:
        """
        Löscht ein Dokument aus der Datenbank.
        
        Args:
            doc_id: Die ID des zu löschenden Dokuments
        """
        self.collection.delete(ids=[doc_id])
    
    def get_document_count(self) -> int:
        """
        Gibt die Anzahl der Dokumente in der Collection zurück.
        
        Returns:
            Anzahl der Dokumente
        """
        return self.collection.count()
        
    def get_all_documents(self) -> Dict[str, Any]:
        """
        Ruft alle Dokumente aus der Collection ab.
        
        Returns:
            Alle Dokumente
        """
        return self.collection.get()
