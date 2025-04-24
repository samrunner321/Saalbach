"""
ChromaDB Manager für den Saalbach Tourismus Chatbot.
Verwaltet die Vektordatenbank für das RAG-System.
Mit verbesserter Fehlerbehandlung und Fallback-Mechanismen.
"""

import os
import sys
import tempfile
import uuid
import traceback
from typing import List, Dict, Any, Optional, Union

# Globale Variable für Fehler-Fallback
CHROMA_INITIALIZED = False
ERROR_MESSAGE = ""

try:
    print("Versuche ChromaDB zu importieren...")
    import chromadb
    print("ChromaDB erfolgreich importiert!")
    
    try:
        from chromadb.utils import embedding_functions
        print("Embedding-Funktionen erfolgreich importiert!")
        CHROMA_INITIALIZED = True
    except Exception as e:
        ERROR_MESSAGE = f"Fehler beim Import der Embedding-Funktionen: {str(e)}"
        print(ERROR_MESSAGE)
        print(traceback.format_exc())
except Exception as e:
    ERROR_MESSAGE = f"Fehler beim Import von ChromaDB: {str(e)}"
    print(ERROR_MESSAGE)
    print(traceback.format_exc())

# Konstanten für die ChromaDB-Konfiguration
DB_DIRECTORY = os.path.join(tempfile.gettempdir(), "saalbach_db")
COLLECTION_NAME = "saalbach_knowledge"

class DummyResponse:
    """Fallback-Klasse, wenn ChromaDB nicht verfügbar ist."""
    def __init__(self):
        self.documents = [[]]
        self.metadatas = [[]]
        self.distances = [[]]
        self.ids = [[]]

class ChromaManager:
    """Verwaltet die ChromaDB für das RAG-System des Saalbach-Chatbots."""
    
    def __init__(self, embedding_model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialisiert den ChromaDB Manager.
        
        Args:
            embedding_model_name: Name des zu verwendenden Embedding-Modells
        """
        self.client = None
        self.collection = None
        self.embedding_function = None
        self.is_functional = False
        
        # Falls ChromaDB nicht importiert werden konnte, gebe Warnung aus
        if not CHROMA_INITIALIZED:
            print(f"WARNUNG: ChromaDB konnte nicht initialisiert werden: {ERROR_MESSAGE}")
            print("RAG-Funktionalität ist eingeschränkt.")
            return
        
        try:
            # Sicherstellen, dass das DB-Verzeichnis existiert
            os.makedirs(DB_DIRECTORY, exist_ok=True)
            
            print(f"ChromaDB-Verzeichnis: {DB_DIRECTORY}")
            print(f"Prüfe, ob das Verzeichnis existiert und Schreibrechte vorhanden sind...")
            
            # Teste Schreibrechte im Verzeichnis
            test_file = os.path.join(DB_DIRECTORY, "test_write.txt")
            try:
                with open(test_file, 'w') as f:
                    f.write("Test")
                os.remove(test_file)
                print("Schreibtest erfolgreich!")
            except Exception as e:
                print(f"Schreibtest fehlgeschlagen: {str(e)}")
                # Versuche, ein anderes Verzeichnis zu verwenden
                DB_DIRECTORY = tempfile.mkdtemp(prefix="saalbach_")
                print(f"Verwende alternatives Verzeichnis: {DB_DIRECTORY}")
            
            # ChromaDB Client initialisieren
            print("Initialisiere ChromaDB Client...")
            self.client = chromadb.PersistentClient(path=DB_DIRECTORY)
            print("ChromaDB Client erfolgreich initialisiert!")
            
            # Embedding-Funktion definieren
            print(f"Initialisiere Embedding-Funktion mit Modell: {embedding_model_name}")
            try:
                self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
                    model_name=embedding_model_name
                )
                print("Embedding-Funktion erfolgreich initialisiert!")
            except Exception as e:
                print(f"Fehler bei der Initialisierung der Embedding-Funktion: {str(e)}")
                print("Versuche Default-Embedding-Funktion...")
                # Fallback auf einfachere Embedding-Funktion
                try:
                    self.embedding_function = None  # ChromaDB verwendet intern ein einfaches Modell
                except Exception as e2:
                    print(f"Auch Default-Embedding-Funktion fehlgeschlagen: {str(e2)}")
                    return
            
            # Collection erstellen oder laden
            try:
                print(f"Versuche, Collection '{COLLECTION_NAME}' zu laden...")
                self.collection = self.client.get_collection(
                    name=COLLECTION_NAME,
                    embedding_function=self.embedding_function
                )
                print(f"Collection '{COLLECTION_NAME}' erfolgreich geladen.")
            except Exception as e:
                print(f"Collection nicht gefunden, erstelle neue: {str(e)}")
                try:
                    self.collection = self.client.create_collection(
                        name=COLLECTION_NAME,
                        embedding_function=self.embedding_function
                    )
                    print(f"Collection '{COLLECTION_NAME}' neu erstellt.")
                except Exception as e2:
                    print(f"Konnte Collection nicht erstellen: {str(e2)}")
                    return
            
            # Alles erfolgreich initialisiert
            self.is_functional = True
            print("ChromaManager vollständig initialisiert und funktionsbereit.")
            
        except Exception as e:
            print(f"Fehler bei der Initialisierung des ChromaManagers: {str(e)}")
            print(traceback.format_exc())
    
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
        if not self.is_functional:
            print("ChromaManager ist nicht funktionsbereit. Dokument wird nicht hinzugefügt.")
            return str(uuid.uuid4()) if not doc_id else doc_id
            
        if not doc_id:
            doc_id = str(uuid.uuid4())
            
        try:
            self.collection.add(
                documents=[text],
                metadatas=[metadata],
                ids=[doc_id]
            )
            return doc_id
        except Exception as e:
            print(f"Fehler beim Hinzufügen des Dokuments: {str(e)}")
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
        if not self.is_functional:
            print("ChromaManager ist nicht funktionsbereit. Dokumente werden nicht hinzugefügt.")
            if not ids:
                ids = [str(uuid.uuid4()) for _ in range(len(texts))]
            return ids
            
        if not ids:
            ids = [str(uuid.uuid4()) for _ in range(len(texts))]
            
        try:
            self.collection.add(
                documents=texts,
                metadatas=metadatas,
                ids=ids
            )
            return ids
        except Exception as e:
            print(f"Fehler beim Batch-Hinzufügen von Dokumenten: {str(e)}")
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
        if not self.is_functional:
            print("ChromaManager ist nicht funktionsbereit. Eine leere Antwort wird zurückgegeben.")
            dummy = DummyResponse()
            return {
                "documents": dummy.documents,
                "metadatas": dummy.metadatas,
                "distances": dummy.distances,
                "ids": dummy.ids
            }
            
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results,
                where=filter_criteria
            )
            return results
        except Exception as e:
            print(f"Fehler bei der Suche: {str(e)}")
            dummy = DummyResponse()
            return {
                "documents": dummy.documents,
                "metadatas": dummy.metadatas,
                "distances": dummy.distances,
                "ids": dummy.ids
            }
    
    def update_document(self, doc_id: str, text: str, metadata: Dict[str, Any]) -> None:
        """
        Aktualisiert ein vorhandenes Dokument.
        
        Args:
            doc_id: Die ID des zu aktualisierenden Dokuments
            text: Der neue Text
            metadata: Die neuen Metadaten
        """
        if not self.is_functional:
            print("ChromaManager ist nicht funktionsbereit. Dokument wird nicht aktualisiert.")
            return
            
        try:
            self.collection.update(
                ids=[doc_id],
                documents=[text],
                metadatas=[metadata]
            )
        except Exception as e:
            print(f"Fehler beim Aktualisieren des Dokuments: {str(e)}")
    
    def delete_document(self, doc_id: str) -> None:
        """
        Löscht ein Dokument aus der Datenbank.
        
        Args:
            doc_id: Die ID des zu löschenden Dokuments
        """
        if not self.is_functional:
            print("ChromaManager ist nicht funktionsbereit. Dokument wird nicht gelöscht.")
            return
            
        try:
            self.collection.delete(ids=[doc_id])
        except Exception as e:
            print(f"Fehler beim Löschen des Dokuments: {str(e)}")
    
    def get_document_count(self) -> int:
        """
        Gibt die Anzahl der Dokumente in der Collection zurück.
        
        Returns:
            Anzahl der Dokumente
        """
        if not self.is_functional:
            print("ChromaManager ist nicht funktionsbereit. 0 Dokumente werden zurückgegeben.")
            return 0
            
        try:
            return self.collection.count()
        except Exception as e:
            print(f"Fehler beim Abrufen der Dokumentenanzahl: {str(e)}")
            return 0
        
    def get_all_documents(self) -> Dict[str, Any]:
        """
        Ruft alle Dokumente aus der Collection ab.
        
        Returns:
            Alle Dokumente
        """
        if not self.is_functional:
            print("ChromaManager ist nicht funktionsbereit. Leere Dokumentenliste wird zurückgegeben.")
            dummy = DummyResponse()
            return {
                "documents": dummy.documents,
                "metadatas": dummy.metadatas,
                "ids": dummy.ids
            }
            
        try:
            return self.collection.get()
        except Exception as e:
            print(f"Fehler beim Abrufen aller Dokumente: {str(e)}")
            dummy = DummyResponse()
            return {
                "documents": dummy.documents,
                "metadatas": dummy.metadatas,
                "ids": dummy.ids
            }
