"""
Einfache RAG-Implementierung (Retrieval Augmented Generation) für den Saalbach Tourismus Chatbot.
Diese Implementierung funktioniert ohne ChromaDB und verwendet einfache Textsuche.
Dient als Fallback, wenn ChromaDB nicht funktioniert.
"""

import os
import re
import glob
import openai
from typing import List, Dict, Any, Optional, Union

class SimpleRAG:
    """
    Einfache RAG-Implementierung, die ohne ChromaDB funktioniert.
    Verwendet einfache Textsuche für das Retrieval.
    """
    
    def __init__(self, openai_api_key: str = None, model: str = "gpt-3.5-turbo"):
        """
        Initialisiert das Simple RAG-System.
        
        Args:
            openai_api_key: OpenAI API-Schlüssel
            model: Zu verwendendes OpenAI-Modell
        """
        self.api_key = openai_api_key
        self.model = model
        
        # Wissensquellen laden
        self.knowledge_base = self._load_knowledge_base()
        
        # Base prompt für LLM-Anfragen
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
- Strukturiere lange Antworten klar mit Überschriften, Nummerierungen oder Aufzählungen
- Beende längere Antworten mit einer Frage, um das Gespräch fortzuführen
- Wenn du über Aktivitäten sprichst, erwähne auch immer:
  * Konkrete Orte/Namen (z.B. bestimmte Wanderwege, Hütten, Hotels)
  * Schwierigkeitsgrad oder Eignung (für wen ist es geeignet?)
  * Kleine persönliche Tipps ("Mein Geheimtipp: Bestell dort unbedingt den Kaiserschmarrn!")
  * Praktische Infos (Öffnungszeiten, Preise, besondere Hinweise)

Sprachliche Besonderheiten:
- Verwende gelegentlich österreichische/alpine Ausdrücke (z.B. "a bisserl", "gemütlich", "zünftig")
- Benutze Ausdrücke wie "Der Wahnsinn!", "Echt cool!", "Ein echtes Erlebnis!"
- Beende Nachrichten gerne mit "Servus!", "Bis bald!" oder ähnlichen Grußformeln
"""
    
    def _load_knowledge_base(self) -> List[Dict[str, Any]]:
        """
        Lädt Markdown-Dateien und extrahiert deren Inhalte.
        
        Returns:
            Liste von Dokumenten mit Metadaten
        """
        documents = []
        knowledge_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "knowledge")
        if not os.path.exists(knowledge_dir):
            # Alternativen ausprobieren
            alternative_dir = os.path.join(os.getcwd(), "knowledge")
            if os.path.exists(alternative_dir):
                knowledge_dir = alternative_dir
            else:
                print("Wissensverzeichnis nicht gefunden.")
                return documents
        
        # Alle Markdown-Dateien im Wissensverzeichnis finden
        markdown_files = glob.glob(os.path.join(knowledge_dir, "*.md"))
        
        for file_path in markdown_files:
            try:
                file_name = os.path.basename(file_path)
                theme = os.path.splitext(file_name)[0]
                
                with open(file_path, 'r', encoding='utf-8') as file:
                    content = file.read()
                
                # Dokument in Abschnitte aufteilen
                sections = self._split_into_sections(content)
                
                for section in sections:
                    heading = section.get("heading", "Allgemein")
                    subheading = section.get("subheading", "")
                    text = section.get("text", "")
                    
                    if text.strip():
                        documents.append({
                            "content": text,
                            "metadata": {
                                "theme": theme,
                                "source_file": file_name,
                                "heading": heading,
                                "subheading": subheading
                            }
                        })
            except Exception as e:
                print(f"Fehler beim Laden von {file_path}: {str(e)}")
        
        print(f"{len(documents)} Dokumente aus {len(markdown_files)} Dateien geladen.")
        return documents
    
    def _split_into_sections(self, content: str) -> List[Dict[str, Any]]:
        """
        Teilt einen Markdown-Text in Abschnitte.
        
        Args:
            content: Der Markdown-Inhalt
            
        Returns:
            Liste von Abschnitten mit Überschriften und Text
        """
        sections = []
        current_section = {
            "heading": "Allgemein",
            "subheading": "",
            "text": ""
        }
        
        lines = content.split('\n')
        
        for line in lines:
            # Hauptüberschrift erkennen (# Titel)
            if line.strip().startswith('# '):
                # Vorherigen Abschnitt speichern, wenn vorhanden
                if current_section["text"].strip():
                    sections.append(current_section.copy())
                
                current_section = {
                    "heading": line.lstrip('# ').strip(),
                    "subheading": "",
                    "text": ""
                }
            
            # Unterüberschrift erkennen (## Titel)
            elif line.strip().startswith('## '):
                # Vorherigen Abschnitt speichern, wenn vorhanden
                if current_section["text"].strip():
                    sections.append(current_section.copy())
                
                current_section = {
                    "heading": current_section["heading"],
                    "subheading": line.lstrip('## ').strip(),
                    "text": ""
                }
            
            # Inhalt zum aktuellen Abschnitt hinzufügen
            else:
                current_section["text"] += line + '\n'
        
        # Letzten Abschnitt speichern, wenn vorhanden
        if current_section["text"].strip():
            sections.append(current_section.copy())
        
        return sections
    
    def _simple_search(self, query: str, n_results: int = 3) -> List[Dict[str, Any]]:
        """
        Einfache Textsuche nach relevanten Dokumenten.
        
        Args:
            query: Die Suchanfrage
            n_results: Anzahl der zurückzugebenden Ergebnisse
            
        Returns:
            Liste mit relevanten Dokumenten
        """
        if not self.knowledge_base:
            return []
        
        # Suchanfrage in Schlüsselwörter aufteilen
        keywords = re.findall(r'\w+', query.lower())
        if not keywords:
            return []
        
        results = []
        
        # Für jedes Dokument berechnen, wie viele Schlüsselwörter enthalten sind
        for doc in self.knowledge_base:
            content = doc["content"].lower()
            matches = 0
            
            for keyword in keywords:
                if keyword in content:
                    matches += 1
            
            # Nur Dokumente mit mindestens einem Treffer berücksichtigen
            if matches > 0:
                results.append({
                    "document": doc,
                    "matches": matches
                })
        
        # Nach Anzahl der Treffer sortieren
        results.sort(key=lambda x: x["matches"], reverse=True)
        
        # Nur die gewünschte Anzahl an Ergebnissen zurückgeben
        top_results = results[:n_results]
        
        return [result["document"] for result in top_results]
    
    def answer_query(self, query: str, chat_history: List[Dict[str, str]] = None) -> str:
        """
        Beantwortet eine Benutzeranfrage.
        
        Args:
            query: Die Benutzeranfrage
            chat_history: Optional, bisheriger Chat-Verlauf
            
        Returns:
            Die generierte Antwort
        """
        try:
            print(f"\n--- Neue Anfrage: '{query}' ---")
            
            # Prüfen, ob der API-Key gesetzt ist
            if not self.api_key:
                return "Servus! Ich brauche einen API-Schlüssel, um dir helfen zu können. Bitte gib einen OpenAI API-Schlüssel in den Einstellungen ein. Danke! 😊"
            
            # Relevante Informationen abrufen
            relevant_docs = self._simple_search(query, n_results=3)
            
            # Prompt erstellen
            context = ""
            if relevant_docs:
                context_parts = []
                for i, doc in enumerate(relevant_docs):
                    context_part = f"INFORMATION {i+1} (Thema: {doc['metadata']['theme']}):\n"
                    if doc['metadata']['heading']:
                        context_part += f"Überschrift: {doc['metadata']['heading']}\n"
                    if doc['metadata']['subheading']:
                        context_part += f"Unterüberschrift: {doc['metadata']['subheading']}\n"
                    context_part += f"{doc['content']}\n\n"
                    context_parts.append(context_part)
                
                context = "\n".join(context_parts)
            else:
                context = "Keine spezifischen Informationen verfügbar. Nutze dein eigenes Wissen über die Region Saalbach-Hinterglemm."
            
            system_prompt = f"{self.base_system_prompt}\n\nZUSÄTZLICHE INFORMATIONEN:\n{context}"
            
            # Chat-Verlauf vorbereiten
            messages = [{"role": "system", "content": system_prompt}]
            
            # Chat-Verlauf hinzufügen, falls vorhanden
            if chat_history:
                recent_history = chat_history[-5:] if len(chat_history) > 5 else chat_history
                messages.extend(recent_history)
            
            # Aktuelle Anfrage hinzufügen
            messages.append({"role": "user", "content": query})
            
            print(f"Sende Anfrage an OpenAI ({self.model})...")
            
            # OpenAI-Client konfigurieren
            client = openai.OpenAI(api_key=self.api_key)
            
            # Anfrage an das LLM senden
            response = client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.8,
                max_tokens=1000
            )
            
            # Antwort zurückgeben
            answer = response.choices[0].message.content
            print(f"Antwort erhalten (Länge: {len(answer)} Zeichen)")
            return answer
            
        except Exception as e:
            error_msg = f"Fehler bei der Anfrage an OpenAI: {str(e)}"
            print(error_msg)
            
            # Im Fehlerfall trotzdem eine freundliche, persönliche Antwort geben
            if "API key" in str(e).lower():
                return "Servus! Aktuell hab ich leider ein kleines technisches Problem mit meiner Verbindung. Könntest du es in ein paar Minuten nochmal probieren? Danke für dein Verständnis! 😊"
            elif "quota" in str(e).lower() or "billing" in str(e).lower():
                return "Grüß dich! Leider bin ich gerade ein bisserl überfordert - zu viele Gäste auf einmal! 😅 Kannst du in 5 Minuten nochmal vorbeischauen? Dann kann ich dir sicher weiterhelfen!"
            else:
                return "Servus! Entschuldige bitte, aktuell kann ich deine Anfrage nicht richtig beantworten. Magst du deine Frage vielleicht anders formulieren? Oder frag mich einfach nach konkreten Tipps zu Wandern, Biken, Skifahren oder guten Restaurants in Saalbach-Hinterglemm!"
