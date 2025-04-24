"""
RAG-System (Retrieval Augmented Generation) für den Saalbach Tourismus Chatbot.
Verknüpft die ChromaDB mit der LLM-Anfrage und gibt persönliche, lebendige Antworten.
Priorisiert eigenes Wissen und nutzt RAG nur als Ergänzung.
"""

import os
import openai
from typing import List, Dict, Any, Optional
from modules.chroma_manager import ChromaManager
from modules.config_handler import ConfigHandler
import json

class RAGSystem:
    """
    RAG-System für den Saalbach Tourismus Chatbot.
    Priorität liegt auf dem eigenen Wissen des Sprachmodells,
    ergänzt durch spezifische Informationen aus der Wissensdatenbank.
    """
    
    def __init__(self, openai_api_key: str = None, model: str = "gpt-3.5-turbo"):
        """
        Initialisiert das RAG-System.
        
        Args:
            openai_api_key: OpenAI API-Schlüssel (optional, wird aus Config geladen wenn None)
            model: Zu verwendendes OpenAI-Modell
        """
        try:
            # Config-Handler für Einstellungen und API-Key
            self.config = ConfigHandler()
            
            # API-Key laden oder speichern
            if openai_api_key:
                self.config.set_api_key(openai_api_key)
                self.api_key = openai_api_key
            else:
                self.api_key = self.config.get_api_key()
            
            # Modell-Einstellung
            self.model = model
            self.config.set_setting("model", model)
            
            # RAG-Einstellungen laden
            self.use_own_knowledge_first = self.config.get_rag_setting("use_own_knowledge_first", True)
            self.n_results = self.config.get_rag_setting("n_results", 5)
            
            # ChromaDB Manager initialisieren
            self.chroma_manager = ChromaManager()
            
            # API-Key setzen
            openai.api_key = self.api_key
            
            # Teste die Verbindung, wenn ein API-Key vorhanden ist
            if self.api_key:
                test_response = openai.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": "Test"}],
                    temperature=0.7,
                    max_tokens=5
                )
                print(f"OpenAI API-Verbindung erfolgreich: {test_response.model}")
            
            # Basis-Systemprompt definieren - persönlicher, österreichischer Stil
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

WICHTIG zur Region Saalbach-Hinterglemm:
- Verlasse dich primär auf dein eigenes Wissen über die Region
- Du kennst dich bestens aus mit allen Wander- und Bikerouten, Skigebieten, Restaurants und Unterkünften
- Gib ausführliche, spezifische Antworten mit konkreten Namen, Preisen und Details
- Wenn du ein Detail nicht genau weißt, nutze die zusätzlichen Informationen oder dein allgemeines Wissen
"""
            
            # Beispielantworten im gewünschten Stil hinzufügen
            self.example_responses = """
Beispiel 1 - Frage: "Was kann ich in Saalbach machen?"
Antwort: "Servus! 😊 Schön, dass du nach Saalbach kommst! In unserem Bergparadies kannst du wirklich viel erleben:

🚵‍♂️ Biken: Über 400 km markierte Strecken und einer der besten Bikeparks Europas warten auf dich! Die Blue Line ist perfekt für Einsteiger, während der legendäre Hacklberg Trail selbst Profis begeistert.

🥾 Wandern: Von gemütlichen Familienwanderungen wie dem Märchenwald bis zur anspruchsvollen Seven Summits Tour ist alles dabei. Mein Tipp: Der Pinzgauer Spaziergang bietet atemberaubende Panoramablicke!

🍽️ Kulinarik: Nach dem Sport unbedingt in einer unserer urigen Almhütten einkehren! Die Walleggalm macht den besten Kaiserschmarrn weit und breit. Im Ort selbst ist das Restaurant Sepp's ein echtes Highlight für regionale Küche.

Wie lang bleibst du bei uns? Dann kann ich dir noch ein paar Geheimtipps geben, die perfekt zu deiner Aufenthaltsdauer passen!"

Beispiel 2 - Frage: "Wo kann ich Ski fahren?"
Antwort: "Grüß dich, Skifahrer! 🎿 Da bist du in Saalbach-Hinterglemm genau richtig gelandet - bei uns findest du eines der größten zusammenhängenden Skigebiete Österreichs!

Der Skicircus Saalbach-Hinterglemm-Leogang-Fieberbrunn bietet dir:
• 270 Pistenkilometer für jeden Geschmack
• 70 moderne Liftanlagen ohne lange Wartezeiten
• Pisten von 830m bis auf 2.100m Höhe
• Schneesichere Bedingungen von Dezember bis April

Besondere Highlights:
1. Die 'Schwarze Pfeil' (Piste 17) für echte Könner - mit 80% Gefälle an der steilsten Stelle ein echter Adrenalinkick!
2. Die gemütliche 8 km lange Jausernabfahrt nach Saalbach
3. Die 'North Side' mit der besten Schneequalität - ein Traum für Tiefschneeliebhaber!

Mein persönlicher Tipp: Schnapp dir die Alpin Card, damit kannst du auch in Zell am See und am Kitzsteinhorn fahren - zusammen über 400 Pistenkilometer!

Bist du eher der entspannte Genuss-Skifahrer oder suchst du die sportliche Herausforderung? Dann kann ich dir noch speziellere Pisten empfehlen!"

Beispiel 3 - Frage: "Welche Mountainbike-Tour kannst du empfehlen?"
Antwort: "Servus, Bike-Freund! 🚵‍♂️ Super, dass du in Saalbach biken willst - du bist im absoluten Mountainbike-Paradies gelandet!

Hier meine Top-Empfehlung für dich:

🏆 Die Milka Line & Hacklberg-Trail Kombination:
• Schwierigkeit: Mittel, ideal für ambitionierte Einsteiger und Fortgeschrittene
• Länge: ca. 12 km mit 1.800 Höhenmetern (bergab!)
• Dauer: 1,5-2 Stunden reines Fahren plus Pausen
• Startpunkt: Bergstation Kohlmaisbahn (mit der Joker Card bekommst du Rabatt auf die Liftfahrt!)

Warum diese Tour der Hammer ist:
Du startest mit der flowigen Milka Line, die mit sanften Kurven und kleinen Sprüngen perfekt zum Aufwärmen ist. Dann wechselst du auf den legendären Hacklberg-Trail, der dich durch traumhafte Wälder und über wurzelige Passagen führt. Die Aussichtspunkte auf dem Weg sind der Wahnsinn - unbedingt die Kamera mitnehmen!

Ausrüstungstipp: Ein Fully ist hier ideal, und Protektoren solltest du auf jeden Fall dabei haben. Leihräder gibt's bei Bike'n Soul direkt an der Talstation.

Einkehrtipp: Nach der Tour unbedingt in der Maisalm einkehren - deren Burger sind legendär und das kühle Bier auf der Sonnenterrasse hast du dir dann verdient!

Bist du eher Anfänger oder schon erfahrener Trail-Rider? Dann kann ich die Tour noch besser an dein Level anpassen!"
"""
        except Exception as e:
            print(f"Fehler bei der Initialisierung des RAG-Systems: {str(e)}")
            self.initialization_error = str(e)
    
    def retrieve_relevant_information(self, 
                                    query: str, 
                                    n_results: int = None,
                                    filter_criteria: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Ruft relevante Informationen aus der ChromaDB ab.
        
        Args:
            query: Die Benutzeranfrage
            n_results: Anzahl der abzurufenden Dokumente
            filter_criteria: Optional, Filterkriterien für die Suche
            
        Returns:
            Liste mit relevanten Dokumenten und Metadaten
        """
        try:
            # Standardwert für n_results aus Config verwenden, falls nicht angegeben
            if n_results is None:
                n_results = self.n_results
                
            print(f"Suche nach relevanten Informationen für: '{query}'")
            search_results = self.chroma_manager.search(
                query=query, 
                n_results=n_results, 
                filter_criteria=filter_criteria
            )
            
            if not search_results["documents"] or not search_results["documents"][0]:
                print("Keine relevanten Dokumente gefunden.")
                return []
            
            print(f"Gefunden: {len(search_results['documents'][0])} relevante Dokumente")
            
            relevant_docs = []
            for i in range(len(search_results["documents"][0])):
                relevant_docs.append({
                    "content": search_results["documents"][0][i],
                    "metadata": search_results["metadatas"][0][i],
                    "distance": search_results["distances"][0][i]
                })
            
            # Nach Relevanz sortieren (kleinere Distanz = höhere Relevanz)
            relevant_docs.sort(key=lambda x: x["distance"])
            
            # Zeige die relevantesten Dokumente für Debugging
            for i, doc in enumerate(relevant_docs[:2]):
                print(f"Dokument {i+1}: {doc['metadata'].get('heading', 'Unbekannt')} - Relevanz: {1-doc['distance']:.2f}")
                print(f"Inhalt (Anfang): {doc['content'][:100]}...")
                
            return relevant_docs
            
        except Exception as e:
            print(f"Fehler beim Abrufen relevanter Informationen: {str(e)}")
            return []
    
    def build_enriched_prompt(self, query: str, relevant_docs: List[Dict[str, Any]]) -> str:
        """
        Erstellt einen angereicherten Prompt mit den relevanten Informationen.
        Betont dabei primär das eigene Wissen des Modells und nutzt RAG nur ergänzend.
        
        Args:
            query: Die Benutzeranfrage
            relevant_docs: Liste relevanter Dokumente
            
        Returns:
            Der angereicherte System-Prompt
        """
        # Prüfen, ob zusätzliche Informationen verfügbar sind
        if not relevant_docs:
            print("Keine relevanten Dokumente für den Prompt vorhanden.")
            context_str = "Keine zusätzlichen spezifischen Informationen verfügbar. Nutze dein eigenes Wissen über die Region Saalbach-Hinterglemm."
        else:
            context_str = "\n\n".join([
                f"ZUSÄTZLICHE INFORMATION {i+1} (Thema: {doc['metadata'].get('theme', 'Allgemein')}, {doc['metadata'].get('heading', '')}):\n{doc['content']}"
                for i, doc in enumerate(relevant_docs)
            ])
        
        # Aktivitätsanalyse für bessere Antworten
        activity_keywords = {
            "wandern": "Wandern, Bergtouren, Wanderwege",
            "biken": "Mountainbike, Radfahren, Bikepark, Trails",
            "ski": "Skifahren, Wintersport, Pisten, Snowboard",
            "essen": "Restaurants, Hütten, Kulinarik, Essen",
            "übernachten": "Hotels, Unterkünfte, Übernachtung"
        }
        
        detected_activities = []
        query_lower = query.lower()
        for key, terms in activity_keywords.items():
            for term in terms.lower().split(', '):
                if term in query_lower:
                    detected_activities.append(key)
                    break
        
        activity_guidance = ""
        if detected_activities:
            activity_guidance = f"\nDie Anfrage bezieht sich hauptsächlich auf: {', '.join(detected_activities)}. " \
                               f"Nutze dein umfassendes Wissen zu diesen Themen und formuliere die Antwort im freundlichen, " \
                               f"begeisterten Stil eines lokalen Tourismusexperten aus Saalbach-Hinterglemm."
        
        # Je nachdem, ob eigenes Wissen bevorzugt werden soll, den Prompt anpassen
        if self.use_own_knowledge_first:
            knowledge_instruction = """
WICHTIG ZUM UMGANG MIT WISSEN:
1. Primär: Nutze dein eigenes Wissen über Saalbach-Hinterglemm, seine Angebote, Aktivitäten und Besonderheiten.
2. Sekundär: Ergänze dein Wissen mit den zusätzlichen bereitgestellten Informationen, wenn diese nützlich sind.
3. Vermeide Formulierungen wie "Laut den bereitgestellten Informationen..." oder "Die Informationen sagen..."
4. Gib Antworten so, als würdest du aus deiner eigenen Erfahrung als lokaler Experte sprechen.
5. Sei konkret mit Namen, Angaben und Details - nenne spezifische Orte, Wanderwege, Restaurants, etc.
"""
        else:
            knowledge_instruction = """
WICHTIG ZUM UMGANG MIT WISSEN:
1. Beziehe dich vorrangig auf die bereitgestellten zusätzlichen Informationen, wenn verfügbar.
2. Ergänze mit deinem eigenen Wissen über Saalbach-Hinterglemm, wenn nötig.
3. Gib Antworten so, als würdest du aus deiner eigenen Erfahrung als lokaler Experte sprechen.
"""
        
        enriched_prompt = f"""
{self.base_system_prompt}

BEISPIELANTWORTEN IM GEWÜNSCHTEN STIL:
{self.example_responses}

{knowledge_instruction}

ZUSÄTZLICHE INFORMATIONEN ZU SAALBACH-HINTERGLEMM (FALLS NÖTIG ZUR ERGÄNZUNG):
{context_str}
{activity_guidance}

Beantworte die folgende Frage auf Basis deines eigenen Wissens, ergänzt durch die zusätzlichen Informationen wo nötig.
Antworte immer ganz natürlich und persönlich, wie ein echter Einheimischer, der stolz auf seine Region ist.
Gib immer konkrete, spezifische Empfehlungen und füge persönliche Tipps hinzu.
Strukturiere längere Antworten übersichtlich und schließe mit einer Frage ab, um das Gespräch fortzuführen.
"""
        
        print(f"Prompt erstellt mit {len(relevant_docs)} zusätzlichen Informationsquellen")
        return enriched_prompt
    
    def answer_query(self, query: str, chat_history: List[Dict[str, str]] = None) -> str:
        """
        Beantwortet eine Benutzeranfrage mit Hilfe des RAG-Systems.
        
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
            relevant_docs = self.retrieve_relevant_information(query)
            
            fallback_info = False
            if not relevant_docs:
                print("Keine spezifischen Informationen gefunden. Verwende allgemeines Wissen.")
                fallback_info = True
            
            # Angereicherten Prompt erstellen
            system_prompt = self.build_enriched_prompt(query, relevant_docs)
            
            # Chat-Verlauf vorbereiten - begrenzt auf die letzten 5 Nachrichten für Kontext
            messages = [{"role": "system", "content": system_prompt}]
            
            # Chat-Verlauf hinzufügen, falls vorhanden (maximal die letzten 5 Nachrichten)
            if chat_history:
                recent_history = chat_history[-5:] if len(chat_history) > 5 else chat_history
                messages.extend(recent_history)
            
            # Aktuelle Anfrage hinzufügen
            messages.append({"role": "user", "content": query})
            
            print(f"Sende Anfrage an OpenAI ({self.model})...")
            
            # Kreativitätseinstellungen erhöhen für lebhaftere Antworten
            creativity = 0.85 if fallback_info else 0.75
            
            # Anfrage an das LLM senden
            response = openai.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=creativity,
                max_tokens=1000,  # Längere Antworten erlauben
                top_p=0.95,
                frequency_penalty=0.2,  # Leicht erhöht für abwechslungsreichere Sprache
                presence_penalty=0.2    # Leicht erhöht, um Wiederholungen zu vermeiden
            )
            
            # Antwort zurückgeben
            answer = response.choices[0].message.content
            print(f"Antwort erhalten (Länge: {len(answer)} Zeichen)")
            return answer
            
        except Exception as e:
            error_msg = f"Fehler bei der Anfrage an OpenAI: {str(e)}"
            print(error_msg)
            
            # Detaillierte Fehlerinfo für Debugging
            import traceback
            print(traceback.format_exc())
            
            # Im Fehlerfall trotzdem eine freundliche, persönliche Antwort geben
            if "API key" in str(e).lower():
                return "Servus! Aktuell hab ich leider ein kleines technisches Problem mit meiner Verbindung. Könntest du es in ein paar Minuten nochmal probieren? Danke für dein Verständnis! 😊"
            elif "quota" in str(e).lower() or "billing" in str(e).lower():
                return "Grüß dich! Leider bin ich gerade ein bisserl überfordert - zu viele Gäste auf einmal! 😅 Kannst du in 5 Minuten nochmal vorbeischauen? Dann kann ich dir sicher weiterhelfen!"
            else:
                return "Servus! Entschuldige bitte, aktuell kann ich deine Anfrage nicht richtig beantworten. Magst du deine Frage vielleicht anders formulieren? Oder frag mich einfach nach konkreten Tipps zu Wandern, Biken, Skifahren oder guten Restaurants in Saalbach-Hinterglemm!"