"""
N8N Webhook Logger für Streamlit
Sendet Feedback-Daten per Webhook an N8N für Verarbeitung
"""

import json
import requests
from datetime import datetime
import hashlib
import streamlit as st
from typing import Optional, Dict, List
import threading
from queue import Queue
import time


class N8NWebhookLogger:
    def __init__(self, n8n_webhook_url: str):
        """
        N8N Webhook Logger
        
        Args:
            n8n_webhook_url: Die N8N Webhook URL
        """
        self.webhook_url = n8n_webhook_url
        self.session_data = {}  # Sammelt Daten pro Session
        self.send_queue = Queue()  # Für asynchrones Senden
        
            
        self._start_background_sender()
        
    def _start_background_sender(self):
        """Startet Background-Thread für Webhook-Sends"""
        def background_sender():
            while True:
                try:
                    if not self.send_queue.empty():
                        data = self.send_queue.get()
                        self._send_to_n8n(data)
                        self.send_queue.task_done()
                    time.sleep(1)  # 1 Sekunde warten
                except Exception as e:
                    print(f"Background sender error: {e}")
        
        # Daemon Thread - stirbt mit der App
        thread = threading.Thread(target=background_sender, daemon=True)
        thread.start()
    
    
    
    def log_interaction(self, 
                       question: str,
                       answer: str,
                       sources: List[str] = None,
                       context_snippet: str = "",
                       confidence_score: float = 0.0) -> str:
        """
        Loggt eine Frage-Antwort-Interaktion in der Session
        """
        # Erstelle eindeutige Interaction ID
        timestamp = datetime.now().isoformat()
        interaction_id = hashlib.md5(f"{timestamp}{question}".encode()).hexdigest()[:12]
        
        session_id = st.session_state.get('session_id', 'unknown')
        
        # Speichere in Session-Data
        if session_id not in self.session_data:
            self.session_data[session_id] = {
                "session_id": session_id,
                "session_start": timestamp,
                "interactions": [],
                "feedback_count": 0,
                "metadata": {
                    "app_version": "1.0",
                    "user_agent": self._get_user_agent()
                }
            }
        
        interaction = {
            "id": interaction_id,
            "timestamp": timestamp,
            "question": question[:500],  # Limitiere Länge
            "answer": answer[:1000],
            "sources": sources or [],
            "context_snippet": context_snippet[:300],
            "confidence_score": confidence_score,
            "feedback": None  # Wird später ergänzt
        }
        
        self.session_data[session_id]["interactions"].append(interaction)
        
        # Hole vorherige Fragen für die Session (ohne die aktuelle)
        previous_questions = []
        for i in self.session_data[session_id]["interactions"][:-1]:  # Alle außer der letzten
            previous_questions.append(i["question"])
        
        # Sende NUR die aktuelle Frage+Antwort mit vorherigen Fragen als Kontext
        self.send_current_qa(interaction_id, session_id, previous_questions)
        
        return interaction_id
    
    def add_feedback(self,
                    interaction_id: str,
                    is_helpful: Optional[bool] = None,
                    is_accurate: Optional[bool] = None,
                    error_type: str = "",
                    user_comment: str = "",
                    expected_answer: str = ""):
        """
        Fügt Feedback zu einer Interaktion hinzu
        """
        session_id = st.session_state.get('session_id', 'unknown')
        
        if session_id not in self.session_data:
            return
        
        # Finde die richtige Interaktion
        for interaction in self.session_data[session_id]["interactions"]:
            if interaction["id"] == interaction_id:
                interaction["feedback"] = {
                    "timestamp": datetime.now().isoformat(),
                    "is_helpful": is_helpful,
                    "is_accurate": is_accurate,
                    "error_type": error_type,
                    "user_comment": user_comment[:500],
                    "expected_answer": expected_answer[:500]
                }
                self.session_data[session_id]["feedback_count"] += 1
                
                # Hole vorherige Fragen für die Session
                previous_questions = []
                for i in self.session_data[session_id]["interactions"]:
                    if i["id"] != interaction_id:  # Alle außer der aktuellen
                        previous_questions.append(i["question"])
                
                # Sende separaten Webhook für Feedback
                self.send_feedback_webhook(interaction, session_id, previous_questions)
                break
    
    
    def _send_to_n8n(self, data: Dict):
        """
        Sendet Daten per HTTP POST an N8N Webhook
        """
        try:
            # Prüfe ob es sich um neue Webhook-Typen handelt
            if "webhook_type" in data:
                # Direkte Payload für neue Webhook-Typen
                payload = data
            else:
                # Alte Session-Daten (fallback)
                payload = {
                    "timestamp": datetime.now().isoformat(),
                    "source": "streamlit_academy_helper",
                    "session_data": data,
                    "webhook_version": "2.0"
                }
            
            headers = {
                'Content-Type': 'application/json',
                'User-Agent': 'StreamlitApp/2.0'
            }
            
            # HTTP POST Request
            response = requests.post(
                self.webhook_url,
                json=payload,
                headers=headers,
                timeout=10
            )
            
            if response.status_code != 200:
                print(f"❌ N8N Webhook failed: {response.status_code}")
                
        except requests.exceptions.Timeout:
            print(f"⏰ N8N Webhook timeout")
        except requests.exceptions.ConnectionError:
            print(f"🔌 N8N Webhook connection error")
        except Exception as e:
            print(f"❌ N8N Webhook error: {type(e).__name__}")
    
    def send_current_qa(self, interaction_id: str, session_id: str, previous_questions: List[str]):
        """
        Sendet nur die aktuelle Frage+Antwort mit vorherigen Fragen als Kontext
        """
        if session_id not in self.session_data:
            return
        
        # Finde die aktuelle Interaktion
        current_interaction = None
        for interaction in self.session_data[session_id]["interactions"]:
            if interaction["id"] == interaction_id:
                current_interaction = interaction
                break
        
        if not current_interaction:
            return
        
        # Erstelle Payload mit nur der aktuellen Frage+Antwort
        payload = {
            "webhook_type": "Frage+Antwort",
            "session_id": session_id,
            "timestamp": datetime.now().isoformat(),
            "aktuelle_frage": current_interaction["question"],
            "aktuelle_antwort": current_interaction["answer"],
            "VorherigeFragen": previous_questions,  # Liste der vorherigen Fragen
            "sources": current_interaction.get("sources", []),
            "confidence_score": current_interaction.get("confidence_score", 0.0),
            "interaction_id": interaction_id
        }
        
        # In Queue einreihen für Background-Sending
        self.send_queue.put(payload)
    
    def send_feedback_webhook(self, interaction: Dict, session_id: str, previous_questions: List[str]):
        """
        Sendet separaten Webhook für Feedback mit Frage+Antwort+Feedback
        """
        if not interaction.get("feedback"):
            return
        
        # Erstelle Payload mit Frage+Antwort+Feedback
        payload = {
            "webhook_type": "Feedback+Frage",
            "session_id": session_id,
            "timestamp": datetime.now().isoformat(),
            "frage": interaction["question"],
            "antwort": interaction["answer"],
            "feedback": interaction["feedback"],
            "VorherigeFragen": previous_questions,  # Liste der vorherigen Fragen
            "sources": interaction.get("sources", []),
            "confidence_score": interaction.get("confidence_score", 0.0),
            "interaction_id": interaction["id"]
        }
        
        # In Queue einreihen für Background-Sending
        self.send_queue.put(payload)
    
    def _get_user_agent(self):
        """Holt User Agent falls verfügbar"""
        try:
            # In Streamlit ist das schwer zu bekommen
            return "Streamlit/Unknown"
        except:
            return "Unknown"
    
    def get_session_stats(self, session_id: str = None):
        """Holt aktuelle Session-Statistiken"""
        if session_id is None:
            session_id = st.session_state.get('session_id', 'unknown')
            
        if session_id not in self.session_data:
            return {"interactions": 0, "feedback": 0}
        
        data = self.session_data[session_id]
        return {
            "interactions": len(data["interactions"]),
            "feedback_count": data["feedback_count"],
            "session_start": data["session_start"]
        }