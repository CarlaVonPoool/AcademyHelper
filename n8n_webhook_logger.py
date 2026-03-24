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
        
        # Debug: Zeige alle Messages
        print(f"🔍 DEBUG: Aktuelle Frage: '{question[:50]}...'")
        print(f"🔍 DEBUG: Anzahl Messages in Session: {len(st.session_state.get('messages', []))}")
        
        # Hole vorherige Fragen aus Streamlit Session State
        previous_questions = []
        for i, msg in enumerate(st.session_state.get('messages', [])):
            print(f"   Message {i}: {msg.get('role')} - '{msg.get('content', '')[:30]}...'")
            if msg.get('role') == 'user':
                previous_questions.append(msg['content'])
        
        print(f"🔍 DEBUG: Gefundene User-Fragen: {len(previous_questions)}")
        for i, q in enumerate(previous_questions):
            print(f"   Frage {i}: '{q[:30]}...'")
        
        # Entferne die aktuelle Frage aus den vorherigen
        if question in previous_questions:
            previous_questions = [q for q in previous_questions if q != question]
            print(f"🔍 DEBUG: Aktuelle Frage aus vorherigen entfernt")
        
        print(f"🔍 DEBUG: Finale vorherige Fragen: {len(previous_questions)}")
        
        # Sende direkt die aktuelle Frage+Antwort
        self.send_current_qa_direct(interaction_id, session_id, question, answer, sources, confidence_score, previous_questions)
        
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
        
        # Finde die entsprechende Nachricht in den Streamlit Messages
        question = None
        answer = None
        for msg in st.session_state.get('messages', []):
            if msg.get('role') == 'assistant' and msg.get('interaction_id') == interaction_id:
                answer = msg['content']
                # Finde die zugehörige Frage (die Nachricht davor)
                msg_index = st.session_state.messages.index(msg)
                if msg_index > 0 and st.session_state.messages[msg_index-1]['role'] == 'user':
                    question = st.session_state.messages[msg_index-1]['content']
                break
        
        if not question or not answer:
            return
        
        # Hole vorherige Fragen
        previous_questions = []
        for msg in st.session_state.get('messages', []):
            if msg.get('role') == 'user' and msg['content'] != question:
                previous_questions.append(msg['content'])
        
        # Erstelle Feedback-Objekt
        feedback = {
            "timestamp": datetime.now().isoformat(),
            "is_helpful": is_helpful,
            "is_accurate": is_accurate,
            "error_type": error_type,
            "user_comment": user_comment[:500],
            "expected_answer": expected_answer[:500]
        }
        
        # Sende separaten Webhook für Feedback
        self.send_feedback_webhook_direct(interaction_id, session_id, question, answer, feedback, previous_questions)
    
    
    def _send_to_n8n(self, data: Dict):
        """
        Sendet Daten per HTTP POST an N8N Webhook
        """
        try:
            # Direkte Payload für Webhook-Typen
            payload = data
            
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
    
    def send_current_qa_direct(self, interaction_id: str, session_id: str, question: str, answer: str, sources: List[str], confidence_score: float, previous_questions: List[str]):
        """
        Sendet nur die aktuelle Frage+Antwort mit vorherigen Fragen als Kontext
        """
        # Erstelle Payload mit nur der aktuellen Frage+Antwort
        payload = {
            "webhook_type": "Frage+Antwort",
            "session_id": session_id,
            "timestamp": datetime.now().isoformat(),
            "aktuelle_frage": question[:500],
            "aktuelle_antwort": answer[:1000],
            "VorherigeFragen": previous_questions,  # Liste der vorherigen Fragen
            "sources": sources or [],
            "confidence_score": confidence_score,
            "interaction_id": interaction_id
        }
        
        # Debug: Zeige was gesendet wird
        print(f"📤 Sende Q&A: '{question[:50]}...'")
        print(f"   Session: {session_id}")
        print(f"   Interaction ID: {interaction_id}")
        print(f"   Previous Questions: {len(previous_questions)}")
        
        # In Queue einreihen für Background-Sending
        self.send_queue.put(payload)
    
    def send_feedback_webhook_direct(self, interaction_id: str, session_id: str, question: str, answer: str, feedback: Dict, previous_questions: List[str]):
        """
        Sendet separaten Webhook für Feedback mit Frage+Antwort+Feedback
        """
        # Erstelle Payload mit Frage+Antwort+Feedback
        payload = {
            "webhook_type": "Feedback+Frage",
            "session_id": session_id,
            "timestamp": datetime.now().isoformat(),
            "frage": question[:500],
            "antwort": answer[:1000],
            "feedback": feedback,
            "VorherigeFragen": previous_questions,  # Liste der vorherigen Fragen
            "sources": [],
            "confidence_score": 0.0,
            "interaction_id": interaction_id
        }
        
        # Debug: Zeige was gesendet wird
        print(f"📤 Sende Feedback: '{question[:50]}...'")
        print(f"   Feedback: {feedback.get('is_helpful')}")
        
        # In Queue einreihen für Background-Sending
        self.send_queue.put(payload)
    
    def _get_user_agent(self):
        """Holt User Agent falls verfügbar"""
        try:
            # In Streamlit ist das schwer zu bekommen
            return "Streamlit/Unknown"
        except:
            return "Unknown"
    
