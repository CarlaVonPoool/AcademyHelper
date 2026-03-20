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
import asyncio
import threading
from queue import Queue
import time
import base64
import hmac
import os

# Optional: Import cryptography nur wenn verfügbar
try:
    from cryptography.fernet import Fernet
    ENCRYPTION_AVAILABLE = True
except ImportError:
    ENCRYPTION_AVAILABLE = False
    print("⚠️ Cryptography library nicht installiert - Verschlüsselung deaktiviert")

class N8NWebhookLogger:
    def __init__(self, n8n_webhook_url: str, api_key: Optional[str] = None):
        """
        N8N Webhook Logger mit Verschlüsselung
        
        Args:
            n8n_webhook_url: Die N8N Webhook URL (HTTPS empfohlen!)
            api_key: Optional API Key für Authentifizierung und Verschlüsselung
        """
        self.webhook_url = n8n_webhook_url
        self.api_key = api_key
        self.session_data = {}  # Sammelt Daten pro Session
        self.send_queue = Queue()  # Für asynchrones Senden
        
        # Verschlüsselungssetup
        self.encryption_enabled = bool(api_key) and ENCRYPTION_AVAILABLE
        if self.encryption_enabled:
            # Generiere Fernet-Key aus API Key
            key_material = hashlib.sha256(api_key.encode()).digest()
            self.cipher = Fernet(base64.urlsafe_b64encode(key_material))
        else:
            self.cipher = None
            
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
    
    def _encrypt_sensitive_data(self, data: Dict) -> Dict:
        """
        Verschlüsselt sensible Daten in der Payload
        """
        if not self.encryption_enabled:
            return data
        
        encrypted_data = data.copy()
        
        # Verschlüssele sensible Felder
        sensitive_fields = ['question', 'answer', 'user_comment', 'expected_answer']
        
        for interaction in encrypted_data.get('interactions', []):
            for field in sensitive_fields:
                if field in interaction and interaction[field]:
                    try:
                        # Verschlüssele den Text
                        encrypted_text = self.cipher.encrypt(interaction[field].encode('utf-8'))
                        interaction[field] = base64.b64encode(encrypted_text).decode('utf-8')
                        interaction[f"{field}_encrypted"] = True
                    except Exception as e:
                        print(f"Verschlüsselungsfehler für {field}: {e}")
                        # Fallback: Feld entfernen statt unverschlüsselt zu senden
                        interaction[field] = "[VERSCHLÜSSELUNGSFEHLER]"
        
        return encrypted_data
    
    def _create_signature(self, payload: Dict) -> str:
        """
        Erstellt HMAC-Signatur für Payload-Integrität
        """
        if not self.api_key:
            return ""
        
        # Erstelle deterministischen String aus Payload
        payload_string = json.dumps(payload, sort_keys=True, separators=(',', ':'))
        
        # HMAC-SHA256 Signatur
        signature = hmac.new(
            self.api_key.encode('utf-8'),
            payload_string.encode('utf-8'), 
            hashlib.sha256
        ).hexdigest()
        
        return signature
    
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
        
        # IMMER sofort nach jeder Frage senden
        # So bekommst du JEDE Frage-Antwort Kombination sofort
        self.send_session_data(session_id, reason="new_question_answered")
        
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
                
                # Bei Feedback: Update senden mit dem aktualisierten Feedback
                self.send_session_data(session_id, reason="feedback_added_to_interaction")
                break
    
    def send_session_data(self, session_id: str = None, reason: str = "session_end"):
        """
        Sendet gesammelte Session-Daten an N8N
        """
        if session_id is None:
            session_id = st.session_state.get('session_id', 'unknown')
        
        if session_id not in self.session_data:
            return
        
        session_data = self.session_data[session_id].copy()
        session_data["session_end"] = datetime.now().isoformat()
        session_data["send_reason"] = reason
        
        # Statistiken hinzufügen
        interactions = session_data["interactions"]
        session_data["statistics"] = {
            "total_interactions": len(interactions),
            "avg_confidence": sum(i.get("confidence_score", 0) for i in interactions) / len(interactions) if interactions else 0,
            "feedback_received": session_data["feedback_count"],
            "helpful_feedback": sum(1 for i in interactions if i.get("feedback") and i["feedback"].get("is_helpful") == True),
            "unhelpful_feedback": sum(1 for i in interactions if i.get("feedback") and i["feedback"].get("is_helpful") == False),
            "accurate_feedback": sum(1 for i in interactions if i.get("feedback") and i["feedback"].get("is_accurate") == True),
            "inaccurate_feedback": sum(1 for i in interactions if i.get("feedback") and i["feedback"].get("is_accurate") == False),
        }
        
        # In Queue einreihen für Background-Sending
        self.send_queue.put(session_data)
        
        # Session-Daten NICHT löschen, außer bei explizitem Ende
        # So können wir Updates senden (z.B. wenn Feedback hinzugefügt wird)
        if reason in ["session_end", "chat_cleared"]:
            del self.session_data[session_id]
    
    def send_all_sessions(self, reason: str = "app_shutdown"):
        """Sendet alle offenen Sessions"""
        for session_id in list(self.session_data.keys()):
            self.send_session_data(session_id, reason=reason)
    
    def _send_to_n8n(self, data: Dict):
        """
        Sendet verschlüsselte Daten per HTTP POST an N8N Webhook
        """
        try:
            # Verschlüssele sensible Daten
            encrypted_session_data = self._encrypt_sensitive_data(data)
            
            # Payload für N8N
            payload = {
                "timestamp": datetime.now().isoformat(),
                "source": "streamlit_academy_helper",
                "session_data": encrypted_session_data,
                "encryption_enabled": self.encryption_enabled,
                "webhook_version": "2.0"
            }
            
            # Erstelle Signatur
            signature = self._create_signature(payload)
            
            headers = {
                'Content-Type': 'application/json',
                'User-Agent': 'StreamlitApp/2.0-Encrypted'
            }
            
            # Sicherheitsheader hinzufügen
            if self.api_key:
                headers['Authorization'] = f'Bearer {self.api_key}'
                headers['X-API-Key'] = self.api_key
                headers['X-Webhook-Signature'] = signature
                headers['X-Timestamp'] = str(int(datetime.now().timestamp()))
            
            # HTTP POST Request über HTTPS
            response = requests.post(
                self.webhook_url,
                json=payload,
                headers=headers,
                timeout=15,  # Längeres Timeout für Verschlüsselung
                verify=True  # SSL Certificate Verification
            )
            
            if response.status_code == 200:
                print(f"✅ Session data sent to N8N: {data['session_id']}")
            else:
                print(f"❌ N8N Webhook failed: {response.status_code} - {response.text}")
                
        except requests.exceptions.Timeout:
            print("⏰ N8N Webhook timeout - data might be lost")
        except requests.exceptions.ConnectionError:
            print("🔌 N8N Webhook connection error - is N8N running?")
        except Exception as e:
            print(f"❌ N8N Webhook error: {e}")
    
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