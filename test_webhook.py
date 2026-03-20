"""
Test-Script für N8N Webhook
Teste direkt ob dein Webhook funktioniert
"""

import requests
import json
from datetime import datetime

# WICHTIG: Ersetze diese URL mit deiner echten N8N Webhook URL
WEBHOOK_URL = "https://poool.app.n8n.cloud/webhook/4542178d-d986-4298-bfc5-fa28389b4027"

def test_simple_request():
    """Test 1: Einfacher POST Request"""
    print("Test 1: Einfacher POST Request...")
    
    payload = {
        "test": "simple",
        "timestamp": datetime.now().isoformat(),
        "message": "Wenn du das siehst, funktioniert der Webhook!"
    }
    
    try:
        response = requests.post(WEBHOOK_URL, json=payload, timeout=10)
        print(f"✅ Status Code: {response.status_code}")
        print(f"✅ Response: {response.text[:200]}")
        return True
    except Exception as e:
        print(f"❌ Fehler: {e}")
        return False

def test_app_format():
    """Test 2: Format wie die App es sendet"""
    print("\nTest 2: App-Format Test...")
    
    payload = {
        "timestamp": datetime.now().isoformat(),
        "source": "streamlit_academy_helper",
        "session_data": {
            "session_id": "test-123",
            "session_start": datetime.now().isoformat(),
            "send_reason": "test_from_python",
            "interactions": [{
                "id": "test-interaction-1",
                "timestamp": datetime.now().isoformat(),
                "question": "TEST: Wie funktioniert der Webhook?",
                "answer": "TEST: Der Webhook empfängt POST Requests.",
                "sources": ["test.md"],
                "confidence_score": 0.99,
                "feedback": None
            }],
            "statistics": {
                "total_interactions": 1,
                "avg_confidence": 0.99,
                "feedback_received": 0
            }
        },
        "encryption_enabled": False,
        "webhook_version": "2.0"
    }
    
    headers = {
        'Content-Type': 'application/json',
        'User-Agent': 'StreamlitApp/2.0-Test'
    }
    
    try:
        response = requests.post(WEBHOOK_URL, json=payload, headers=headers, timeout=15)
        print(f"✅ Status Code: {response.status_code}")
        print(f"✅ Response: {response.text[:200]}")
        return True
    except Exception as e:
        print(f"❌ Fehler: {e}")
        return False

def test_minimal():
    """Test 3: Minimaler Request"""
    print("\nTest 3: Minimaler Request...")
    
    try:
        response = requests.post(WEBHOOK_URL, json={"test": True}, timeout=5)
        print(f"✅ Status Code: {response.status_code}")
        return True
    except Exception as e:
        print(f"❌ Fehler: {e}")
        return False

if __name__ == "__main__":
    print("🧪 N8N Webhook Test")
    print("=" * 50)
    print(f"Webhook URL: {WEBHOOK_URL}")
    print("=" * 50)
    
    # Führe alle Tests aus
    results = []
    results.append(test_simple_request())
    results.append(test_app_format())
    results.append(test_minimal())
    
    print("\n" + "=" * 50)
    if all(results):
        print("✅ ALLE TESTS ERFOLGREICH!")
        print("Der Webhook funktioniert. Das Problem liegt woanders.")
    else:
        print("❌ EINIGE TESTS FEHLGESCHLAGEN!")
        print("Prüfe ob:")
        print("1. Die Webhook URL korrekt ist")
        print("2. Der N8N Workflow aktiv ist")
        print("3. N8N erreichbar ist")