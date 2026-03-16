import streamlit as st
import os
import glob
import re
import zipfile
import io
from typing import List
from dotenv import load_dotenv
import anthropic
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pickle

load_dotenv()

# Passwort-Authentifizierung
def check_password():
    """Überprüft das Passwort und zeigt Login-Form an."""
    
    # Passwort aus Umgebungsvariablen laden
    correct_password = os.environ.get("APP_PASSWORD", "please-set-password-in-secrets")
    
    if st.session_state.authenticated:
        return True
    
    st.title("🔐 RAG Chat Assistant - Login")
    st.markdown("**Bitte gib das Passwort ein, um fortzufahren:**")
    
    with st.form("password_form"):
        password = st.text_input("Passwort:", type="password", help="Kontaktiere den Administrator für das Passwort")
        submit = st.form_submit_button("🚀 Anmelden")
        
        if submit:
            if password == correct_password:
                st.session_state.authenticated = True
                st.success("✅ Erfolgreich angemeldet!")
                st.rerun()
            else:
                st.error("❌ Falsches Passwort! Bitte versuche es erneut.")
                st.stop()
    
    # Info-Box für Benutzer
    with st.expander("ℹ️ Über diese App"):
        st.markdown("""
        **RAG Chat Assistant** ist ein intelligenter Chat-Assistent mit Dokumenten-basierter Suche.
        
        **Features:**
        - 🔍 Intelligente Dokumentensuche
        - 💬 Chat-basierte Q&A
        - 📁 Dokument-Upload (einzelne Dateien oder ZIP)
        - 🔧 Erweiterte Suchfunktionen
        
        **Benötigst du Zugang?** Kontaktiere den Administrator.
        """)
    
    st.stop()

# Globale Variablen
if 'document_store' not in st.session_state:
    st.session_state.document_store = None
if 'embeddings' not in st.session_state:
    st.session_state.embeddings = None
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False

def simple_preprocessing(text: str) -> str:
    """Einfache Text-Vorverarbeitung."""
    text = text.lower()
    text = re.sub(r'[^\w\säöüß]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def extract_simple_keywords(text: str) -> List[str]:
    """Einfache Keyword-Extraktion."""
    words = simple_preprocessing(text).split()
    
    stopwords = {
        'der', 'die', 'und', 'in', 'den', 'von', 'zu', 'das', 'mit', 'sich', 'des', 'auf', 'für',
        'ist', 'im', 'dem', 'nicht', 'ein', 'eine', 'als', 'auch', 'es', 'an', 'werden', 'aus',
        'er', 'hat', 'dass', 'sie', 'nach', 'wird', 'bei', 'einer', 'um', 'am', 'sind', 'noch',
        'wie', 'einem', 'über', 'einen', 'so', 'zum', 'war', 'haben', 'nur', 'oder', 'aber',
        'vor', 'zur', 'bis', 'mehr', 'durch', 'man', 'sein', 'wurde', 'sei', 'in', 'wenn'
    }
    
    keywords = [word for word in words if len(word) > 3 and word not in stopwords]
    return list(set(keywords))[:10]

def chunk_text(text: str, chunk_size: int = 1500, overlap: int = 400) -> List[str]:
    """Teilt Text in Chunks auf."""
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        
        # Versuche bei Satzende zu teilen
        if end < len(text):
            for i in range(end, max(start + overlap, end - 200), -1):
                if text[i] in '.!?\n':
                    end = i + 1
                    break
        
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        
        start = end - overlap
        if start >= len(text):
            break
    
    return chunks

def load_markdown_files(docs_dir: str) -> List[dict]:
    """Lädt Markdown-Dateien aus einem Verzeichnis."""
    documents = []
    
    for file_path in glob.glob(os.path.join(docs_dir, "**/*.md"), recursive=True):
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
                
                # Text in Chunks aufteilen
                chunks = chunk_text(content)
                
                for i, chunk in enumerate(chunks):
                    documents.append({
                        'content': chunk,
                        'processed_content': simple_preprocessing(chunk),
                        'source': file_path,
                        'chunk_id': i,
                        'title': os.path.basename(file_path).replace('.md', '')
                    })
        except Exception as e:
            st.warning(f"Fehler beim Laden von {file_path}: {e}")
    
    return documents

@st.cache_resource
def load_embedding_model():
    """Lädt das Embedding-Modell."""
    return SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

def create_embeddings(documents: List[dict]) -> tuple:
    """Erstellt Embeddings für alle Dokumente."""
    model = load_embedding_model()
    
    # Erstelle Embeddings für Original- und verarbeiteten Text
    texts = [doc['content'] for doc in documents]
    processed_texts = [doc['processed_content'] for doc in documents]
    
    st.write(f"Erstelle Embeddings für {len(texts)} Dokument-Chunks...")
    
    original_embeddings = model.encode(texts, show_progress_bar=True)
    processed_embeddings = model.encode(processed_texts, show_progress_bar=True)
    
    return original_embeddings, processed_embeddings

def search_documents(query: str, top_k: int = 40) -> List[dict]:
    """Sucht in den Dokumenten."""
    if st.session_state.document_store is None or st.session_state.embeddings is None:
        return []
    
    model = load_embedding_model()
    
    # Verschiedene Query-Varianten
    queries = [query]
    queries.append(simple_preprocessing(query))
    
    keywords = extract_simple_keywords(query)
    if keywords:
        queries.append(" ".join(keywords[:5]))
    
    all_results = []
    seen_docs = set()
    
    for q in queries:
        query_embedding = model.encode([q])
        
        # Suche in Original- und verarbeiteten Embeddings
        orig_similarities = cosine_similarity(query_embedding, st.session_state.embeddings[0])[0]
        proc_similarities = cosine_similarity(query_embedding, st.session_state.embeddings[1])[0]
        
        # Kombiniere die Scores
        combined_similarities = (orig_similarities + proc_similarities) / 2
        
        # Top Ergebnisse für diese Query
        top_indices = np.argsort(combined_similarities)[::-1][:top_k]
        
        for idx in top_indices:
            doc_hash = hash(st.session_state.document_store[idx]['content'][:100])
            if doc_hash not in seen_docs:
                result = st.session_state.document_store[idx].copy()
                result['score'] = float(combined_similarities[idx])
                all_results.append(result)
                seen_docs.add(doc_hash)
    
    # Sortiere nach Score
    all_results.sort(key=lambda x: x['score'], reverse=True)
    return all_results[:top_k]

def process_uploaded_files(uploaded_files, source_type="Upload"):
    """Verarbeitet hochgeladene Dateien zu Dokumenten."""
    documents = []
    with st.spinner(f"Verarbeite {len(uploaded_files)} Dateien..."):
        try:
            for uploaded_file in uploaded_files:
                content = uploaded_file.read().decode('utf-8')
                chunks = chunk_text(content)
                
                for i, chunk in enumerate(chunks):
                    documents.append({
                        'content': chunk,
                        'processed_content': simple_preprocessing(chunk),
                        'source': uploaded_file.name,
                        'chunk_id': i,
                        'title': uploaded_file.name.replace('.md', '')
                    })
            
            st.write(f"✅ {len(uploaded_files)} Dateien verarbeitet → {len(documents)} Chunks erstellt")
            return documents
            
        except Exception as e:
            st.error(f"Fehler beim Verarbeiten der Dateien: {e}")
            return []

def process_zip_file(uploaded_zip):
    """Verarbeitet eine ZIP-Datei mit Markdown-Dateien."""
    documents = []
    with st.spinner("Extrahiere und verarbeite ZIP-Datei..."):
        try:
            with zipfile.ZipFile(io.BytesIO(uploaded_zip.read()), 'r') as zip_file:
                md_files = [f for f in zip_file.namelist() if f.endswith('.md') and not f.startswith('__MACOSX')]
                
                if not md_files:
                    st.error("Keine .md Dateien in der ZIP-Datei gefunden!")
                    return []
                
                st.write(f"📁 Gefundene .md Dateien in ZIP: {len(md_files)}")
                
                for file_path in md_files:
                    try:
                        with zip_file.open(file_path) as file:
                            content = file.read().decode('utf-8')
                            chunks = chunk_text(content)
                            
                            # Verwende nur den Dateinamen, nicht den ganzen Pfad
                            filename = os.path.basename(file_path)
                            
                            for i, chunk in enumerate(chunks):
                                documents.append({
                                    'content': chunk,
                                    'processed_content': simple_preprocessing(chunk),
                                    'source': file_path,  # Vollständiger Pfad für Referenz
                                    'chunk_id': i,
                                    'title': filename.replace('.md', '')
                                })
                    except Exception as e:
                        st.warning(f"Fehler bei Datei {file_path}: {e}")
                
                st.write(f"✅ ZIP verarbeitet → {len(documents)} Chunks erstellt")
                return documents
                
        except Exception as e:
            st.error(f"Fehler beim Verarbeiten der ZIP-Datei: {e}")
            return []

def finalize_document_processing(documents):
    """Finalisiert die Dokumentverarbeitung mit Embeddings."""
    try:
        with st.spinner("Erstelle Embeddings..."):
            embeddings = create_embeddings(documents)
        
        st.session_state.document_store = documents
        st.session_state.embeddings = embeddings
        st.success(f"🎉 Index erfolgreich erstellt aus {len(documents)} Text-Stücken!")
        
        # Zeige Statistiken
        sources = set(doc['title'] for doc in documents)
        st.info(f"📊 Verarbeitet: {len(sources)} Dokumente mit {len(documents)} Chunks")
        
    except Exception as e:
        st.error(f"Fehler beim Erstellen der Embeddings: {e}")

def answer_question(question: str) -> str:
    """Beantwortet eine Frage."""
    if st.session_state.document_store is None:
        return "Index wurde noch nicht initialisiert. Bitte lade zuerst Dokumente."
    
    # Suche relevante Dokumente
    results = search_documents(question)
    
    if not results:
        return "Keine relevanten Dokumente gefunden."
    
    # Gruppiere nach Quellen
    sources = set()
    source_to_chunks = {}
    
    for doc in results:
        source = doc['source']
        title = doc['title']
        sources.add(source)
        
        if title not in source_to_chunks:
            source_to_chunks[title] = []
        source_to_chunks[title].append(doc['content'])
    
    # Erstelle Kontext
    context_parts = []
    for title, chunks in source_to_chunks.items():
        combined_content = "\n\n".join(chunks)
        context_parts.append(f"=== Dokument: {title} ===\n{combined_content}")
    
    context = "\n\n".join(context_parts)
    source_list = "\n".join([f"- {os.path.basename(s).replace('.md', '')}" for s in sources])
    
    # Claude API aufrufen
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return "Fehler: ANTHROPIC_API_KEY Umgebungsvariable ist nicht gesetzt."
    
    client = anthropic.Anthropic(api_key=api_key)
    
    system_prompt = """Du bist ein Experten-Assistent. Analysiere ALLE bereitgestellten Dokumente gründlich.

ANALYSEPROZESS:
1. Lies ALLE Dokumente vollständig durch
2. Identifiziere alle relevanten Informationen zur Frage
3. Verknüpfe Informationen aus verschiedenen Dokumenten
4. Erstelle eine umfassende, vollständige Antwort

ANTWORTSTRUKTUR:
1. Vollständige Beantwortung der Frage
2. Praktische Schritte mit KOMPLETTEN Navigationspfaden
3. Alle relevanten Details aus ALLEN Dokumenten
4. Übersichtliche Struktur mit Überschriften

QUELLENANGABE:
Füge am Ende hinzu:
"## Quellen
Die Informationen stammen aus folgenden Dokumenten:
[HIER LISTE DER VERWENDETEN QUELLEN]"

KRITISCH: Denke dir NIE etwas aus! Nutze NUR das Wissen aus den bereitgestellten Dokumenten!"""
    
    try:
        message = client.messages.create(
            max_tokens=4096,
            model="claude-sonnet-4-20250514",
            system=system_prompt,
            messages=[
                {"role": "user", "content": f"Kontext aus mehreren Dokumenten:\n\n{context}\n\n---\n\nFrage: {question}\n\nVerfügbare Quellen:\n{source_list}"}
            ]
        )
        return message.content[0].text
    except Exception as e:
        return f"Fehler bei der Antwortgenerierung: {e}"

# Streamlit UI
st.set_page_config(page_title="RAG Chat Assistant", page_icon="🔐", layout="wide")

# Passwort-Schutz prüfen
check_password()

# Logout-Button in der Sidebar
with st.sidebar:
    if st.button("🚪 Abmelden"):
        st.session_state.authenticated = False
        st.rerun()

st.title("🤖 RAG Chat Assistant")
st.markdown("Ein intelligenter Chat-Assistent mit Dokumenten-basierter Suche")

# Sidebar
with st.sidebar:
    st.header("⚙️ Konfiguration")
    
    # Automatisches Laden der Academy-Dokumente
    local_docs_path = "./documents"
    if os.path.exists(local_docs_path):
        default_docs_dir = os.environ.get("DOCS_DIR", local_docs_path)
    else:
        default_docs_dir = os.environ.get("DOCS_DIR", "./documents")
    
    docs_dir = st.text_input("Dokumente-Verzeichnis:", value=default_docs_dir)
    
    if st.button("📚 Dokumente laden und indexieren"):
        if docs_dir and os.path.exists(docs_dir):
            with st.spinner("Lade und verarbeite Dokumente..."):
                try:
                    documents = load_markdown_files(docs_dir)
                    if documents:
                        st.write(f"Gefunden: {len(documents)} Dokument-Chunks")
                        
                        with st.spinner("Erstelle Embeddings..."):
                            embeddings = create_embeddings(documents)
                        
                        st.session_state.document_store = documents
                        st.session_state.embeddings = embeddings
                        st.success(f"✅ Index erstellt aus {len(documents)} Text-Stücken!")
                    else:
                        st.error("Keine Markdown-Dateien gefunden!")
                except Exception as e:
                    st.error(f"Fehler beim Indexieren: {e}")
        else:
            st.error("Verzeichnis existiert nicht!")
    
    if st.session_state.document_store:
        st.success(f"✅ {len(st.session_state.document_store)} Chunks bereit!")
    else:
        st.warning("⚠️ Index noch nicht geladen")
    
    if st.button("🗑️ Chat-Verlauf löschen"):
        st.session_state.messages = []
        st.rerun()

# Chat Interface
st.header("💬 Chat")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Stelle deine Frage..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    if st.session_state.document_store:
        with st.chat_message("assistant"):
            with st.spinner("Suche nach relevanten Informationen..."):
                response = answer_question(prompt)
            st.markdown(response)
        
        st.session_state.messages.append({"role": "assistant", "content": response})
    else:
        with st.chat_message("assistant"):
            error_msg = "⚠️ Bitte lade zuerst Dokumente über die Sidebar."
            st.error(error_msg)
        st.session_state.messages.append({"role": "assistant", "content": error_msg})

# Debug-Bereich
with st.expander("🔍 Debug-Informationen"):
    if st.session_state.document_store and st.text_input("Debug-Suche:", key="debug_query"):
        debug_query = st.session_state.debug_query
        if st.button("🔍 Debug-Suche ausführen"):
            with st.spinner("Führe Debug-Suche durch..."):
                results = search_documents(debug_query, top_k=10)
                st.write(f"**Gefunden: {len(results)} relevante Chunks**")
                
                for i, result in enumerate(results, 1):
                    st.write(f"**{i}. Score: {result['score']:.4f}**")
                    st.write(f"Quelle: {result['title']}")
                    st.write(f"Inhalt: {result['content'][:500]}...")
                    st.write("---")