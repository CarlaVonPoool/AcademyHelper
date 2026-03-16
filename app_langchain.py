import streamlit as st
import os
from typing import List
from dotenv import load_dotenv
import anthropic
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import re

load_dotenv()

# Globale Variablen
if 'vectordb' not in st.session_state:
    st.session_state.vectordb = None
if 'messages' not in st.session_state:
    st.session_state.messages = []

def simple_preprocessing(text: str) -> str:
    """Einfache Text-Vorverarbeitung ohne spaCy."""
    # Lowercase
    text = text.lower()
    
    # Entfernen von Sonderzeichen, aber Umlaute beibehalten
    text = re.sub(r'[^\w\säöüß]', ' ', text)
    
    # Mehrfache Leerzeichen entfernen
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()

def extract_simple_keywords(text: str) -> List[str]:
    """Einfache Keyword-Extraktion basierend auf Wortlänge."""
    words = simple_preprocessing(text).split()
    
    # Deutsche Stopwörter (vereinfachte Liste)
    stopwords = {
        'der', 'die', 'und', 'in', 'den', 'von', 'zu', 'das', 'mit', 'sich', 'des', 'auf', 'für',
        'ist', 'im', 'dem', 'nicht', 'ein', 'eine', 'als', 'auch', 'es', 'an', 'werden', 'aus',
        'er', 'hat', 'dass', 'sie', 'nach', 'wird', 'bei', 'einer', 'um', 'am', 'sind', 'noch',
        'wie', 'einem', 'über', 'einen', 'so', 'zum', 'war', 'haben', 'nur', 'oder', 'aber',
        'vor', 'zur', 'bis', 'mehr', 'durch', 'man', 'sein', 'wurde', 'sei', 'in', 'wenn'
    }
    
    # Keywords: Wörter länger als 3 Zeichen und nicht in Stopwörtern
    keywords = []
    for word in words:
        if len(word) > 3 and word not in stopwords:
            keywords.append(word)
    
    return list(set(keywords))[:10]  # Top 10 unique keywords

def preprocess_documents(documents):
    """Vorverarbeitet Dokumente mit einfacher Textverarbeitung."""
    processed_docs = []
    for doc in documents:
        original_content = doc.page_content
        processed_content = simple_preprocessing(original_content)
        
        doc.page_content = f"{original_content}\n\n[Verarbeitet]: {processed_content}"
        processed_docs.append(doc)
    
    return processed_docs

@st.cache_data
def load_and_index(docs_dir: str, persist_dir: str = "chroma_db"):
    """Lädt Markdown-Dateien und erstellt einen Vektorindex."""
    
    loader = DirectoryLoader(docs_dir, glob="**/*.md")
    documents = loader.load()
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=400,
        separators=["\n## ", "\n### ", "\n\n", "\n", " ", ""]
    )
    chunks = splitter.split_documents(documents)
    
    if not chunks:
        return None, f"Keine Dokumente in '{docs_dir}' gefunden."
    
    processed_chunks = preprocess_documents(chunks)
    embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectordb = Chroma.from_documents(processed_chunks, embedding=embedder, persist_directory=persist_dir)
    
    return vectordb, f"Index erstellt aus {len(processed_chunks)} Text-Stücken."

def expand_query(question: str) -> List[str]:
    """Erstelle mehrere Suchvarianten."""
    queries = [question]
    
    # Vorverarbeitete Version
    processed = simple_preprocessing(question)
    queries.append(processed)
    
    # Nur Keywords
    keywords = extract_simple_keywords(question)
    if keywords:
        queries.append(" ".join(keywords[:5]))
    
    return queries

def answer_question(question: str) -> str:
    """Beantwortet eine Frage mit verbesserter Suche."""
    if st.session_state.vectordb is None:
        return "Index wurde noch nicht initialisiert. Bitte lade zuerst Dokumente."
    
    # Multi-Query Retrieval
    queries = expand_query(question)
    
    all_results = []
    seen_docs = set()
    
    for query in queries:
        results = st.session_state.vectordb.similarity_search_with_score(query, k=25)
        for doc, score in results:
            doc_hash = hash(doc.page_content[:100])
            if doc_hash not in seen_docs:
                all_results.append((doc, score))
                seen_docs.add(doc_hash)
    
    # Sortiere nach Score und nimm die besten 40
    all_results.sort(key=lambda x: x[1])
    results = [doc for doc, _ in all_results[:40]]
    
    # Kontext erstellen
    context_parts = []
    sources = set()
    source_to_chunks = {}
    
    for doc in results:
        source = doc.metadata.get('source', 'Unknown')
        sources.add(source)
        
        title = source.split('/')[-1].replace('.md', '') if source != 'Unknown' else 'Unknown'
        content = doc.page_content.split('\n\n[Verarbeitet]:')[0]
        
        if title not in source_to_chunks:
            source_to_chunks[title] = []
        source_to_chunks[title].append(content)
    
    for title, chunks in source_to_chunks.items():
        combined_content = "\n\n".join(chunks)
        context_parts.append(f"=== Dokument: {title} ===\n{combined_content}")
    
    context = "\n\n".join(context_parts)
    source_list = "\n".join([f"- {s.split('/')[-1].replace('.md', '')}" for s in sources if s != 'Unknown'])
    
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return "Fehler: ANTHROPIC_API_KEY Umgebungsvariable ist nicht gesetzt."
    
    client = anthropic.Anthropic(api_key=api_key)
    
    system_prompt = """Du bist ein Experten-Assistent. Analysiere ALLE bereitgestellten Dokumente gründlich.

ANALYSEPROZESS:
1. Lies ALLE Dokumente vollständig durch
2. Identifiziere alle relevanten Informationen zur Frage
3. Verknüpfe Informationen aus verschiedenen Dokumenten
4. Prüfe auf Widersprüche oder ergänzende Details
5. Erstelle eine umfassende, vollständige Antwort

ANTWORTSTRUKTUR:
1. Vollständige Beantwortung der Frage
2. Praktische Schritte mit KOMPLETTEN Navigationspfaden
3. Alle relevanten Details aus ALLEN Dokumenten
4. Übersichtliche Struktur mit Überschriften
5. Verbindungen zwischen verschiedenen Informationen

WICHTIG - Vollständige Anleitungen:
- "Rechnung" ≠ "Eingangsrechnung" ≠ "Ausgangsrechnung" ≠ "Wiederkehrende Eingangsrechnung" ≠  "Wiederkehrende Ausgangsrechnung"
- Beginne IMMER bei Schritt 1
- Führe vom Anfang bis zum Ende durch den Prozess
- Erkläre Voraussetzungen zuerst
- Baue logische Schritt-für-Schritt Ketten
- Verbinde zusammengehörige Schritte aus verschiedenen Chunks
- Nutze ALLE relevanten Informationen aus ALLEN passenden Dokumenten
- Fokussiere auf den spezifischen Kontext der Frage

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
st.set_page_config(page_title="RAG Chat Assistant", page_icon="🤖", layout="wide")

st.title("🤖 RAG Chat Assistant")
st.markdown("Ein intelligenter Chat-Assistent mit Dokumenten-basierter Suche")

# Sidebar
with st.sidebar:
    st.header("⚙️ Konfiguration")
    
    default_docs_dir = os.environ.get("DOCS_DIR", "./documents")
    docs_dir = st.text_input(
        "Dokumente-Verzeichnis:", 
        value=default_docs_dir
    )
    
    if st.button("📚 Dokumente laden und indexieren"):
        if docs_dir and os.path.exists(docs_dir):
            with st.spinner("Lade und indexiere Dokumente..."):
                try:
                    vectordb, message = load_and_index(docs_dir)
                    if vectordb:
                        st.session_state.vectordb = vectordb
                        st.success(message)
                    else:
                        st.error(message)
                except Exception as e:
                    st.error(f"Fehler beim Indexieren: {e}")
        else:
            st.error("Verzeichnis existiert nicht!")
    
    if st.session_state.vectordb:
        st.success("✅ Index ist bereit!")
    else:
        st.warning("⚠️ Index noch nicht geladen")
    
    if st.button("🗑️ Chat-Verlauf löschen"):
        st.session_state.messages = []
        st.rerun()

# Chat Interface
st.header("💬 Chat")

# Chat-Verlauf anzeigen
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat-Eingabe
if prompt := st.chat_input("Stelle deine Frage..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    if st.session_state.vectordb:
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
    if st.session_state.vectordb and st.text_input("Debug-Suche:", key="debug_query"):
        debug_query = st.session_state.debug_query
        if st.button("🔍 Debug-Suche ausführen"):
            with st.spinner("Führe Debug-Suche durch..."):
                queries = expand_query(debug_query)
                st.write("**Query-Varianten:**", queries)
                
                results = st.session_state.vectordb.similarity_search_with_score(debug_query, k=10)
                for i, (doc, score) in enumerate(results, 1):
                    st.write(f"**{i}. Score: {score:.4f}**")
                    st.write(f"Quelle: {doc.metadata.get('source', 'Unknown')}")
                    content = doc.page_content.split('\n\n[Verarbeitet]:')[0][:500]
                    st.write(f"Inhalt: {content}...")
                    st.write("---")