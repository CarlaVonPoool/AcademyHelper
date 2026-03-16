import streamlit as st
import os
import spacy
from typing import List
from dotenv import load_dotenv
import anthropic
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma

load_dotenv()

# Globale Variablen
if 'vectordb' not in st.session_state:
    st.session_state.vectordb = None
if 'nlp' not in st.session_state:
    st.session_state.nlp = None
if 'messages' not in st.session_state:
    st.session_state.messages = []

def initialize_spacy():
    """Initialisiert das deutsche spaCy-Modell für Lemmatization."""
    try:
        st.session_state.nlp = spacy.load("de_core_news_sm")
    except OSError:
        st.error("Deutsches spaCy-Modell nicht gefunden. Installiere es mit: python -m spacy download de_core_news_sm")
        st.stop()

def lemmatize_text(text: str) -> str:
    """Lemmatisiert deutschen Text unter Beibehaltung der Lesbarkeit."""
    if st.session_state.nlp is None:
        initialize_spacy()
    
    doc = st.session_state.nlp(text)
    lemmatized = " ".join([token.lemma_ for token in doc])
    return lemmatized

def extract_keywords(text: str) -> List[str]:
    """Extrahiert wichtige Keywords aus dem Text (Nomen, Verben)."""
    if st.session_state.nlp is None:
        initialize_spacy()
    
    doc = st.session_state.nlp(text)
    keywords = []
    for token in doc:
        if token.pos_ in ['NOUN', 'PROPN', 'VERB'] and not token.is_stop:
            keywords.append(token.lemma_.lower())
    return list(set(keywords))

def preprocess_documents(documents):
    """Vorverarbeitet Dokumente mit Lemmatization."""
    if st.session_state.nlp is None:
        initialize_spacy()
    
    processed_docs = []
    for doc in documents:
        original_content = doc.page_content
        lemmatized_content = lemmatize_text(original_content)
        
        doc.page_content = f"{original_content}\n\n[Lemmatisiert]: {lemmatized_content}"
        processed_docs.append(doc)
    
    return processed_docs

@st.cache_data
def load_and_index(docs_dir: str, persist_dir: str = "chroma_db"):
    """Lädt Markdown-Dateien und erstellt einen Vektorindex mit Lemmatization."""
    
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
    
    return vectordb, f"Index erstellt aus {len(processed_chunks)} lemmatisierten Text-Stücken."

def expand_query(question: str) -> List[str]:
    """Erstelle mehrere Suchvarianten mit Kontext-Spezifizierung"""
    if st.session_state.nlp is None:
        initialize_spacy()
    
    queries = [question]
    
    # Lemmatisierte Version
    lemmatized = lemmatize_text(question)
    queries.append(lemmatized)
    
    # Nur Keywords (für fokussierte Suche)
    keywords = extract_keywords(question)
    if keywords:
        queries.append(" ".join(keywords[:5]))
    
    return queries

def answer_question(question: str) -> str:
    """Beantwortet eine Frage mit verbesserter Suche."""
    if st.session_state.vectordb is None:
        return "Index wurde noch nicht initialisiert. Bitte lade zuerst Dokumente."
    
    if st.session_state.nlp is None:
        initialize_spacy()
    
    # Multi-Query Retrieval mit Query Expansion
    queries = expand_query(question)
    
    all_results = []
    seen_docs = set()
    
    # Sammle Ergebnisse von allen Query-Varianten
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
    
    # Kontext mit Quellen und besserer Strukturierung
    context_parts = []
    sources = set()
    source_to_chunks = {}
    
    for doc in results:
        source = doc.metadata.get('source', 'Unknown')
        sources.add(source)
        
        title = source.split('/')[-1].replace('.md', '') if source != 'Unknown' else 'Unknown'
        content = doc.page_content.split('\n\n[Lemmatisiert]:')[0]
        
        if title not in source_to_chunks:
            source_to_chunks[title] = []
        source_to_chunks[title].append(content)
    
    # Erstelle strukturierten Kontext gruppiert nach Dokumenten
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

# Sidebar für Konfiguration
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
    # Benutzer-Nachricht hinzufügen
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Assistent-Antwort generieren
    if st.session_state.vectordb:
        with st.chat_message("assistant"):
            with st.spinner("Suche nach relevanten Informationen..."):
                response = answer_question(prompt)
            st.markdown(response)
        
        # Antwort speichern
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
                    content = doc.page_content.split('\n\n[Lemmatisiert]:')[0][:500]
                    st.write(f"Inhalt: {content}...")
                    st.write("---")