"""
Frontend Streamlit per Assistente AI
=====================================
Interfaccia chat che permette agli utenti di interagire con l'assistente AI.
Il frontend è completamente separato dalla logica AI, comunicando con il backend via API REST.
"""

import streamlit as st
import requests
from typing import Optional
import time

# ============================================================================
# Configurazione Streamlit
# ============================================================================

st.set_page_config(
    page_title="AI Assistant",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="expanded"
)

# URL del backend API
BACKEND_URL = "http://localhost:8000"

# ============================================================================
# Funzioni Helper
# ============================================================================

def check_backend_health() -> bool:
    """
    Verifica se il backend è raggiungibile e funzionante
    
    Returns:
        bool: True se il backend è online, False altrimenti
    """
    try:
        response = requests.get(f"{BACKEND_URL}/health", timeout=3)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False

def send_query_to_backend(query: str) -> Optional[dict]:
    """
    Invia una query al backend e ritorna la risposta
    
    Args:
        query: La domanda dell'utente
    
    Returns:
        dict: Risposta dal backend o None in caso di errore
    """
    try:
        response = requests.post(
            f"{BACKEND_URL}/chat",
            json={"query": query},
            timeout=30  # Timeout più lungo per permettere processing
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Errore dal backend: {response.status_code}")
            return None
    
    except requests.exceptions.Timeout:
        st.error("⏱️ Timeout: il backend sta impiegando troppo tempo a rispondere")
        return None
    
    except requests.exceptions.ConnectionError:
        st.error("🔌 Impossibile connettersi al backend. Assicurati che sia in esecuzione.")
        return None
    
    except Exception as e:
        st.error(f"❌ Errore imprevisto: {str(e)}")
        return None

def format_response(response_data: dict) -> str:
    """
    Formatta la risposta del backend per la visualizzazione
    
    Args:
        response_data: Dati della risposta dal backend
    
    Returns:
        str: Risposta formattata in Markdown
    """
    result = response_data.get("result", "Nessuna risposta disponibile")
    decision = response_data.get("decision", "unknown")
    sources = response_data.get("sources", [])
    
    # Emoji per il tipo di ricerca
    emoji = "📚" if decision == "rag_search" else "🌐"
    tool_name = "RAG (Documenti Interni)" if decision == "rag_search" else "Web Search"
    
    # Costruisci la risposta formattata
    formatted = f"{result}\n\n"
    formatted += f"---\n"
    formatted += f"{emoji} **Metodo usato:** {tool_name}\n"
    
    if sources:
        formatted += f"\n**📎 Fonti:**\n"
        for i, source in enumerate(sources, 1):
            formatted += f"{i}. `{source}`\n"
    
    return formatted

# ============================================================================
# Inizializzazione Session State
# ============================================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "backend_online" not in st.session_state:
    st.session_state.backend_online = check_backend_health()

# ============================================================================
# Sidebar
# ============================================================================

with st.sidebar:
    st.title("🤖 AI Assistant")
    st.markdown("---")
    
    # Status del backend
    st.subheader("📡 Stato Backend")
    if st.button("🔄 Verifica Connessione"):
        st.session_state.backend_online = check_backend_health()
    
    if st.session_state.backend_online:
        st.success("✅ Backend Online")
    else:
        st.error("❌ Backend Offline")
        st.info("Avvia il backend con:\n```bash\ncd backend\npython main.py\n```")
    
    st.markdown("---")
    
    # Informazioni
    st.subheader("ℹ️ Come Funziona")
    st.markdown("""
    Questo assistente AI decide automaticamente come rispondere:
    
    - **📚 RAG (Documenti)**: Per domande su Python, FastAPI, LangGraph, ChromaDB, ML
    
    - **🌐 Web Search**: Per notizie attuali e informazioni generali
    
    L'agente usa LangGraph per orchestrare la decisione!
    """)
    
    st.markdown("---")
    
    # Pulsante per pulire la chat
    if st.button("🗑️ Pulisci Chat"):
        st.session_state.messages = []
        st.rerun()
    
    st.markdown("---")
    
    # Esempi di domande
    st.subheader("💡 Esempi di Domande")
    
    example_rag = st.button("📚 Cos'è LangGraph?")
    example_web = st.button("🌐 Ultime notizie sull'AI")
    example_fastapi = st.button("⚡ Come funziona FastAPI?")

# ============================================================================
# Main Chat Interface
# ============================================================================

st.title("💬 Chat con l'Assistente AI")

st.markdown("""
Fai una domanda all'assistente! L'agente deciderà automaticamente se cercare nei documenti 
interni (RAG) o sul web.
""")

# Avviso se backend offline
if not st.session_state.backend_online:
    st.warning("⚠️ Il backend non è raggiungibile. Verifica che sia in esecuzione.")

# Mostra la cronologia della chat
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle example buttons
query_to_send = None

if example_rag:
    query_to_send = "Cos'è LangGraph?"
elif example_web:
    query_to_send = "Quali sono le ultime notizie sull'intelligenza artificiale?"
elif example_fastapi:
    query_to_send = "Come funziona FastAPI?"

# Input dell'utente
if prompt := st.chat_input("Scrivi la tua domanda qui..."):
    query_to_send = prompt

# Process query
if query_to_send:
    # Aggiungi messaggio utente alla chat
    st.session_state.messages.append({
        "role": "user",
        "content": query_to_send
    })
    
    # Mostra messaggio utente
    with st.chat_message("user"):
        st.markdown(query_to_send)
    
    # Mostra risposta assistente con spinner
    with st.chat_message("assistant"):
        with st.spinner("🤔 Sto pensando..."):
            # Invia query al backend
            response_data = send_query_to_backend(query_to_send)
            
            if response_data:
                # Formatta e mostra la risposta
                formatted_response = format_response(response_data)
                st.markdown(formatted_response)
                
                # Aggiungi alla cronologia
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": formatted_response
                })
            else:
                error_msg = "❌ Impossibile ottenere una risposta. Riprova."
                st.error(error_msg)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg
                })

# ============================================================================
# Footer
# ============================================================================

st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray; font-size: 12px;'>
    🚀 Powered by FastAPI + LangGraph + Streamlit
    </div>
    """,
    unsafe_allow_html=True
)
