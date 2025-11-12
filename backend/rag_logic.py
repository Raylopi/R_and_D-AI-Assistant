"""
LangGraph Agent Logic for RAG vs Web Search Routing
====================================================
Questo modulo implementa un agente intelligente che decide automaticamente
se rispondere a una query usando RAG (Retrieval Augmented Generation) da documenti
o effettuare una ricerca web.
"""

import os
from typing import TypedDict, Literal, Annotated
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from tavily import TavilyClient
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command


# ============================================================================
# Definizione dello State del Grafo
# ============================================================================

class AgentState(TypedDict):
    """Stato condiviso tra i nodi del grafo LangGraph"""
    query: str
    decision: Literal["rag_search", "web_search", ""]
    result: str
    documents: list[str]


# ============================================================================
# Configurazione e Inizializzazione
# ============================================================================

# Inizializza il modello LLM
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    api_key=os.getenv("OPENAI_API_KEY", "your-openai-api-key")
)

# Inizializza Tavily per la ricerca web
tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY", "your-tavily-api-key"))

# Inizializza ChromaDB in-memory con documenti fittizi
def initialize_vector_store():
    """Crea un vector store in-memory con documenti di esempio"""
    
    # Documenti fittizi per il RAG
    sample_documents = [
        Document(
            page_content="Python è un linguaggio di programmazione ad alto livello, interpretato e general-purpose. "
                        "È noto per la sua sintassi chiara e leggibile, che enfatizza la leggibilità del codice.",
            metadata={"source": "python_intro.txt"}
        ),
        Document(
            page_content="FastAPI è un framework web moderno e veloce per la creazione di API con Python 3.7+. "
                        "È basato su standard come OpenAPI e JSON Schema, e offre validazione automatica dei dati.",
            metadata={"source": "fastapi_docs.txt"}
        ),
        Document(
            page_content="LangGraph è un framework per costruire agenti e applicazioni multi-agente con LLM. "
                        "Permette di creare grafi stateful con cicli, condizioni e persistenza dello stato.",
            metadata={"source": "langgraph_guide.txt"}
        ),
        Document(
            page_content="Il machine learning è un campo dell'intelligenza artificiale che si concentra sulla "
                        "costruzione di sistemi che apprendono dai dati e migliorano le loro prestazioni nel tempo.",
            metadata={"source": "ml_basics.txt"}
        ),
        Document(
            page_content="ChromaDB è un database vettoriale open-source progettato per applicazioni AI. "
                        "Supporta embeddings e ricerca per similarità, ideale per sistemi RAG.",
            metadata={"source": "chroma_overview.txt"}
        )
    ]
    
    # Split dei documenti (opzionale per documenti così piccoli)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    splits = text_splitter.split_documents(sample_documents)
    
    # Crea vector store
    embeddings = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY", "your-openai-api-key"))
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        collection_name="rag_documents"
    )
    
    return vectorstore

# Inizializza il vector store globalmente
vectorstore = initialize_vector_store()


# ============================================================================
# Nodi del Grafo LangGraph
# ============================================================================

def router_node(state: AgentState) -> Command[Literal["rag_search", "web_search"]]:
    """
    Nodo Router: Decide quale strumento usare (RAG o Web Search)
    
    Analizza la query dell'utente e determina se la domanda riguarda:
    - Conoscenze nei documenti interni (RAG)
    - Informazioni attuali/esterne (Web Search)
    """
    
    query = state["query"]
    
    # Prompt per il routing
    system_prompt = """Sei un router intelligente. Analizza la domanda dell'utente e decidi quale strumento usare:

- 'rag_search': se la domanda riguarda Python, FastAPI, LangGraph, ChromaDB, machine learning, o concetti di programmazione
- 'web_search': se la domanda riguarda notizie attuali, eventi recenti, informazioni generali non tecniche

Rispondi SOLO con 'rag_search' o 'web_search', senza altre parole."""
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Domanda: {query}")
    ]
    
    response = llm.invoke(messages)
    decision = response.content.strip().lower()
    
    # Validazione della decisione
    if decision not in ["rag_search", "web_search"]:
        decision = "rag_search"  # Default fallback
    
    # Ritorna Command con la decisione di routing
    return Command(
        goto=decision,
        update={"decision": decision}
    )


def rag_search_node(state: AgentState) -> Command[Literal[END]]:
    """
    Nodo RAG: Esegue ricerca nei documenti locali
    
    Usa ChromaDB per trovare documenti rilevanti e genera una risposta
    basata sui documenti recuperati.
    """
    
    query = state["query"]
    
    # Ricerca similarità nel vector store
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    relevant_docs = retriever.invoke(query)
    
    # Estrai il contenuto dei documenti
    documents_content = "\n\n".join([doc.page_content for doc in relevant_docs])
    
    # Genera risposta usando RAG
    rag_prompt = f"""Usa i seguenti documenti per rispondere alla domanda. 
Se la risposta non è nei documenti, dillo chiaramente.

DOCUMENTI:
{documents_content}

DOMANDA: {query}

RISPOSTA:"""
    
    messages = [
        SystemMessage(content="Sei un assistente AI che risponde basandosi solo sui documenti forniti."),
        HumanMessage(content=rag_prompt)
    ]
    
    response = llm.invoke(messages)
    result = response.content
    
    # Salva i documenti recuperati per riferimento
    doc_sources = [doc.metadata.get("source", "unknown") for doc in relevant_docs]
    
    return Command(
        goto=END,
        update={
            "result": result,
            "documents": doc_sources
        }
    )


def web_search_node(state: AgentState) -> Command[Literal[END]]:
    """
    Nodo Web Search: Esegue ricerca web tramite Tavily
    
    Usa l'API Tavily per cercare informazioni aggiornate sul web
    e sintetizza i risultati in una risposta coerente.
    """
    
    query = state["query"]
    
    try:
        # Esegui ricerca web con Tavily
        search_results = tavily_client.search(
            query=query,
            search_depth="basic",
            max_results=3
        )
        
        # Estrai risultati
        results_content = "\n\n".join([
            f"Fonte: {result.get('url', 'N/A')}\n{result.get('content', '')}"
            for result in search_results.get('results', [])
        ])
        
        # Genera risposta basata sui risultati web
        web_prompt = f"""Usa i seguenti risultati di ricerca web per rispondere alla domanda.

RISULTATI WEB:
{results_content}

DOMANDA: {query}

RISPOSTA (sintetizza le informazioni in modo chiaro):"""
        
        messages = [
            SystemMessage(content="Sei un assistente AI che sintetizza informazioni da ricerche web."),
            HumanMessage(content=web_prompt)
        ]
        
        response = llm.invoke(messages)
        result = response.content
        
        # Salva le URL per riferimento
        sources = [result.get('url', 'N/A') for result in search_results.get('results', [])]
        
        return Command(
            goto=END,
            update={
                "result": result,
                "documents": sources
            }
        )
    
    except Exception as e:
        # Fallback in caso di errore
        error_message = f"Errore durante la ricerca web: {str(e)}"
        return Command(
            goto=END,
            update={
                "result": error_message,
                "documents": []
            }
        )


# ============================================================================
# Costruzione del Grafo LangGraph
# ============================================================================

def build_agent_graph() -> StateGraph:
    """Costruisce e compila il grafo dell'agente"""
    
    # Crea il grafo
    graph_builder = StateGraph(AgentState)
    
    # Aggiungi nodi
    graph_builder.add_node("router", router_node)
    graph_builder.add_node("rag_search", rag_search_node)
    graph_builder.add_node("web_search", web_search_node)
    
    # Aggiungi edge di partenza
    graph_builder.add_edge(START, "router")
    
    # Il routing è gestito dal Command nel router_node
    # Non servono conditional_edges perché usiamo Command.goto
    
    # Compila il grafo
    agent_graph = graph_builder.compile()
    
    return agent_graph


# ============================================================================
# Funzione Principale per Eseguire l'Agente
# ============================================================================

def run_agent(query: str) -> dict:
    """
    Esegue l'agente con la query fornita
    
    Args:
        query: La domanda dell'utente
    
    Returns:
        dict con campi:
            - query: la domanda originale
            - decision: quale tool è stato usato
            - result: la risposta generata
            - sources: fonti utilizzate
    """
    
    # Stato iniziale
    initial_state = {
        "query": query,
        "decision": "",
        "result": "",
        "documents": []
    }
    
    # Compila e esegui il grafo
    graph = build_agent_graph()
    final_state = graph.invoke(initial_state)
    
    # Prepara risposta
    response = {
        "query": final_state["query"],
        "decision": final_state["decision"],
        "result": final_state["result"],
        "sources": final_state["documents"]
    }
    
    return response


# ============================================================================
# Test Standalone (opzionale)
# ============================================================================

if __name__ == "__main__":
    # Test del sistema
    test_queries = [
        "Cos'è FastAPI?",
        "Quali sono le ultime notizie sull'intelligenza artificiale?",
        "Come funziona ChromaDB?"
    ]
    
    for query in test_queries:
        print(f"\n{'='*80}")
        print(f"QUERY: {query}")
        print('='*80)
        
        result = run_agent(query)
        
        print(f"\nDecisione: {result['decision']}")
        print(f"\nRisposta:\n{result['result']}")
        print(f"\nFonti: {', '.join(result['sources'])}")
