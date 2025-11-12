"""
FastAPI Backend per Assistente AI
==================================
Questo modulo espone un'API REST che riceve query dall'utente
e le processa tramite l'agente LangGraph definito in rag_logic.py
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
import logging
from rag_logic import run_agent

# ============================================================================
# Configurazione Logging
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# Inizializzazione FastAPI
# ============================================================================

app = FastAPI(
    title="AI Assistant API",
    description="API per un assistente AI che decide automaticamente tra RAG e Web Search",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configurazione CORS per permettere richieste dal frontend Streamlit
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In produzione, specifica l'URL del frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# Modelli Pydantic
# ============================================================================

class ChatRequest(BaseModel):
    """Schema per la richiesta di chat"""
    query: str = Field(
        ...,
        description="La domanda o richiesta dell'utente",
        min_length=1,
        max_length=1000,
        examples=["Cos'è FastAPI?", "Quali sono le ultime notizie sull'AI?"]
    )

class ChatResponse(BaseModel):
    """Schema per la risposta di chat"""
    query: str = Field(description="La domanda originale dell'utente")
    decision: str = Field(description="Il tool scelto dall'agente (rag_search o web_search)")
    result: str = Field(description="La risposta generata dall'agente")
    sources: list[str] = Field(description="Le fonti utilizzate per generare la risposta")
    status: str = Field(default="success", description="Stato della richiesta")

class HealthResponse(BaseModel):
    """Schema per la risposta di health check"""
    status: str = Field(description="Stato del servizio")
    message: str = Field(description="Messaggio descrittivo")

# ============================================================================
# Endpoints
# ============================================================================

@app.get("/", response_model=HealthResponse)
async def root():
    """
    Endpoint root per verificare che l'API sia attiva
    """
    return HealthResponse(
        status="online",
        message="AI Assistant API is running. Visit /docs for API documentation."
    )

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint per monitoraggio
    """
    return HealthResponse(
        status="healthy",
        message="All systems operational"
    )

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Endpoint principale per processare query dell'utente
    
    Riceve una domanda, la processa tramite l'agente LangGraph
    che decide automaticamente se usare RAG o Web Search.
    
    Args:
        request: ChatRequest con la query dell'utente
    
    Returns:
        ChatResponse con la risposta dell'agente e metadati
    
    Raises:
        HTTPException: In caso di errori durante il processing
    """
    
    try:
        logger.info(f"Ricevuta query: {request.query}")
        
        # Esegui l'agente
        agent_response = run_agent(request.query)
        
        logger.info(f"Agente ha scelto: {agent_response['decision']}")
        
        # Prepara la risposta
        response = ChatResponse(
            query=agent_response["query"],
            decision=agent_response["decision"],
            result=agent_response["result"],
            sources=agent_response["sources"],
            status="success"
        )
        
        return response
    
    except Exception as e:
        logger.error(f"Errore durante il processing: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Errore interno del server: {str(e)}"
        )

# ============================================================================
# Startup/Shutdown Events
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """
    Evento eseguito all'avvio dell'applicazione
    """
    logger.info("=" * 80)
    logger.info("AI Assistant API Starting...")
    logger.info("=" * 80)
    logger.info("Initializing LangGraph agent...")
    logger.info("Vector store ready with sample documents")
    logger.info("API documentation available at: http://localhost:8000/docs")
    logger.info("=" * 80)

@app.on_event("shutdown")
async def shutdown_event():
    """
    Evento eseguito allo shutdown dell'applicazione
    """
    logger.info("AI Assistant API Shutting down...")

# ============================================================================
# Main (per esecuzione diretta)
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
