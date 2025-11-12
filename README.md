# 🤖 AI Assistant - R&D Project

Un assistente AI intelligente che decide automaticamente se rispondere usando **RAG (Retrieval Augmented Generation)** da documenti interni o effettuare una **ricerca web**, orchestrato da **LangGraph**.

## 📋 Indice

- [Architettura](#-architettura)
- [Tecnologie](#-tecnologie)
- [Prerequisiti](#-prerequisiti)
- [Installazione](#-installazione)
- [Configurazione](#-configurazione)
- [Avvio del Progetto](#-avvio-del-progetto)
- [Utilizzo](#-utilizzo)
- [Struttura del Progetto](#-struttura-del-progetto)
- [API Documentation](#-api-documentation)
- [Troubleshooting](#-troubleshooting)

---

## 🏗️ Architettura

Il progetto è diviso in **due componenti principali** completamente separati:

```
┌─────────────────────────────────────────────────────────────┐
│                         FRONTEND                             │
│                     (Streamlit UI)                           │
│  - Interfaccia chat                                          │
│  - Gestione conversazione                                    │
│  - Nessuna logica AI                                         │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       │ HTTP POST /chat
                       │
┌──────────────────────▼──────────────────────────────────────┐
│                         BACKEND                              │
│                     (FastAPI + LangGraph)                    │
│                                                              │
│  ┌────────────────────────────────────────────────────────┐ │
│  │              LangGraph Agent (rag_logic.py)            │ │
│  │                                                          │ │
│  │  ┌──────────┐                                           │ │
│  │  │  Router  │  ──► Decide quale tool usare             │ │
│  │  │   Node   │      (RAG vs Web Search)                 │ │
│  │  └────┬─────┘                                           │ │
│  │       │                                                  │ │
│  │   ┌───▼────────┐         ┌──────────────┐              │ │
│  │   │ RAG Search │         │ Web Search   │              │ │
│  │   │    Node    │         │     Node     │              │ │
│  │   │            │         │              │              │ │
│  │   │ ChromaDB + │         │   Tavily     │              │ │
│  │   │ LangChain  │         │   API        │              │ │
│  │   └────────────┘         └──────────────┘              │ │
│  └────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

### 🎯 Funzionamento

1. **Frontend (Streamlit)** - Interfaccia chat utente
   - Raccoglie la domanda dell'utente
   - Invia richiesta HTTP POST al backend
   - Visualizza la risposta in modo user-friendly

2. **Backend (FastAPI)** - API REST
   - Riceve la richiesta tramite endpoint `/chat`
   - Delega il processing all'agente LangGraph
   - Ritorna la risposta formattata

3. **Agente LangGraph** - Orchestratore intelligente
   - **Router Node**: Analizza la query e decide il tool appropriato
   - **RAG Search Node**: Cerca nei documenti locali usando ChromaDB
   - **Web Search Node**: Effettua ricerche web tramite Tavily

---

## 🛠️ Tecnologie

### Backend
- **FastAPI** - Framework web ad alte prestazioni
- **LangGraph** - Orchestrazione agenti multi-step
- **LangChain** - Framework per applicazioni LLM
- **ChromaDB** - Vector database per RAG
- **Tavily** - API per ricerca web
- **OpenAI GPT** - Modello linguistico

### Frontend
- **Streamlit** - Framework per UI interattive
- **Requests** - Client HTTP per API calls

---

## ✅ Prerequisiti

- **Python 3.11+**
- **OpenAI API Key** ([ottienila qui](https://platform.openai.com/api-keys))
- **Tavily API Key** ([ottienila qui](https://tavily.com))
- **pip** (package manager Python)

---

## 📦 Installazione

### 1. Clona il repository

```bash
git clone https://github.com/tuousername/R_and_D-AI-Assistant.git
cd R_and_D-AI-Assistant
```

### 2. Installa dipendenze Backend

```bash
cd backend
pip install -r requirements.txt
```

### 3. Installa dipendenze Frontend

```bash
cd ../frontend
pip install -r requirements.txt
```

---

## 🔑 Configurazione

### Variabili d'Ambiente

Crea un file `.env` nella cartella `backend`:

```bash
# backend/.env
OPENAI_API_KEY=sk-your-openai-api-key-here
TAVILY_API_KEY=tvly-your-tavily-api-key-here
```

**Importante:** Non committare mai le API keys nel repository!

### Aggiungi `.env` al `.gitignore`

```bash
echo "backend/.env" >> .gitignore
```

---

## 🚀 Avvio del Progetto

Il progetto richiede **due terminali separati**: uno per il backend, uno per il frontend.

### Terminal 1 - Avvia il Backend

```bash
cd backend
python main.py
```

Il backend sarà disponibile su:
- API: http://localhost:8000
- Docs interattive: http://localhost:8000/docs

Dovresti vedere:

```
INFO:     AI Assistant API Starting...
INFO:     Initializing LangGraph agent...
INFO:     Vector store ready with sample documents
INFO:     Uvicorn running on http://0.0.0.0:8000
```

### Terminal 2 - Avvia il Frontend

```bash
cd frontend
streamlit run app.py
```

Il frontend si aprirà automaticamente nel browser su http://localhost:8501

---

## 💬 Utilizzo

### Esempi di Domande

#### Domande per RAG (Documenti Interni) 📚

Queste domande attiveranno la ricerca nei documenti locali:

```
- Cos'è FastAPI?
- Come funziona LangGraph?
- Spiegami ChromaDB
- Che cos'è il machine learning?
```

#### Domande per Web Search 🌐

Queste domande attiveranno la ricerca web:

```
- Quali sono le ultime notizie sull'intelligenza artificiale?
- Cosa sta succedendo nel mondo oggi?
- Chi ha vinto l'ultimo campionato di calcio?
```

### Interfaccia Streamlit

L'interfaccia mostra:
- 💬 Chat history completa
- 📚/🌐 Indicazione del metodo usato (RAG o Web)
- 📎 Fonti utilizzate per la risposta
- 📡 Stato connessione backend
- 💡 Esempi di domande preimpostate

---

## 📂 Struttura del Progetto

```
R_and_D-AI-Assistant/
│
├── backend/
│   ├── main.py              # FastAPI app con endpoint /chat
│   ├── rag_logic.py         # Logica LangGraph (Router, RAG, Web Search)
│   ├── requirements.txt     # Dipendenze backend
│   ├── Dockerfile           # Container Docker per backend
│   └── .env                 # API keys (da creare)
│
├── frontend/
│   ├── app.py               # Streamlit UI
│   └── requirements.txt     # Dipendenze frontend
│
└── README.md                # Questo file
```

---

## 📖 API Documentation

### `POST /chat`

**Endpoint principale** per inviare query all'assistente.

#### Request

```json
{
  "query": "Cos'è FastAPI?"
}
```

#### Response

```json
{
  "query": "Cos'è FastAPI?",
  "decision": "rag_search",
  "result": "FastAPI è un framework web moderno e veloce...",
  "sources": ["fastapi_docs.txt"],
  "status": "success"
}
```

#### Campi

- **query**: La domanda originale
- **decision**: `rag_search` o `web_search`
- **result**: La risposta generata
- **sources**: Array di fonti utilizzate
- **status**: Stato della richiesta

### Altri Endpoints

- `GET /` - Root endpoint
- `GET /health` - Health check
- `GET /docs` - Documentazione interattiva Swagger

---

## 🐳 Docker (Opzionale)

### Build dell'immagine Docker

```bash
cd backend
docker build -t ai-assistant-backend .
```

### Esegui il container

```bash
docker run -p 8000:8000 \
  -e OPENAI_API_KEY=your-key \
  -e TAVILY_API_KEY=your-key \
  ai-assistant-backend
```

---

## 🔧 Troubleshooting

### Backend non si avvia

**Problema:** Import errors o dipendenze mancanti

```bash
cd backend
pip install -r requirements.txt --upgrade
```

### Frontend non si connette al backend

**Problema:** `Connection refused` o timeout

**Soluzione:**
1. Verifica che il backend sia in esecuzione su porta 8000
2. Clicca "Verifica Connessione" nella sidebar
3. Controlla i log del backend per errori

### Errori API Keys

**Problema:** `401 Unauthorized` o errori autenticazione

**Soluzione:**
1. Verifica che il file `.env` esista in `backend/`
2. Controlla che le API keys siano valide
3. Riavvia il backend dopo aver modificato `.env`

### ChromaDB errors

**Problema:** Errori di inizializzazione vector store

```bash
pip install chromadb --upgrade
```

### Rate limit errors

**Problema:** Troppi request alle API

**Soluzione:**
- Attendi qualche secondo tra le richieste
- Considera l'upgrade del piano API se necessario

---

## 🎓 Concetti Chiave

### LangGraph
Framework per costruire agenti LLM con **grafi stateful**. Permette di:
- Definire nodi (funzioni) e edge (transizioni)
- Creare routing condizionale
- Mantenere stato condiviso

### RAG (Retrieval Augmented Generation)
Tecnica che combina:
1. **Retrieval**: Ricerca documenti rilevanti in un vector database
2. **Augmentation**: Aggiunta del contesto al prompt
3. **Generation**: LLM genera risposta basata sui documenti

### Conditional Routing
L'agente usa un **router intelligente** che:
- Analizza la query con un LLM
- Decide dinamicamente quale tool chiamare
- Usa `Command.goto()` per navigare nel grafo

---

## 📝 Note per lo Sviluppo

### Aggiungere Nuovi Documenti

Modifica la funzione `initialize_vector_store()` in `backend/rag_logic.py`:

```python
sample_documents = [
    Document(
        page_content="Il tuo nuovo contenuto...",
        metadata={"source": "nuovo_doc.txt"}
    ),
    # ... altri documenti
]
```

### Cambiare il Modello LLM

In `backend/rag_logic.py`, modifica:

```python
llm = ChatOpenAI(
    model="gpt-4",  # o "gpt-3.5-turbo", etc.
    temperature=0,
)
```

### Personalizzare il Routing

Modifica il prompt in `router_node()` per cambiare la logica di decisione.

---

## 🤝 Contribuire

Questo è un progetto R&D educativo. Sentiti libero di:
- Fare fork e sperimentare
- Aggiungere nuovi nodi al grafo
- Integrare altri vector databases
- Migliorare l'UI Streamlit

---

## 📄 Licenza

Questo progetto è fornito "as-is" per scopi educativi e di ricerca.

---

## 👤 Autore

Progetto R&D AI Assistant

---

## 🙏 Ringraziamenti

- **LangChain & LangGraph** - Framework eccezionale per agenti AI
- **FastAPI** - Best web framework Python
- **Streamlit** - UI rapide e belle
- **Tavily** - Ottima API di ricerca web

---

**Happy Coding! 🚀**