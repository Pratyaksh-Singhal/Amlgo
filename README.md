# Document RAG Chatbot

## Project Overview

This project implements a Retrieval-Augmented Generation (RAG) chatbot capable of answering questions from a provided document. The chatbot retrieves relevant document chunks using semantic search and generates grounded responses using a Large Language Model (LLM).

The application supports:

* Document ingestion and preprocessing
* Sentence-aware chunking
* Embedding generation
* Vector database storage using ChromaDB
* Semantic retrieval
* LLM-based answer generation
* Real-time streaming responses in Streamlit
* Source chunk display for transparency

---

## Project Architecture

```text
User Query
    │
    ▼
Retriever (ChromaDB)
    │
    ▼
Relevant Chunks
    │
    ▼
Prompt Builder
    │
    ▼
Groq LLM
    │
    ▼
Generated Answer
    │
    ▼
Streamlit UI
```

### Workflow

1. Load and clean PDF document.
2. Split document into semantic chunks.
3. Generate embeddings using BAAI/bge-base-en-v1.5.
4. Store embeddings in ChromaDB.
5. Retrieve top relevant chunks for a user query.
6. Inject retrieved chunks into the prompt.
7. Generate grounded response using Groq LLM.
8. Stream answer token-by-token in Streamlit.
9. Display retrieved source chunks.

---

## Folder Structure

```text
project/
│
├── data/
│   └── AI Training Document.pdf
│
├── chunks/
│   └── chunks.json
│
├── vectordb/
│   └── chroma database files
│
├── notebooks/
│   └── try.ipynb
│
├── src/
│   ├── ingestion.py
│   ├── retriever.py
│   ├── generator.py
│   └── pipeline.py
│
├── app.py
├── requirements.txt
├── README.md
└── report.pdf
```

---

## Embedding Model Choice

### Model

BAAI/bge-base-en-v1.5

### Why?

* Strong semantic retrieval performance
* Lightweight and efficient
* Well suited for RAG applications
* Open-source and easy to deploy

---

## Vector Database Choice

### ChromaDB

Reasons:

* Easy local deployment
* Persistent storage support
* Native LangChain integration
* Fast semantic similarity search

---

## LLM Choice

### Qwen/Qwen3-32B (Groq)

Reasons:

* Fast inference through Groq
* Strong instruction-following capabilities
* Suitable for document question answering

---

## Preprocessing and Embedding Generation

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 2: Configure Environment Variables

Create `.env`

```env
GROQ_API_KEY= <YOUR_API_KEY>
PDF_PATH = <Data file path>
VECTORDB_PATH = <vectordb path>
CHUNKS_PATH = <chunks path>
EMBED_MODEL = BAAI/bge-base-en-v1.5
CHROMA_DIR  = vectordb
GROQ_MODEL = qwen/qwen3-32b
COLLECTION_NAME = rag_docs
```

### Step 3: Build Vector Database

```bash
python src/ingestion.py
```

This performs:
* Document loading
* Cleaning
* Chunking
* Embedding generation
* ChromaDB creation

---

## Running the Chatbot

```bash
streamlit run app.py
```
Features:
* Chat interface
* Streaming responses
* Source chunk visualization
* Clear chat button
* Model information sidebar
* Indexed chunk count display

---

## Sample Queries

1. What happens when a buyer purchases an item on eBay?

2. Can eBay record phone conversations?

3. What is the eBay Money Back Guarantee?

4. How are international purchases handled?

5. What happens if a seller violates eBay policies?

---
## Demo

### Demo Video

Add your recording link here:

```text
https://drive.google.com/file/d/1hu_K3qwh0Q7pkL4GDZyvaKpsQCs3zCcy/view?usp=sharing
```

### GitHub Repository

```text
https://github.com/Pratyaksh-Singhal/Amlgo
```
