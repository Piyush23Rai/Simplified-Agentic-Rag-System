# Simplified Agentic RAG System — Gemini + LangGraph

## 1. Overview

This project implements a **Simplified Agentic Retrieval-Augmented Generation (RAG) system** using Google Gemini (Vertex AI) and LangGraph.  

**Key Features:**
- Combines a **Knowledge Base (KB)** with a **Large Language Model (LLM)** to generate factual, citation-backed answers.
- Includes a **self-critique loop**:
  - The LLM reviews its own answer.
  - If incomplete, it retrieves additional KB snippets and refines the answer.
- Logs each step using **Python logging** and optionally **MLflow** for experiment tracking.

**Workflow Steps:**
1. **Retrieve KB** → Get top 5 relevant KB snippets from Pinecone.
2. **Generate Answer** → LLM generates initial answer citing KB snippets.
3. **Self-Critique** → LLM checks for completeness.
4. **Refine Answer** → If critique indicates missing info, retrieve 1 extra snippet and regenerate answer.

---

## 2. Problem Statement

> Build a simplified Agentic RAG system using Gemini + LangGraph that:
> - Retrieves up to 5 KB snippets
> - Generates an initial answer
> - Critiques it
> - If necessary, refines the answer by pulling one more snippet
> - Returns a citation-backed response

This ensures answers are **grounded in factual content** and **self-verified**.

**Example Queries:**
- "What are best practices for caching?"
- "How should I set up CI/CD pipelines?"
- "What are performance tuning tips?"
- "How do I version my APIs?"
- "What should I consider for error handling?"

---

## 3. Folder Structure

```
project_root/
│
├─ agentic_rag/
│   ├─ index_kb.py               # Preprocess and index KB JSON into Pinecone
│   ├─ agentic_rag_simplified.py # LangGraph workflow: retrieve → generate → critique → refine
│
├─ data/
│   └─ self_critique_loop_dataset.json  # Knowledge Base JSON
│
├─ run.py                        # Script to run the Agentic RAG workflow
├─ .env                          # Environment variables
├─ requirements.txt              # Python dependencies
└─ README.md
```

---

## 4. Environment Setup

1. **Clone the repository** and navigate to the project root.
2. **Install dependencies**:

```bash
pip install -r requirements.txt
```

3. **Create a `.env` file** in the project root with the following parameters:

```
# Pinecone API key for vector database
PINECONE_API_KEY=<YOUR_PINCEONE_API_KEY>

# Google API key for Vertex AI / Gemini
GOOGLE_API_KEY=<YOUR GOOGLE API KEY>

# Flag for using Vertex AI endpoints (True/False)
GOOGLE_GENAI_USE_VERTEXAI=False

# MLflow tracking server URI 
MLFLOW_TRACKING_URI=<YOUR_URI>
```

4. **Set environment variables**:

On Linux/macOS:
```bash
export $(cat .env | xargs)
```

On Windows (PowerShell):
```powershell
Get-Content .env | ForEach-Object { $name, $value = $_.Split('='); [System.Environment]::SetEnvironmentVariable($name,$value) }
```

---

## 5. Running the Pipeline

### Step 1 — Index the Knowledge Base
```bash
python agentic_rag/index_kb.py
```
This will:
- Load `self_critique_loop_dataset.json`
- Generate embeddings using Gemini embeddings
- Upsert the KB into Pinecone

### Step 2 — Run the Agentic RAG Workflow
```bash
python run.py
```
- Invokes the LangGraph workflow
- Retrieves KB snippets
- Generates an answer
- Critiques and refines if necessary
- Logs each step via `logging` and MLflow (if configured)

---

## 6. Logging & MLflow

- **Python Logging**: Each step (retrieval, generation, critique, refinement) prints info to console.
- **MLflow Tracking** (optional):
  - Nested runs for each node:
    - `Retrieve_Step`
    - `Generate_Step`
    - `Critique_Step`
    - `Refine_Step`
  - Logs parameters like question, retrieved docs, critique, final answer.
  - Saves artifacts like `initial_answer.txt` and `refined_answer.txt`.

**MLflow Run Hierarchy Example:**
```
Parent Run: Agentic_RAG_Run
└── Retrieve_Step
└── Generate_Step
└── Critique_Step
└── Refine_Step
```

---
---

## 7. Notes

- Keep **temperature=0** in LLM calls for deterministic results.
- Each answer **must cite KB snippets** as `[KBxxx]`.
- Maximum **1 refinement per query** for simplicity.
- Compatible with **Python 3.10+**.
