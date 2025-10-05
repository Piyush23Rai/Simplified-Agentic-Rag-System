"""
agentic_rag_simplified.py
-------------------------
Defines a simplified Agentic RAG system using LangGraph with:
- Retriever
- Answer generator
- Self-critique
- Refinement loop
"""

import os
import mlflow
import logging
from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, END
from pydantic import BaseModel, Field
from typing import List

# ---------- CONFIG ----------
INDEX_NAME = "agentic-rag-kb"
MODEL_NAME = "gemini-2.0-flash"
EMBED_MODEL = "models/gemini-embedding-001"
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")

# ---------- LOGGING ----------
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------- STATE ----------
class AgentState(BaseModel):
    question: str
    retrieved_snippets: List[str] = Field(default_factory=list)
    initial_answer: str = ""
    critique: str = ""
    refined_answer: str = ""

# ---------- NODES ----------
def retrieve_kb(state: AgentState):
    mlflow.log_param("step", "retrieve_kb")
    embeddings = GoogleGenerativeAIEmbeddings(model=EMBED_MODEL)
    vectorstore = PineconeVectorStore(index_name=INDEX_NAME, embedding=embeddings)
    results = vectorstore.similarity_search(state.question, k=5)

    snippets = [r.page_content for r in results]
    logger.info(f"Retrieved {len(snippets)} snippets.")
    return state.copy(update={"retrieved_snippets": snippets})

def generate_answer(state: AgentState):
    mlflow.log_param("step", "generate_answer")
    llm = init_chat_model(MODEL_NAME, model_provider="google_genai", temperature=0)
    prompt = f"Question: {state.question}\nSnippets:\n" + "\n".join(state.retrieved_snippets)
    response = llm.invoke(prompt)
    return state.copy(update={"initial_answer": response.content})

def critique_answer(state: AgentState):
    mlflow.log_param("step", "critique_answer")
    llm = init_chat_model(MODEL_NAME, model_provider="google_genai", temperature=0)
    prompt = f"Answer:\n{state.initial_answer}\n\nBased on the snippets, is the answer COMPLETE or should it be REFINED?"
    response = llm.invoke(prompt)
    return state.copy(update={"critique": response.content})


def refine_answer(state: AgentState):
    mlflow.log_param("step", "refine_answer")
    if "COMPLETE" in state.critique.upper():
        logger.info("Answer deemed COMPLETE.")
        return state

    embeddings = GoogleGenerativeAIEmbeddings(model=EMBED_MODEL)
    vectorstore = PineconeVectorStore(index_name=INDEX_NAME, embedding=embeddings)
    extra_results = vectorstore.similarity_search(state.question, k=1)

    new_snippets = state.retrieved_snippets + [r.page_content for r in extra_results]
    llm = init_chat_model(MODEL_NAME, model_provider="google_genai", temperature=0)
    prompt = f"Refine this answer using all snippets:\nQuestion: {state.question}\nSnippets:\n" + "\n".join(new_snippets)
    response = llm.invoke(prompt)

    return state.copy(update={"refined_answer": response.content})

# ---------- GRAPH ----------
def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("retrieve_kb", retrieve_kb)
    graph.add_node("generate_answer", generate_answer)
    graph.add_node("critique_answer", critique_answer)
    graph.add_node("refine_answer", refine_answer)

    graph.set_entry_point("retrieve_kb")
    graph.add_edge("retrieve_kb", "generate_answer")
    graph.add_edge("generate_answer", "critique_answer")
    graph.add_conditional_edges("critique_answer",
        lambda s: "refine_answer" if "REFINE" in s.critique.upper() else END,
        {"refine_answer": "refine_answer", END: END}
    )
    return graph.compile()
