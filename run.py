"""
run.py
------
Executes the full Agentic RAG pipeline with nested MLflow runs for each question and step.
"""

import mlflow
import time
import logging
from agentic_rag.index_kb import create_index, index_kb
from agentic_rag.agentic_rag_simplified import build_graph, AgentState

# ---------- Logging ----------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------- Questions ----------
QUESTIONS = [
    "What are best practices for caching?",
    "How should I set up CI/CD pipelines?",
    "What are performance tuning tips?",
    "How do I version my APIs?",
    "What should I consider for error handling?"
]

def run_agentic_rag():
    """Run the agentic RAG workflow for all questions with nested MLflow logging."""
    mlflow.set_experiment("Agentic RAG Gemini")

    with mlflow.start_run(run_name="Agentic-RAG-Run"):

        # Step 1: Index KB
        logger.info("Creating and indexing KB...")
        with mlflow.start_run(run_name="index_kb", nested=True):
            try:
                create_index(delete_index=False)
                index_kb()
                mlflow.log_metric("indexing_status", 1)
            except Exception as e:
                logger.error(f"Indexing failed: {e}")
                mlflow.log_metric("indexing_status", 0)
                mlflow.log_param("indexing_error", str(e))
                raise

        # Step 2: Build workflow
        workflow = build_graph()

        # Step 3: Process questions
        for i, question in enumerate(QUESTIONS, start=1):
            logger.info(f"Running Question {i}: {question}")
            with mlflow.start_run(run_name=f"Question-{i}", nested=True):
                mlflow.log_param("question_text", question)
                question_start = time.time()

                try:
                    init_state = AgentState(question=question)
                    # Step 3a: Retrieve KB snippets
                    with mlflow.start_run(run_name="retrieve_kb", nested=True):
                        retrieved_state = workflow.nodes["retrieve_kb"].invoke(init_state)
                        snippets_count = len(retrieved_state.retrieved_snippets)
                        mlflow.log_metric("retrieved_snippets", snippets_count)
                        logger.info(f"Retrieved {snippets_count} snippets.")

                    # Step 3b: Generate initial answer
                    with mlflow.start_run(run_name="generate_answer", nested=True):
                        answer_state = workflow.nodes["generate_answer"].invoke(retrieved_state)
                        initial_answer = answer_state.initial_answer
                        mlflow.log_text(initial_answer, "initial_answer.txt")

                    # Step 3c: Critique answer
                    with mlflow.start_run(run_name="critique_answer", nested=True):
                        critique_state = workflow.nodes["critique_answer"].invoke(answer_state)
                        critique_text = critique_state.critique
                        mlflow.log_text(critique_text, "critique.txt")
                        logger.info(f"Critique result: {critique_text}")

                    # Step 3d: Refine if needed
                    final_answer = initial_answer
                    if critique_text.startswith("REFINE"):
                        with mlflow.start_run(run_name="refine_answer", nested=True):
                            refined_state = workflow.nodes["refine_answer"].invoke(critique_state)
                            final_answer = refined_state.refined_answer
                            mlflow.log_text(final_answer, "refined_answer.txt")

                    # Log final answer
                    mlflow.log_text(final_answer, "final_answer.txt")
                    elapsed = time.time() - question_start
                    mlflow.log_metric("question_runtime_sec", elapsed)

                    print(f"\n=============================\nQuestion {i}: {question}\n=============================")
                    print(f"Final Answer:\n{final_answer}\n")

                except Exception as e:
                    logger.error(f"Error processing Question {i}: {e}")
                    mlflow.log_param("question_error", str(e))
                    continue

if __name__ == "__main__":
    run_agentic_rag()
