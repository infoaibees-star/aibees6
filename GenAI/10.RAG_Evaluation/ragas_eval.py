# ============================================================
# RAGAS Evaluation for AIBees IT Helpdesk RAG App
# ============================================================
# What does RAGAS check?
#   1. Faithfulness       - Is the answer based on the context?
#   2. Answer Relevancy   - Does the answer match the question?
#   3. Context Precision  - Are the retrieved chunks relevant?
#   4. Context Recall     - Did we retrieve all needed chunks?
# ============================================================

import os
import streamlit as st
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper


# ── Step 1: Run RAGAS evaluation ─────────────────────────────────────────────
# This function takes the question, the answer the LLM gave,
# the chunks retrieved from FAISS, and the expected correct answer.
# It returns a simple dictionary of scores between 0 and 1.

def run_ragas_evaluation(question, answer, retrieved_chunks, ground_truth, llm, embedding_model):
    """
    Runs RAGAS metrics and returns a dict of scores.

    Parameters:
        question        : The user's question (string)
        answer          : The LLM's answer (string)
        retrieved_chunks: List of text chunks from FAISS similarity search
        ground_truth    : The correct expected answer (string)
        llm             : Your LangChain LLM (e.g. ChatVertexAI)
        embedding_model : Your LangChain embeddings (e.g. VertexAIEmbeddings)

    Returns:
        scores_dict     : A dictionary like {"faithfulness": 0.9, "answer_relevancy": 0.8, ...}
    """

    # RAGAS needs its own wrapped versions of your LLM and embeddings
    ragas_llm        = LangchainLLMWrapper(llm)
    ragas_embeddings = LangchainEmbeddingsWrapper(embedding_model)

    # RAGAS expects the data in a HuggingFace Dataset format
    # "contexts" must be a list of lists — one list of chunks per question
    data = {
        "question":    [question],
        "answer":      [answer],
        "contexts":    [retrieved_chunks],   # e.g. ["chunk1 text", "chunk2 text", ...]
        "ground_truth":[ground_truth],
    }

    dataset = Dataset.from_dict(data)

    # Run all 4 metrics at once
    results = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=ragas_llm,
        embeddings=ragas_embeddings,
    )

    # Convert results to a plain Python dictionary, round to 2 decimal places
    scores = results.to_pandas().fillna(0).to_dict(orient="records")[0]
    scores = {key: round(float(value), 2) for key, value in scores.items()
              if isinstance(value, (int, float))}

    return scores


# ── Step 2: Display scores in Streamlit ──────────────────────────────────────
# This function shows the scores as colored metric cards in your Streamlit UI.

def display_ragas_scores(scores):
    """
    Shows RAGAS scores as Streamlit metric cards with color indicators.

    Green = good (score >= 0.7)
    Orange = okay (score 0.4 - 0.7)
    Red = bad (score < 0.4)
    """

    st.subheader("📊 RAGAS Evaluation Results")

    # Helper to pick an emoji based on score
    def score_emoji(score):
        if score >= 0.7:
            return "🟢"
        elif score >= 0.4:
            return "🟡"
        else:
            return "🔴"

    # Show 4 metric cards in a row
    col1, col2, col3, col4 = st.columns(4)

    faithfulness_score     = scores.get("faithfulness", 0)
    relevancy_score        = scores.get("answer_relevancy", 0)
    precision_score        = scores.get("context_precision", 0)
    recall_score           = scores.get("context_recall", 0)

    col1.metric(
        label=f"{score_emoji(faithfulness_score)} Faithfulness",
        value=faithfulness_score,
        help="Is the answer grounded in the retrieved context? (Higher = better)"
    )
    col2.metric(
        label=f"{score_emoji(relevancy_score)} Answer Relevancy",
        value=relevancy_score,
        help="Does the answer actually address the question? (Higher = better)"
    )
    col3.metric(
        label=f"{score_emoji(precision_score)} Context Precision",
        value=precision_score,
        help="Are the retrieved chunks all relevant? (Higher = better)"
    )
    col4.metric(
        label=f"{score_emoji(recall_score)} Context Recall",
        value=recall_score,
        help="Did we retrieve all chunks needed to answer? (Higher = better)"
    )

    # Show a simple interpretation guide
    with st.expander("📖 How to read these scores"):
        st.markdown("""
        | Score | Meaning |
        |-------|---------|
        | 🟢 0.7 – 1.0 | Good — the RAG pipeline is working well |
        | 🟡 0.4 – 0.7 | Okay — there is room for improvement |
        | 🔴 0.0 – 0.4 | Poor — the pipeline needs attention |

        **Faithfulness** — If this is low, the LLM is making up facts not in the document.
        **Answer Relevancy** — If this is low, the answer is off-topic or too vague.
        **Context Precision** — If this is low, FAISS is returning irrelevant chunks.
        **Context Recall** — If this is low, important chunks are being missed during retrieval.
        """)