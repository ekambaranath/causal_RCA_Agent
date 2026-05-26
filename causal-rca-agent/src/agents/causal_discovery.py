"""
Causal Discovery Agent — Algorithm-augmented agent

Agentic practices demonstrated:
  - Algorithm as tool: causal DAG built by algorithm, LLM interprets output
  - Tool augmentation: LLM decides algorithm parameters, reads DAG output
"""

from typing import Any, Dict

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage

from src.tools.causal_tools import build_causal_dag
from src.utils.config import settings
from src.utils.logging import get_logger
from src.utils.state import RCAState

log = get_logger("causal_discovery")


def _build_llm() -> ChatOllama:
    return ChatOllama(
        base_url=settings.ollama_base_url,
        model=settings.ollama_model_large,
        num_predict=256,
        temperature=0.0,
        timeout=settings.llm_timeout,
    )


def node_causal_discovery(state: RCAState) -> Dict[str, Any]:
    """
    Runs causal DAG algorithm on evidence, then asks the LLM to
    interpret the resulting graph in plain English.
    """
    evidence = state["evidence"]
    log.info("causal_discovery_start")

    # Step 1: Algorithm builds the causal DAG (no LLM involved here)
    dag = build_causal_dag(evidence)
    log.info(
        "dag_built",
        nodes=len(dag["nodes"]),
        edges=len(dag["edges"]),
        root_candidates=dag["root_candidates"],
        spurious_removed=dag["spurious_removed"],
    )

    # Step 2: LLM interprets the DAG output in plain English
    llm = _build_llm()

    edges_summary = "\n".join(
        f"  {e['cause_svc']} → {e['effect_svc']} "
        f"(gap: {e['gap_min']}min, confidence: {e['confidence']})"
        for e in dag["edges"][:6]
    )
    candidates = ", ".join(dag["root_candidates"][:3]) or "none identified"

    prompt = (
        f"You are an SRE analysing a causal graph from a production incident.\n\n"
        f"Causal edges discovered:\n{edges_summary}\n\n"
        f"Root cause candidates: {candidates}\n"
        f"Spurious correlations eliminated: {dag['spurious_removed']}\n\n"
        f"In 2 sentences: what does this causal graph tell us about the incident? "
        f"Which candidate is most suspicious and why?"
    )

    response    = llm.invoke([HumanMessage(content=prompt)])
    interpretation = response.content.strip()

    log.info("causal_discovery_complete", interpretation=interpretation[:80])

    return {
        "causal_dag": dag,
        "steps": [
            f"🔬 Causal Discovery: {len(dag['edges'])} causal edges, "
            f"{dag['spurious_removed']} spurious removed. "
            f"Candidates: {', '.join(dag['root_candidates'][:2])}. "
            f"{interpretation}"
        ],
    }
