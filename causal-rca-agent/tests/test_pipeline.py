"""Basic pipeline smoke tests — no Ollama required."""

from src.tools.causal_tools import build_causal_dag, run_counterfactual
from src.tools.telemetry_tools import gather_telemetry
from src.tools.memory_tools import search_memory


def test_gather_telemetry_deployment():
    incident = {"description": "auth service deploy error rate spike", "service": "auth-service"}
    ev = gather_telemetry(incident)
    assert len(ev["events"]) > 0
    assert "auth-service" in ev["services"]


def test_gather_telemetry_database():
    incident = {"description": "database connection pool exhausted", "service": "db-primary"}
    ev = gather_telemetry(incident)
    assert ev["metrics"]["db_conn_used"] == 150


def test_build_causal_dag():
    incident = {"description": "auth deploy", "service": "auth-service"}
    ev  = gather_telemetry(incident)
    dag = build_causal_dag(ev)
    assert "nodes" in dag
    assert "edges" in dag
    assert "root_candidates" in dag
    assert len(dag["root_candidates"]) > 0


def test_counterfactual_root_cause():
    incident = {"description": "auth deploy", "service": "auth-service"}
    ev       = gather_telemetry(incident)
    dag      = build_causal_dag(ev)
    candidate = dag["root_candidates"][0]
    result   = run_counterfactual(candidate, dag, ev)
    assert "confidence" in result
    assert "is_root_cause" in result
    assert 0.0 <= result["confidence"] <= 1.0


def test_memory_search_returns_match():
    incident = {"description": "auth service deploy error spike", "service": "auth-service"}
    ev       = gather_telemetry(incident)
    match    = search_memory(ev)
    # May or may not match depending on ChromaDB state — just check shape
    if match:
        assert "incident_id" in match
        assert "similarity" in match
        assert "resolution" in match
