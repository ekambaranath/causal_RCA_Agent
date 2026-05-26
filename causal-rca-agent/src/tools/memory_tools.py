"""
Incident memory using ChromaDB.

Stores causal signatures of resolved incidents and searches for
similar past incidents to short-circuit the full causal pipeline.
"""

import hashlib
import json
import time
from typing import Any, Dict, Optional

import chromadb
from chromadb.utils import embedding_functions

from src.utils.config import settings
from src.utils.logging import get_logger

log = get_logger("memory_tools")

_client: Optional[chromadb.PersistentClient] = None
_collection = None


def _get_collection():
    global _client, _collection
    if _collection is None:
        _client = chromadb.PersistentClient(path=settings.chroma_persist_dir)
        ef = embedding_functions.DefaultEmbeddingFunction()
        _collection = _client.get_or_create_collection(
            name="incident_memory",
            embedding_function=ef,
            metadata={"hnsw:space": "cosine"},
        )
        _seed_memory()
    return _collection


def _seed_memory():
    """Pre-load known incident patterns so the system works out of the box."""
    col = _collection
    if col.count() > 0:
        return

    seeds = [
        {
            "id": "incident_247",
            "description": "auth service deploy jwt validation latency downstream timeouts error rate spike",
            "meta": {
                "root_cause":   "auth-service deployment v2.1 introduced JWT validation regression",
                "resolution":   "Revert auth-service to v2.0.9 and increase JWT cache TTL to 300s",
                "ttr_minutes":  4,
                "confidence":   0.97,
                "incident_type": "deployment",
            },
        },
        {
            "id": "incident_198",
            "description": "database connection pool exhausted analytics job db primary order payment service timeouts",
            "meta": {
                "root_cause":   "Analytics reporting job opened 140 connections, exhausting pool limit of 150",
                "resolution":   "Kill analytics job and increase DB connection pool limit to 300",
                "ttr_minutes":  12,
                "confidence":   0.91,
                "incident_type": "database",
            },
        },
        {
            "id": "incident_312",
            "description": "payments service memory leak heap growth gc pause request timeout cascade frontend checkout failures",
            "meta": {
                "root_cause":   "Memory leak in payments-service payment processor thread pool",
                "resolution":   "Rolling restart of payments-service pods; add heap limit -Xmx512m",
                "ttr_minutes":  6,
                "confidence":   0.88,
                "incident_type": "memory_leak",
            },
        },
    ]

    for s in seeds:
        col.add(
            ids=[s["id"]],
            documents=[s["description"]],
            metadatas=[{k: str(v) for k, v in s["meta"].items()}],
        )
    log.info("memory_seeded", count=len(seeds))


def search_memory(evidence: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Build a signature string from evidence and search ChromaDB.
    Returns the best match if similarity > MEMORY_SIMILARITY_THRESHOLD.
    """
    col = _get_collection()

    events   = evidence.get("events", [])
    services = evidence.get("services", [])
    deploys  = evidence.get("deploys", [])

    event_keywords = " ".join(
        e["event"].replace("_", " ") for e in events
    )
    deploy_keywords = " ".join(
        f"deploy {d['service']}" for d in deploys
    )
    signature = f"{event_keywords} {deploy_keywords} {' '.join(services)}"

    results = col.query(query_texts=[signature], n_results=1)
    if not results["ids"] or not results["ids"][0]:
        return None

    distance   = results["distances"][0][0]
    similarity = round(1 - distance, 2)
    threshold  = settings.memory_similarity_threshold

    log.info("memory_search", similarity=similarity, threshold=threshold)

    if similarity < threshold:
        return None

    meta = results["metadatas"][0][0]
    return {
        "incident_id":   results["ids"][0][0],
        "similarity":    similarity,
        "root_cause":    meta.get("root_cause"),
        "resolution":    meta.get("resolution"),
        "ttr_minutes":   int(meta.get("ttr_minutes", 0)),
        "confidence":    float(meta.get("confidence", 0)),
        "incident_type": meta.get("incident_type"),
    }


def store_incident(
    incident: Dict[str, Any],
    evidence: Dict[str, Any],
    verdict: Dict[str, Any],
) -> str:
    """Persist a resolved incident into ChromaDB for future matching."""
    col  = _get_collection()
    uid  = hashlib.md5(
        f"{incident.get('id', '')}_{time.time()}".encode()
    ).hexdigest()[:12]
    iid  = f"incident_{uid}"

    events   = evidence.get("events", [])
    services = evidence.get("services", [])
    text     = " ".join(e["event"].replace("_", " ") for e in events)
    text    += " " + " ".join(services)

    col.add(
        ids=[iid],
        documents=[text],
        metadatas=[{
            "root_cause":    str(verdict.get("root_cause", "")),
            "resolution":    str(verdict.get("intervention", "")),
            "ttr_minutes":   str(verdict.get("ttr_minutes", 0)),
            "confidence":    str(verdict.get("confidence", 0)),
            "incident_type": str(incident.get("type", "unknown")),
        }],
    )
    log.info("memory_stored", incident_id=iid)
    return iid
