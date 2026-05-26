"""
Mock telemetry tools.

In production replace with real OpenTelemetry / Prometheus / k8s API calls.
The incident_type keyword drives which realistic template is returned.
"""

import time
from typing import Any, Dict


def _ts(seconds_ago: int) -> float:
    return time.time() - seconds_ago


# ── Incident templates ────────────────────────────────────────────────────────

TEMPLATES: Dict[str, Dict[str, Any]] = {
    "deployment": {
        "events": [
            {"ts": _ts(260), "service": "ci-cd",            "event": "deploy_auth_service_v2.1_started",      "type": "deploy"},
            {"ts": _ts(240), "service": "ci-cd",            "event": "deploy_auth_service_v2.1_completed",    "type": "deploy"},
            {"ts": _ts(200), "service": "auth-service",     "event": "jwt_validation_latency_spike_340ms",   "type": "metric"},
            {"ts": _ts(195), "service": "api-gateway",      "event": "upstream_timeout_to_auth_service",     "type": "error"},
            {"ts": _ts(188), "service": "checkout-service", "event": "dependency_timeout_on_auth",           "type": "error"},
            {"ts": _ts(180), "service": "checkout-service", "event": "error_rate_breach_5pct",               "type": "alert"},
        ],
        "metrics":  {"error_rate_pct": 5.2, "p99_latency_ms": 2340, "p50_latency_ms": 890, "rps": 1420},
        "deploys":  [{"service": "auth-service", "version": "v2.1", "minutes_ago": 4, "rolled_back": False}],
        "services": ["auth-service", "api-gateway", "checkout-service"],
    },
    "database": {
        "events": [
            {"ts": _ts(420), "service": "analytics-job",   "event": "analytics_report_job_started",          "type": "job"},
            {"ts": _ts(380), "service": "db-primary",      "event": "connection_pool_at_90pct_capacity",     "type": "metric"},
            {"ts": _ts(360), "service": "db-primary",      "event": "connection_pool_exhausted_150_of_150",  "type": "alert"},
            {"ts": _ts(355), "service": "order-service",   "event": "db_query_timeout_p99_820ms",            "type": "error"},
            {"ts": _ts(348), "service": "payment-service", "event": "db_query_timeout_p99_950ms",            "type": "error"},
            {"ts": _ts(340), "service": "user-service",    "event": "db_connection_refused",                 "type": "error"},
        ],
        "metrics":  {"error_rate_pct": 8.7, "db_conn_used": 150, "db_conn_limit": 150, "p99_latency_ms": 820},
        "deploys":  [],
        "services": ["analytics-job", "db-primary", "order-service", "payment-service"],
    },
    "memory_leak": {
        "events": [
            {"ts": _ts(900), "service": "payments-service", "event": "heap_memory_growth_4mb_per_min",       "type": "metric"},
            {"ts": _ts(720), "service": "payments-service", "event": "heap_memory_at_78pct_capacity",        "type": "metric"},
            {"ts": _ts(480), "service": "payments-service", "event": "gc_pause_duration_1200ms",             "type": "metric"},
            {"ts": _ts(420), "service": "payments-service", "event": "request_processing_timeout",           "type": "error"},
            {"ts": _ts(360), "service": "api-gateway",      "event": "upstream_timeout_payments_service",    "type": "error"},
            {"ts": _ts(300), "service": "frontend",         "event": "checkout_failures_spike",              "type": "alert"},
        ],
        "metrics":  {"error_rate_pct": 3.1, "heap_used_pct": 91, "gc_pause_ms": 1200, "p99_latency_ms": 4100},
        "deploys":  [],
        "services": ["payments-service", "api-gateway", "frontend"],
    },
}


def gather_telemetry(incident: Dict[str, Any]) -> Dict[str, Any]:
    """
    Selects the right mock template based on incident keywords.
    Returns a structured evidence packet.
    """
    desc = (incident.get("description", "") + " " + incident.get("service", "")).lower()

    if any(kw in desc for kw in ["deploy", "version", "release", "rollout", "auth"]):
        template = TEMPLATES["deployment"]
    elif any(kw in desc for kw in ["db", "database", "connection", "pool", "analytics"]):
        template = TEMPLATES["database"]
    elif any(kw in desc for kw in ["memory", "leak", "heap", "gc", "payment"]):
        template = TEMPLATES["memory_leak"]
    else:
        template = TEMPLATES["deployment"]  # default

    return {
        "events":   template["events"],
        "metrics":  template["metrics"],
        "deploys":  template["deploys"],
        "services": template["services"],
        "window_minutes": 15,
        "total_signals": len(template["events"]),
    }
