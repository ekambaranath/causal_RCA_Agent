# 🔬 Causal RCA Agent

> **A production-grade agentic AI system that finds the true root cause of incidents using causal reasoning — not correlation. 7 specialist agents. 8 agentic practices. Zero cloud cost.**

---

## 🔍 What Problem Does It Solve?

Every observability tool shows you *what* broke. None tell you *why*.

- Engineers spend 70% of incident time on root cause analysis
- Correlation-based AI surfaces noise — CPU spike correlates with errors, but which caused which?
- No system reasons about *interventions*: "if this deploy hadn't happened, would the failure still occur?"

**Causal RCA Agent fixes this:**

| Problem | Solution |
|---|---|
| Correlation ≠ causation | PC algorithm builds a causal DAG, not a correlation matrix |
| "What failed" not "why" | Counterfactual engine asks: `do(candidate=removed)` → does failure disappear? |
| Every incident from scratch | Memory agent matches past incidents in <8 seconds |
| 20 alerts for 1 root cause | Critic + ranker deliver one ranked list, downstream effects dismissed |
| Silent model failures | Critic quality gate rejects unsupported verdicts and retries |

---

## 🏗️ Architecture — 7 Agents

```
Incident
    │
    ▼
┌─────────────────────┐
│  🧠 Supervisor Agent  │  Plans, routes, escalates
└──────────┬──────────┘
     ┌─────┴──────┐
     │ (parallel)  │
     ▼             ▼
┌─────────┐  ┌──────────┐
│ 📡 Evidence│  │ 🧠 Memory │   ← runs simultaneously
│ Collector │  │   Agent   │
└─────┬────┘  └─────┬────┘
      └──────┬───────┘
             │ memory hit? → fast-path
             ▼
     ┌───────────────┐
     │ 🔬 Causal      │  PC algorithm → Causal DAG
     │ Discovery      │
     └──────┬─────────┘
            ▼
     ┌───────────────┐
     │ 🔎 Counterfactual│  ReAct loop: do() operator
     │     Agent      │  per candidate until confidence > 0.70
     └──────┬─────────┘
            ▼
     ┌───────────────┐    ❌ FAIL
     │  🛡️ Critic     │──────────────► Increment Retry → loop back
     │  Quality Gate  │
     └──────┬─────────┘ ✅ PASS
            ▼
     ┌───────────────┐
     │ 📋 Intervention │  Structured JSON output
     │    Ranker      │
     └──────┬─────────┘
            ▼
     ┌───────────────┐
     │ 🧠 Supervisor  │  Finalizes verdict or escalates
     │   Finalize     │
     └──────┬─────────┘
            ▼
       Final Answer
```

---

## ✅ Agentic Practices Demonstrated

| Practice | Where |
|---|---|
| **ReAct loop** | Counterfactual Agent — Reason → Act → Observe → Repeat |
| **Tool augmentation** | Causal Discovery — algorithm as tool, LLM interprets |
| **Critic / self-reflection** | Critic Agent — quality gate, rejects unsupported verdicts |
| **Supervisor / worker** | Supervisor orchestrates 6 specialist workers |
| **Long-term memory** | Memory Agent — ChromaDB, fast-path short-circuit |
| **Parallel execution** | Evidence Collector ∥ Memory Agent |
| **Structured output** | Intervention Ranker — enforced JSON schema |
| **Human escalation** | Supervisor Finalize — escalate flag when confidence < 0.70 |
| **State-safe retry** | `node_increment_retry` node — never in edge routing functions |

---

## ⚡ Quick Start

### 1. Setup (installs everything including Ollama + models)

```bash
chmod +x scripts/setup.sh scripts/run.sh
./scripts/setup.sh
```

### 2. Run

```bash
./scripts/run.sh
```

Open **port 8000** in Codespaces when prompted.

### 3. Try it

Open the UI and try one of the sample incidents, or type your own:

- *"auth service error rate spiked after deployment of v2.1"*
- *"database connection pool exhausted causing timeouts"*
- *"payments service memory leak causing GC pauses and cascading failures"*

---

## 💰 Cost

| Resource | Cost |
|---|---|
| Ollama (tinyllama + phi3) | **FREE** — runs locally |
| ChromaDB | **FREE** — local disk |
| Everything else | **FREE** |
| **Total per analysis** | **$0.00** |

---

## 📁 Project Structure

```
causal-rca-agent/
├── src/
│   ├── api.py                      # FastAPI server
│   ├── agents/
│   │   ├── supervisor.py           # Orchestrator — plans, routes, finalizes
│   │   ├── evidence_collector.py   # Gathers telemetry signals
│   │   ├── memory_agent.py         # ChromaDB incident memory
│   │   ├── causal_discovery.py     # Builds causal DAG
│   │   ├── counterfactual_agent.py # ReAct loop — tests candidates
│   │   ├── critic.py               # Quality gate — PASS/FAIL
│   │   └── intervention_ranker.py  # Ranked JSON action list
│   ├── chains/
│   │   └── graph.py                # LangGraph pipeline assembly
│   ├── tools/
│   │   ├── telemetry_tools.py      # Mock telemetry (swap for real OTel)
│   │   ├── causal_tools.py         # Causal DAG + counterfactual engine
│   │   └── memory_tools.py         # ChromaDB read/write
│   └── utils/
│       ├── config.py               # All settings via env vars
│       ├── logging.py              # Structured logging
│       └── state.py                # LangGraph shared state schema
├── static/index.html               # Chat UI
├── tests/test_pipeline.py          # Smoke tests
├── scripts/
│   ├── setup.sh                    # One-time install
│   └── run.sh                      # Start server
├── .devcontainer/devcontainer.json # Codespaces config
├── .env.example                    # Environment template
└── requirements.txt
```

---

## ⚙️ Configuration

```bash
OLLAMA_MODEL=tinyllama          # Small model (critic, ranker, memory)
OLLAMA_MODEL_LARGE=phi3         # Large model (supervisor, causal, counterfactual)
MAX_RETRIES=3                   # Max critic retry cycles
CONFIDENCE_THRESHOLD=0.70       # Min confidence before escalating to human
MEMORY_SIMILARITY_THRESHOLD=0.80 # Min ChromaDB cosine similarity for fast-path
```

---

## 🧪 Tests

```bash
python -m pytest tests/ -v
```
