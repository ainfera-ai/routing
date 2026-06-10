# Ainfera Routing

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

**The Inference of AI Agents.** Agent-native inference routing — constrained optimization across 50+ models, not a billing gateway.

Ainfera Routing is the control plane for production agent workloads: pick the highest-quality model that satisfies hard budget and latency limits, dispatch, capture outcomes, and compound routing intelligence from real traffic.

## Why this exists

Gateways like LiteLLM, OpenRouter, and Portkey optimize single calls for human teams. Ainfera Routing targets autonomous agents: hard caps, workflow-aware traces, identity-bound receipts, and an outcome loop that only compounds when traffic flows through Ainfera.

## Quick links

| Resource | URL |
| --- | --- |
| Methodology (Notion) | [Ainfera Routing — Methodology v1.0](https://www.notion.so/366b49507d6c8168bc85db981a59b9dd) |
| API | [api.ainfera.ai](https://api.ainfera.ai) |
| Python SDK | [ainfera-ai/sdk](https://github.com/ainfera-ai/sdk) |
| Specs | [ainfera-ai/specs](https://github.com/ainfera-ai/specs) |
| MCP | [mcp.ainfera.ai](https://mcp.ainfera.ai) |

## Starter routing templates

Copy a template into your tenant routing policy or agent config:

| Template | Goal | File |
| --- | --- | --- |
| Latency-first | Minimize p95 under quality floor | [`templates/latency-first.yaml`](templates/latency-first.yaml) |
| Cost-first | Minimize cost under quality floor | [`templates/cost-first.yaml`](templates/cost-first.yaml) |
| Compliance-first | Strict veto + no fallback | [`templates/compliance-first.yaml`](templates/compliance-first.yaml) |

Policy shape is documented in [`schema/routing-policy.schema.json`](schema/routing-policy.schema.json).

## Minimal curl

```bash
export AINFERA_API_KEY="ainfera_..."

curl -sS https://api.ainfera.ai/v1/inference \
  -H "Authorization: Bearer $AINFERA_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "ainfera-inference",
    "messages": [{"role": "user", "content": "Route this agent call"}],
    "max_tokens": 256
  }'
```

Use `model: "ainfera-inference"` to let Ainfera route across the live catalog. (`ainfera-mithril` / `ainfera-auto` are legacy aliases and still resolve identically.) See [docs](https://ainfera.ai/docs) for Agent Cards, workflows, and audit verification.

## Production E2E smoke

Full L1–L5 verification against live `api.ainfera.ai` (~3 min):

```bash
git clone https://github.com/ainfera-ai/routing
cd routing
./scripts/ainfera-e2e.sh          # human-readable
MODE=json ./scripts/ainfera-e2e.sh   # agent-parseable
```

Spec: [`scripts/AINFERA-E2E.md`](scripts/AINFERA-E2E.md)

Outcome capture (methodology §16): [`docs/outcome-capture.md`](docs/outcome-capture.md)

## Repo map (ainfera-ai org)

| Repo | Role |
| --- | --- |
| **routing** (this repo) | Public methodology, templates, onboarding |
| api.ainfera.ai | L2 routing runtime + `/v1/inference` (service) |
| [sdk](https://github.com/ainfera-ai/sdk) | Python client |
| [specs](https://github.com/ainfera-ai/specs) | Ontology + API contracts (ATS/AAMC deprecated, AIN-243) |
| [ainfera-mcp](https://github.com/ainfera-ai/ainfera-mcp) | MCP reference adapter |
| Framework adapters | `ainfera-mcp`, `ainfera-hermes`, `ainfera-openclaw`, `ainfera-letta`, `ainfera-langgraph`, `ainfera-langchain`, `ainfera-llamaindex`, `ainfera-crewai`, `ainfera-google-adk` |

## License

Apache 2.0 — see [LICENSE](LICENSE).
