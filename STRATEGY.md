# Ainfera strategy — routing control plane (AIN-195)

Logged 2026-05-19 · consolidated 2026-05-20.

## One-line positioning

Ainfera makes AI agents economically and operationally viable in production by owning the **routing intelligence layer** for inference — not a billing gateway, not a model catalog.

## What we build

| Layer | Owner | Notes |
| --- | --- | --- |
| **Ainfera Routing** | `ainfera-ai/routing` + API L2 | Constrained optimization: max quality s.t. budget + latency caps |
| **Identity + audit** | API L1/L4 | Agent Cards, hash-chained receipts, offline verify |
| **Adapters** | `ainfera-*` repos | Thin wrappers → one OpenAI-compat core (`/v1/chat/completions`, `/v1/inference`) |
| **Ainfera OS** | `ainfera-os` | Five fleet agents dogfood adapters before external GTM |
| **Varda** | Fleet orchestration | Spec → Linear → ship discipline |

## What we defer

- **Payment rails** (CDP, Stripe, x402 top-up) — blocked on SG incorporation
- **Manwe first-user** (AIN-111) — founder-run migration, not agent-deliverable

## Adapter execution order

1. **MCP** (`mcp.ainfera.ai`) — reference: `cloudflare/smoke-mcp.sh`
2. **Fleet dogfood** — OpenClaw, Hermes (founder), [Letta](https://github.com/ainfera-ai/ainfera-letta) (Namo), LangGraph
3. **GTM-only** — LlamaIndex, CrewAI, ADK — publish after MCP pattern proven

Each adapter ships with `curl-example.sh`: signup → inference → audit verify.

## Canonical methodology

[Ainfera Routing — Methodology v1.1](https://www.notion.so/366b49507d6c8168bc85db981a59b9dd)
