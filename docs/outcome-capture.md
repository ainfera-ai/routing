# Outcome capture (methodology §16)

Routing intelligence compounds only when production traffic records **which cell was served** and **what happened** (latency, cost, quality proxy, errors).

## What ships today (audit-backed)

Every inference through `POST /v1/inference`, `/v1/chat/completions`, or `/v1/messages` writes hash-chained audit events, including:

| Event | Payload hints |
| --- | --- |
| `inference.routed` | Model slug, router (`ainfera-mithril` when applicable; `ainfera-auto` is its silent alias and is reported as `ainfera-mithril` in the audit chain) |
| `provider.responded` / `receipt.created` | Tokens, cost, provider |
| `inference.rejected_*` | Cap / funds refusal with policy context |

These events are the **v0 outcome pipe** — verifiable offline via `ainfera-verify` and `GET /v1/audit/{agent_id}/verify`.

## Cell coordinates (target schema)

Methodology §16 defines capture keys:

```
task_type × model × constraint_band → {latency_ms, cost_usd, quality_proxy, outcome_class}
```

`constraint_band` derives from the tenant routing policy template (latency-first, cost-first, compliance-first) in [`templates/`](../templates/).

## Next instrumentation (not blocking adapter GTM)

1. Structured `routing.outcome` audit payload on every dispatch (selected slug + band + fallback reason)
2. Rollup table for Tier-1 / Tier-2 cell coverage dashboard (AIN-208 acceptance)
3. Export hook for corporate `q_empirical` (excludes segregated fleet agents per STRATEGY.md)

Until (1) lands, treat **audit chain + routing policy YAML** as the canonical capture surface for external adapters.
