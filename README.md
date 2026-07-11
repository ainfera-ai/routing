# Ainfera Routing — Policy Templates & Methodology

**The Inference of AI Agents.** Agent-native inference routing — constrained optimization across models, not a billing gateway.

[![CI](https://github.com/ainfera-ai/routing/actions/workflows/ci.yml/badge.svg)](https://github.com/ainfera-ai/routing/actions)
[![License](https://img.shields.io/badge/license-Apache--2.0-green)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11+-blue)](pyproject.toml)

## Overview

Production-grade policy templates and routing methodology for AI agent inference. This repo defines how Ainfera routes requests to the optimal model based on quality, cost, and latency constraints — with every decision recorded in a signed audit chain.

## What's Here

| Directory | Description |
|---|---|
| [`ainfera_routing/`](ainfera_routing) | Core routing library |
| [`templates/`](templates) | Policy templates for common agent workloads |
| [`schema/`](schema) | JSON schemas for routing policies |
| [`docs/`](docs) | Methodology documentation |
| [`tests/`](tests) | Test suite |
| [`scripts/`](scripts) | Operational scripts |

## Quick Start

```bash
# Install
pip install -e .

# Use a policy template
python -m ainfera_routing --template agentic-coding

# Custom policy
python -m ainfera_routing --config my-policy.yaml
```

## Policy Templates

| Template | Use Case | Strategy |
|---|---|---|
| `agentic-coding` | Code generation, debugging | Quality-first, fallback to cost |
| `research` | Long-context research, analysis | Context-length aware |
| `high-throughput` | Bulk processing | Cost-optimized |
| `balanced` | General purpose | Quality/cost equilibrium |

## Methodology

Ainfera's routing is **outcome-aware**: it learns from observed outcomes (latency, quality, cost) at the orchestration boundary, not from static model labels. See [`docs/`](docs) and [`STRATEGY.md`](STRATEGY.md) for the full methodology.

## License

Apache 2.0 — see [LICENSE](LICENSE).
