# Paper 0.2 authoring contract

## Scope

Paper 0.2 develops the heterogeneous-network extension of Paper 0.1. Its unit of analysis is a resource-bounded network of non-equivalent humans, learned models, deterministic tools, services, and memory systems.

## Required distinctions

- Treat `G=(V,E)` as a typed, stateful computational graph. A route is not generally ordinary function composition.
- Index capability, reliability, cost, and latency by task distribution, interface, and budget where relevant.
- Call only finite measured behavior a **system-level closure signature**. Never infer mathematical closure from a finite frontier.
- Identify where task-defining computation occurs: controller, local primitive, closure service, interpreter, verifier, or human.
- Separate representability, route acquisition, execution reliability, verification coverage, and efficiency.
- Humans, LLMs, APIs, databases, and interpreters may share an interface-level node abstraction; this never implies internal, cognitive, moral, or phenomenological equivalence.
- Distributed cognition is a system-level analytical stance. Do not present it as settling the extended-mind thesis or consciousness.
- More nodes, agents, calls, or tools do not automatically increase capability or intelligence.

## Evidence discipline

- Label proposals, examples, measured results, and formal consequences separately.
- State distributions, budgets, routing policies, interface schemas, retries, and verifier coverage for empirical claims.
- Treat products of component reliabilities only as independent-error baselines; report correlated failures, retries, and escalation.
- Preserve the distinction between a local tool and a tool that delegates the benchmark-defining search or closure.
- Do not import changing cross-paper results without checking their committed artifacts and status.

## Build and review

- Build `paper0_2.tex` with `latexmk -pdf -interaction=nonstopmode -halt-on-error`.
- Keep native TikZ diagrams legible at 100% zoom and provide prose that stands without the figure.
- Resolve overfull boxes, undefined references/citations, and substantive LaTeX warnings.
- Visually inspect the title page, the network figure, formal definitions, tables, and bibliography.

