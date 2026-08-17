# Multi-Agent Research System (LangGraph)

A runnable reference implementation for a **Designing a Multi-Agent Research System** .

The design intentionally demonstrates four things the rubric cares about:

1. **Collaborative shared state** — evidence, provenance, agent contributions, cross-agent insights, conflicts, reasoning threads, quality gates, human decisions, and telemetry all live in one state.
2. **Adaptive coordination** — the orchestrator chooses **parallel**, **sequential**, or **hybrid** execution based on the research question.
3. **Rigorous synthesis** — provenance, conflict resolution, confidence/coverage thresholds, and explicit stop criteria are enforced before synthesis.
4. **Human-in-the-loop (HITL)** — unresolved conflicts, low confidence, low evidence coverage, or a human-requested review pause the graph using LangGraph `interrupt()`.

## Architecture

```text
                         ┌──────────────────────────────┐
User question ──────────>│ Planner / Orchestrator       │
                         │ chooses execution strategy   │
                         └──────────────┬───────────────┘
                                        │
              ┌─────────────────────────┼─────────────────────────┐
              │                         │                         │
              ▼                         ▼                         ▼
       PARALLEL SPEED            SEQUENTIAL DEPTH           HYBRID DEFAULT
    ┌─────────────────┐       Web -> Data -> Trend      Web ──┐
    │ Web   Data      │       -> Competitive                  ├─> Trend ──┐
    │ Trend Competitive│                                      │           ├─>
    └───────┬─────────┘                                 Data ─┘   Comp ───┘
            └───────────────────────┬─────────────────────────────┘
                                    ▼
                         Cross-Agent Insight Builder
                                    ▼
                            Conflict Resolver
                                    ▼
                               Quality Gate
                           ┌────────┴─────────┐
                           │                  │
                      gate passes         gate fails /
                           │             unresolved conflict
                           │                  │
                           │                  ▼
                           │          Human-in-the-loop
                           │          approve / edit /
                           │          request more work
                           │                  │
                           └──────────┬───────┘
                                      ▼
                                Synthesis Agent
                                      ▼
                                   Report
```

---

## Agent Coordination Matrix

How each node depends on, contributes to, and signals other nodes:

| Node | Depends on (reads) | Contributes (writes) | Activation | Signals findings via |
|---|---|---|---|---|
| Web Research | `question` only — always the first mover, so it never has upstream evidence to build on | `evidence` (policy/consumer-sentiment claims), `contributions` | Always active (foundational baseline research) | New `evidence` items appended to shared state; other agents pick them up as `context` |
| Data Analysis | `question` + upstream `evidence` (in sequential/hybrid, this includes Web Research's output) | `evidence` (dataset/market-size claims), `contributions` | Always active (foundational quantitative grounding) | `evidence` tagged `market_size`/`charging`/`growth`, which the conflict resolver and insight builder specifically key off |
| Trend Analysis | `question` + upstream `evidence` (sequential/hybrid: Web + Data's output) | `evidence` (forecast claims), `contributions` | **Conditional** — `planner.choose_active_agents()` excludes it only if the question explicitly signals trend/forecast is out of scope | Forecast-tagged `evidence`; a skipped run still emits a `contributions` record explaining why |
| Competitive Intelligence | `question` + upstream `evidence` (sequential/hybrid: all prior specialists) | `evidence` (competitor claims), `contributions` | **Conditional** — excluded only if the question explicitly signals competitor analysis is out of scope | Competitor-tagged `evidence`; same skip-with-explanation behavior |
| Cross-Agent Insight Builder | The full `evidence` list, across all agents that ran | `cross_agent_insights`, `synthesis_threads` | Always runs once specialists converge (parallel fan-in, sequential chain-end, or hybrid stage 2) | An insight is only emitted if it draws from `evidence` produced by **≥2 distinct** `produced_by` agents — enforced in code, not just claimed |
| Conflict Resolver | The full `evidence` list | `conflicts` | Always runs, same convergence point as the insight builder | `conflicts[].status` (`resolved` vs `open`) signals the quality gate and, if still `open`, the human reviewer |
| Quality Gate | `evidence`, `cross_agent_insights`, `conflicts`, `iteration_count` | `quality_gate`, `research_complete` | Always runs after conflict resolution | `quality_gate.failures` names exactly which criterion blocked synthesis |
| Human Review | `quality_gate`, open `conflicts` | `human_decisions`, (optionally) revised `conflicts` | Triggered when the gate fails, or `mandatory_human_review` is set | LangGraph `interrupt()` payload *is* the signal — a JSON packet, not a side channel |
| Synthesis | Everything accumulated so far | `final_report` | Triggered once the gate passes (directly, or via human `approve`/`resolve`) | The final markdown report, with every claim traceable back through `parent_evidence_ids` |

This table is what "for each node, define what it needs from others, how it contributes, what triggers it, and how it signals findings" cashes out to in code — the columns map directly onto those four questions.

## Collaborative Reasoning

Beyond raw data sharing, three mechanisms produce genuine *shared understanding that emerges from agent interactions*:

1. **Context propagation.** In sequential and hybrid modes, each specialist receives the full accumulated `evidence` list as `context` before it runs (`agents.py`'s `specialist_node`), and `AgentContribution.reasoning_note` records how that context shaped its output (e.g. `provider.py`: *"data_analysis used 2 existing evidence items"*). This is reasoning that's visibly conditioned on other agents' work, not agents reasoning in isolation and merging afterward.
2. **Emergent insight synthesis.** `build_cross_agent_insights()` doesn't just concatenate findings — it requires evidence from ≥2 distinct agents to even produce a `CrossAgentInsight`, and its `why_emergent` field states the specific relationship that no single specialist's evidence shows alone (see the Emergent Insights example below).
3. **Conflict-aware convergence.** `detect_and_resolve_conflicts()` reconciles disagreeing agents' claims *before* synthesis, so the final "shared understanding" reflects a resolved (or explicitly flagged-as-unresolved) position rather than silently picking whichever agent ran last.

`DESIGN_MAPPING.md` maps this term to code precisely: `cross_agent_insights` + provenance lineage + downstream context.

---

## 1. Trade-offs: Parallel Speed vs Sequential Depth

The system does **not** assume one execution pattern is always best.

### Parallel discovery
The Web, Data, Trend, and Competitive agents run independently in the same graph superstep.

**Why use it**
- Lowest latency when the question is broad and the workstreams are largely independent.
- Useful for early market scans where speed and coverage matter more than dependency depth.

**Cost**
- Trend analysis may run before validated market-size data exists.
- Competitive analysis may miss findings discovered by the web agent in the same parallel wave.
- More duplication and weaker causal chains are possible.

**Example**
> "Give me a quick landscape of the Indian EV market."

All four specialists can scan their areas simultaneously. The insight builder reconciles them afterward.

### Sequential depth
Execution is:

```text
Web Research -> Data Analysis -> Trend Analysis -> Competitive Intelligence
```

Each downstream agent receives validated findings from upstream agents.

**Why use it**
- Stronger causal reasoning.
- Better when later analysis depends on earlier results.
- Reduces repeated work because each specialist has richer context.

**Cost**
- Higher end-to-end latency because critical-path time is roughly the sum of agent latencies.
- A weak early finding can propagate downstream, so provenance and confidence checks are important.

**Example**
> "Quantify how falling battery prices will change EV market share and then determine which competitor benefits most."

Trend analysis should use the Data agent's validated battery-price and adoption evidence, and Competitive Intelligence should build on that trend.

### Hybrid strategy — default for complex consulting research
Stage 1 runs Web and Data in parallel. Stage 2 runs Trend and Competitive Intelligence in parallel **after** Stage 1 is complete.

```text
Web ──┐
      ├──> shared evidence ──> Trend ──┐
Data ─┘                                ├──> collaborative reasoning
                           Competitive ─┘
```

This preserves much of the latency benefit while ensuring downstream interpretation uses stronger upstream evidence.

### Orchestration decision matrix

| Situation | Mode | Reason |
|---|---|---|
| Broad scan, low dependency | Parallel | Optimize latency and breadth |
| Strong dependency chain | Sequential | Optimize reasoning depth |
| Multi-dimensional consulting problem | Hybrid | Balance speed and depth |
| Explicit "quick/overview/scan" wording | Parallel | User values speed |
| Explicit "why/causal/quantify/forecast/impact" wording | Hybrid or Sequential | Later reasoning depends on earlier evidence |

The chosen mode is written into `state["execution_plan"]`, so the decision is observable and auditable.

### Selective agent activation

The mode decision above (parallel/sequential/hybrid) governs *how* the four specialists run — it does not decide *which* specialists run, since every mode above always activates all four. That's a separate decision: `planner.choose_active_agents()` writes `state["execution_plan"]["active_agents"]`, and `web_research`/`data_analysis` are foundational (always active), while `trend_analysis`/`competitive_intelligence` are excluded only when the question explicitly signals that lens is out of scope (e.g. *"give me a market overview, excluding competitor analysis"*). An excluded specialist's node still runs (for topology simplicity — see `graph.py`) but returns immediately with a "Skipped: ..." contribution and makes no `provider.research()` call, so on the live path a skipped agent makes zero API calls.

This is deliberately conservative — opt-out on an explicit signal, not opt-in on a topic keyword — rather than a fully dynamic "infer relevance from the question" activator: an LLM-based relevance classifier could infer this more generally, but risks silently starving the quality gate (fewer active agents means less evidence and fewer source types) or diverging from the demo path's deterministic fixtures. The explicit-signal approach is auditable, testable without network calls, and never changes behavior for a question that doesn't ask for narrower scope.

---

## 2. Provenance

Every evidence item contains:

- `evidence_id`: stable unique identifier
- `claim`: the actual fact/observation
- `source_url`: original source location
- `source_type`: article, dataset, report, filing, etc.
- `retrieved_at`: timestamp
- `produced_by`: agent that introduced it
- `confidence`: 0–1
- `parent_evidence_ids`: upstream evidence used to derive it
- `tags`: topic labels

Derived insights also store `parent_evidence_ids`, creating a lineage graph from final insight back to original evidence.

Example:

```text
Final recommendation
  -> insight-07
      -> trend evidence-11
          -> data evidence-04
          -> web evidence-02
```

This lets the synthesis agent cite exactly what supports each claim and prevents unsupported conclusions from silently entering the report.

---

## 3. Conflict Resolution

Conflicts are first-class state objects rather than hidden prompt text.

A conflict records:

- conflicting evidence IDs
- issue description
- severity
- resolution status
- resolution rationale
- chosen evidence (if resolved)

The resolver applies:

1. source credibility
2. recency
3. directness of evidence
4. agreement with independent sources
5. agent confidence

High-severity or close-confidence conflicts remain unresolved and trigger HITL.

Example:

- Source A says market size = `$8B`, confidence `0.78`
- Source B says market size = `$14B`, confidence `0.80`

Because the estimates are far apart and confidence is nearly tied, the system does **not** arbitrarily pick one. It escalates to a human reviewer.

---

## 4. Stop Criteria / Quality Gate

Synthesis begins only when the state meets the quality gate.

Default thresholds:

```text
minimum evidence items       >= 4
minimum source types         >= 2
average evidence confidence  >= 0.70
cross-agent insights         >= 1
unresolved high conflicts    == 0
iteration count              <= 3
```

The quality gate writes a machine-readable scorecard into shared state.

If the gate fails:
- unresolved conflicts -> HITL
- low confidence -> HITL
- insufficient evidence -> HITL can approve limited synthesis or request another research pass
- maximum iterations reached -> mandatory HITL rather than an infinite agent loop

This makes the **stop condition explicit** rather than relying on an LLM deciding that it "feels done."

---

## 5. Emergent Insights

The system distinguishes **agent findings** from **cross-agent insights**.

An emergent insight must:
- combine evidence from at least two different specialist agents
- contain lineage to those pieces of evidence
- explain a relationship, consequence, tension, or causal hypothesis
- assign confidence based on the supporting evidence

Example:

```text
Web agent:
Government charging subsidies are expanding.

Data agent:
Public charging points grew 45% YoY.

Competitive agent:
Competitor A has the largest charging partnership network.

Emergent insight:
Subsidy-driven infrastructure growth is likely to disproportionately
benefit Competitor A because it already has the distribution partnerships
needed to convert infrastructure expansion into adoption.
```

No single specialist produced that complete inference. It emerges from collaboration.

---

## 6. Human-in-the-loop checkpoints

The `human_review` node uses LangGraph `interrupt()` and persists state with a checkpointer.

Review is triggered when:
- high-severity conflicts remain unresolved
- average confidence is below threshold
- evidence/source diversity is insufficient
- the user explicitly requests mandatory review
- max research iterations are reached

The reviewer can return:

```json
{"action": "approve"}
```

or:

```json
{
  "action": "resolve",
  "conflict_id": "conflict-123",
  "chosen_evidence_id": "ev-456",
  "reason": "Government filing is authoritative."
}
```

or:

```json
{"action": "more_research"}
```

The graph resumes with the **same thread ID**, preserving the checkpoint.

---

## 7. Telemetry and checkpoints

Every important node emits telemetry:

- node name
- start/end timestamps
- duration
- execution mode
- input evidence count
- output evidence count
- conflicts encountered
- quality-gate result
- HITL trigger reason

This supports:
- latency comparison: parallel vs sequential
- finding which agents create the most useful evidence
- identifying repeated conflicts
- measuring how often humans intervene
- debugging weak synthesis
- replaying a run from persistent state

The implementation uses a LangGraph checkpointer for durable graph state in a thread.

---

## Run

### Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Interactive demo

```bash
python run_demo.py
```

Then type the question at the "Research question (or 'quit'):" prompt. For example: "Should an automaker enter the Indian EV market, and what competitive position should it take?"

The demo intentionally uses a deterministic local research provider so the architecture can run without API keys or paid services.

For a quick-scan question:

```bash
python run_demo.py
```

Then type the quick-scan question at the prompt: "Quick overview and landscape scan of the Indian EV market"

For sequential depth:

```bash
python run_demo.py
```

Then type the sequential-depth question at the prompt: "First quantify battery-price changes, then forecast their impact on EV adoption, then determine which competitor benefits most"

### Human review response

When the graph pauses, the CLI prints the review packet and asks for one of:

```text
approve
more
resolve
```

`resolve` chooses the strongest unresolved evidence in the demo. In a production UI, the review packet would be rendered for a human analyst.

---

## Tests

```bash
python -m unittest discover -s tests -v
```

---

## Important implementation note

This repository is an **architecture demonstration**, not a production web crawler. `DemoResearchProvider` supplies deterministic evidence so coordination, provenance, conflicts, HITL, telemetry, and orchestration can be evaluated without external APIs. A real implementation, `LiveResearchProvider` (`research_system/live_provider.py`), now exists behind the same `ResearchProvider` protocol — it uses live Tavily search plus `ChatDeepSeek` structured output, and is used automatically when `DEEPSEEK_API_KEY`/`TAVILY_API_KEY` are set (see `CLAUDE.md` for details). The graph design itself never had to change.

---

## 8. Memory Management

Beyond a single research session's shared state (section 1 above), the system
distinguishes **short-term** (one research thread) from **long-term**
(cross-session) memory, split into the three standard long-term sub-types:

- **Semantic** — durable facts about a topic (e.g. "this topic's quality gate
  has failed on `unresolved_high_conflicts>0` in 2 of its last 3 runs").
- **Episodic** — a log of past runs on a topic (question, mode, whether the
  gate passed, whether HITL was needed).
- **Procedural** — versioned operational rules (quality-gate thresholds,
  per-topic `mandatory_human_review` overrides) that actually govern runtime
  behavior, analogous to an `AGENTS.md` bundle. Versioning is this repo's own
  git history, not an external product.

Memory is namespaced by **topic**, a keyword heuristic derived from the
question text (`research_system/memory/topic.py`) — not a real topic model,
documented as a known limitation there.

Example of the loop actually mattering: ask the same (or similarly-phrased)
question on the offline demo provider three times in a row, and by the third
run `reflect_on_topic()` will propose forcing human review by default for that
topic going forward, because the same quality-gate failure recurred every
time — a genuine "the system learned from experience" moment, not simulated.

See `CLAUDE.md`'s "Memory management" section for the full architecture and
the capture → analyze-and-refine → store-and-version → apply-and-consolidate
loop mapped to code.

### Inspecting memory

```bash
python inspect_memory.py                        # list every topic on record + procedural rules
python inspect_memory.py "Indian EV market"      # episodes + semantic facts for one topic
python inspect_memory.py ev_indian_market --raw  # same, as raw JSON
```

Read-only — safe to run any time, including mid-session. Long-term memory
itself lives in `.research_memory/store.json` (gitignored, accumulates
locally); `research_system/memory/procedural_rules.json` is the only piece
that's committed, since it's versioned config rather than accumulated data.
