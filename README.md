# Multi-Agent Research System (LangGraph)

A runnable reference implementation for the Coursera **Designing a Multi-Agent Research System** assignment.

The design intentionally demonstrates four things the rubric cares about:

1. **Collaborative shared state** — evidence, provenance, agent contributions, cross-agent insights, conflicts, reasoning threads, quality gates, human decisions, and telemetry all live in one state.
2. **Adaptive coordination** — the orchestrator chooses **parallel**, **sequential**, or **hybrid** execution based on the research question.
3. **Rigorous synthesis** — provenance, conflict resolution, confidence/coverage thresholds, and explicit stop criteria are enforced before synthesis.
4. **Human-in-the-loop (HITL)** — unresolved conflicts, low confidence, low evidence coverage, or a human-requested review pause the graph using LangGraph `interrupt()`.

## Architecture

```text
                         ┌──────────────────────────────┐
User question ──────────>│ Planner / Orchestrator      │
                         │ chooses execution strategy   │
                         └──────────────┬───────────────┘
                                        │
              ┌─────────────────────────┼─────────────────────────┐
              │                         │                         │
              ▼                         ▼                         ▼
       PARALLEL SPEED            SEQUENTIAL DEPTH           HYBRID DEFAULT
    ┌─────────────────┐       Web -> Data -> Trend       Web ──┐
    │ Web   Data      │       -> Competitive                  ├─> Trend ──┐
    │ Trend Competitive│                                     │           ├─>
    └───────┬─────────┘                                  Data ─┘  Comp ───┘
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
