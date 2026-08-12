# Assignment-to-Code Mapping

| Assignment / rubric requirement | Implementation |
|---|---|
| Shared Research Context | `ResearchState.evidence`, `question`, `execution_plan` |
| Agent Contributions | `ResearchState.contributions` |
| Cross-Agent Insights | `ResearchState.cross_agent_insights`, `build_cross_agent_insights()` |
| Collaborative Reasoning | `cross_agent_insights` + provenance lineage + downstream context |
| Synthesis Threads | `ResearchState.synthesis_threads` |
| Agent coordination | `graph.py` parallel, sequential, hybrid graph branches |
| Inputs from other agents | Each specialist reads `state["evidence"]` |
| Agent signaling | Reducer-backed evidence/contribution lists |
| Dynamic activation | `planner.choose_execution_plan()` |
| Parallel vs sequential | Three explicit graph strategies |
| Additional investigation | HITL `more_research` routes back to planner |
| Synthesis stop criteria | `quality.quality_gate_node()` |
| Provenance | `Evidence` schema + parent IDs + source metadata |
| Conflict resolution | `detect_and_resolve_conflicts()` + first-class `Conflict` state |
| Human-in-the-loop | `human_review_node()` + LangGraph `interrupt()` |
| Checkpoints | `InMemorySaver` in demo; replace with durable saver in production |
| Telemetry | `timed_node()` + append-only telemetry state |
| Emergent intelligence | Insight requires evidence from >=2 specialist agents |
