# Assignment-to-Code Mapping

| Assignment / rubric requirement | Implementation |
|---|---|
| Shared Research Context | `ResearchState.evidence`, `question`, `execution_plan` |
| Agent Contributions | `ResearchState.contributions` |
| Cross-Agent Insights | `ResearchState.cross_agent_insights`, `build_cross_agent_insights()` |
| Collaborative Reasoning | `cross_agent_insights` + provenance lineage + downstream context (named explicitly in `README.md`'s "Collaborative Reasoning" section) |
| Synthesis Threads | `ResearchState.synthesis_threads` |
| Agent coordination | `graph.py` parallel, sequential, hybrid graph branches; see `README.md`'s "Agent Coordination Matrix" for the per-node depends-on/contributes/activation/signals breakdown |
| Inputs from other agents | Each specialist reads `state["evidence"]` |
| Agent signaling | Reducer-backed evidence/contribution lists |
| Dynamic activation | `planner.choose_execution_plan()` (parallel/sequential/hybrid pattern) + `planner.choose_active_agents()` (which of the four specialists actually run, per question) |
| Parallel vs sequential | Three explicit graph strategies |
| Additional investigation | HITL `more_research` routes back to planner |
| Synthesis stop criteria | `quality.quality_gate_node()` |
| Provenance | `Evidence` schema + parent IDs + source metadata |
| Conflict resolution | `detect_and_resolve_conflicts()` + first-class `Conflict` state |
| Human-in-the-loop | `human_review_node()` + LangGraph `interrupt()` |
| Checkpoints | `InMemorySaver` in demo; replace with durable saver in production |
| Short-term memory durability | `SqliteSaver` via `config.py`'s `_durable_checkpointer()` |
| Semantic long-term memory | `research_system/memory/semantic.py` |
| Episodic long-term memory | `research_system/memory/episodic.py` |
| Procedural long-term memory | `research_system/memory/procedural.py` + `procedural_rules.json`, versioned via this repo's git history |
| Telemetry | `timed_node()` + append-only telemetry state |
| Emergent intelligence | Insight requires evidence from >=2 specialist agents |
