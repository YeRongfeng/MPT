# Legacy State Map

## Ownership

| Legacy artifact | Status after migration | ARIS successor |
|---|---|---|
| Global `academic-paper-writing` skill | Archived outside discovery path | `mpt-paper-writing` plus ARIS skills |
| `.paper-review/memory/session.yaml` | Read-only venue and project context | MPT adapter context |
| `.paper-review/memory/review/current_state.yaml` | Read-only round 27 recovery point | `.aris/` workflow state for new runs |
| `.paper-review/audit/reviews/` | Immutable historical review archive | `.aris/traces/` and ARIS audit artifacts |
| `.paper-review/audit/revision-log.md` | Immutable historical revision record | ARIS workflow-specific logs |

## Recovery

The complete legacy skill and runtime snapshots live under
`.aris/migration/`. Verify their canonical tar hashes against
`.aris/migration/legacy-state.yaml` before restoring.

Restoring means copying the archived skill back into the Codex global skill
directory and continuing from the live `.paper-review` state. Never merge ARIS
trace files into legacy review rounds or renumber the old history.
