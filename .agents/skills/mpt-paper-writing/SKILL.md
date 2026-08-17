---
name: mpt-paper-writing
description: Evidence-gated writing and revision workflow for the MPT rough-terrain global-planning paper. Use for MPT abstract, Method, experiments, related-work, claim, notation, or full-paper tasks in /home/yrf/MPT, especially when repository evidence, Stage 1 versus Stage 2 status, T-RL framing, or migration from the legacy academic-paper-writing skill matters. Routes general literature, audit, and full-paper operations to project-local ARIS skills while preserving MPT-specific boundaries.
---

# MPT Paper Writing

Use ARIS as the general research and paper engine while enforcing the MPT
project contract in `references/mpt-authority-and-boundaries.md`.

## Start

1. Read `references/mpt-authority-and-boundaries.md` completely.
2. Read `.aris/migration/legacy-state.yaml` and the live
   `.paper-review/memory/session.yaml` when they exist.
3. Treat `.paper-review/` as a read-only legacy audit trail. Write new ARIS
   state under `.aris/` and normal paper artifacts under their ARIS-owned paths.
4. Inspect current repository authorities before accepting a summary or editing
   prose. Never infer method maturity from the presence of an entry point.

## Route The Task

### Narrow writing or review

For one paragraph, section, equation, abstract, outline, or focused review:

- Edit only the requested writing artifact.
- Keep the old Mode C behavior: use the current source and repository evidence,
  then run one relevant fresh review after a content-changing edit.
- Give a fresh reviewer only the current artifact, authoritative source paths,
  venue metadata, and task scope. Do not expose prior raw review reports or tell
  it which answer to confirm.
- Use ARIS `research-lit` and `novelty-check` when the task changes novelty,
  related work, or positioning. Mark unavailable source comparisons unverified.
- Do not invoke the full `paper-writing` pipeline for a narrow task.

### Evidence and claim gates

Use the corresponding project-local ARIS skill:

- `experiment-audit` before treating an experiment as trustworthy evidence.
- `result-to-claim` after trustworthy results exist and before drafting claims.
- `paper-claim-audit` when a complete paper reports numerical results.
- `citation-audit` after the bibliography and citation contexts stabilize.

Preserve ARIS verdict semantics. A same-family semantic review is provisional;
deterministic checks may be accepted only for what they actually verify.

### Full paper workflow

Use ARIS `paper-writing` only when editable full-paper source or a sufficiently
complete `NARRATIVE_REPORT.md` exists. Pass these MPT defaults explicitly:

```text
venue: IEEE_JOURNAL
assurance: submission
AUTO_PROCEED: false
human checkpoint: true
illustration: false
```

Do not silently default to ICLR. Pause when a claim lacks evidence, a manual
figure is missing, a contract assertion is contested, or source is incomplete.

### Research and experiments

Do not invoke `research-pipeline`, `experiment-bridge`, `run-experiment`,
`experiment-queue`, remote execution, or paid compute merely because a writing
review requests more evidence. First produce a scoped experiment proposal with
the frozen split, controls, denominators, selection rule, stop rule, expected
cost, and artifact path. Launch only after explicit user approval.

## Finish

- Report which claims are established, provisional, diagnostic, pending, or
  unsupported.
- State which files and evidence support each material revision.
- Keep unresolved components explicit rather than filling them with plausible
  architecture or results.
- Update ARIS audit artifacts for the invoked workflow, but never rewrite the
  legacy `.paper-review/audit/` history.

For legacy-to-ARIS artifact ownership and recovery, read
`references/legacy-state-map.md`.
