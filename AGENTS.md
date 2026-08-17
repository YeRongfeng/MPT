<!-- ARIS-CODEX:BEGIN -->
## ARIS Codex Skill Scope
ARIS Codex packages installed in this project: skills-codex
Managed entries: 83
Manifest: `.aris/installed-skills-codex.txt`
ARIS repo root: `/home/yrf/.codex/vendor/aris-e12e07c`
Project skill path: `.agents/skills/<skill-name>`
For ARIS Codex workflows, prefer the project-local skills under `.agents/skills/`.
When a skill needs ARIS helper scripts, resolve the repo root from the manifest or set it explicitly:
`ARIS_REPO=$(awk -F'\t' '$1=="repo_root"{print $2; exit}' "/home/yrf/MPT/.aris/installed-skills-codex.txt")`
Do not edit or delete symlinked skills in place; update upstream or rerun:
`bash /home/yrf/.codex/vendor/aris-e12e07c/tools/install_aris_codex.sh "/home/yrf/MPT" --reconcile`
For copied Codex installs, use:
`bash /home/yrf/.codex/vendor/aris-e12e07c/tools/smart_update_codex.sh --project "/home/yrf/MPT"`
<!-- ARIS-CODEX:END -->

## MPT ARIS Policy

- Use `.agents/skills/mpt-paper-writing/` as the default entry point for MPT
  abstract, Method, experiments, related-work, notation, claim, and full-paper
  tasks. It preserves the project-specific contract while routing general work
  to ARIS.
- Treat the installed ARIS source as pinned to commit
  `e12e07c7b85ee1a4dc07e5463089aa16836af2bf`. Do not run `git pull`, change the
  manifest repo root, or reconcile against another revision without an explicit
  update audit.
- Override autonomous ARIS defaults for this project: `AUTO_PROCEED=false`,
  `HUMAN_CHECKPOINT=true`, and `AUTO_WRITE=false` unless the user explicitly
  requests a broader unattended workflow.
- Do not launch `experiment-bridge`, `run-experiment`, `experiment-queue`, SSH,
  remote jobs, paid compute, or a new local training run without explicit user
  approval of the experiment scope and resource use.
- Preserve Stage 1 versus validation-gated Stage 2, frozen manifests, controls,
  denominators, selection rules, and diagnostic/validation/strict-OOD/final
  evidence labels. Never access a protected final test to improve writing or
  choose a configuration.
- Verify technical claims against current code, produced artifacts,
  `docs/research/METHOD_NOMENCLATURE.md`, and
  `docs/research/STAGE2_RESEARCH_PROTOCOL.md` before revising prose.
- Treat `.paper-review/` as an immutable legacy audit trail. New ARIS state and
  traces belong under `.aris/`; never renumber or overwrite legacy review rounds.

=== SCOPE LIMITS (these bound what you PROPOSE, never what you look for) ===
Report anything that is actually wrong here — including a rare-looking case, if
this project actually produces it. Then keep the fix in scope:
1. This is not a security paper. Verification is welcome; over-defense is not.
   Unless this project states otherwise, assume a cooperating operator on their
   own machine; if it has a real adversary, it will say so and that scope wins.
2. Do not add hashes, checksums or fingerprints unless the hash replaces a
   materially more expensive operation AND its result changes what happens next.
3. No defensive scaffolding: no feature flags, migration frameworks, compat
   layers or wrappers for cases that do not occur here.
4. No corner-case obsession: exotic encodings, symlink races, RTL text and
   millisecond races are out of scope unless the case is reachable through this
   project's supported use — its documented inputs, its published interface,
   its real data. Reachable is enough; you do not need a reproduction.
   Constructible in principle is not enough.
5. Where judgement is needed, judge. Do not replace it with a scoring table, a
   checklist, or a re-verification loop over something already settled.
6. None of this overrides security, migration, verification or review that the
   user, this project's own conventions, or a higher-priority rule asked for.
   Those were requested; they are the work, not scope creep.
Shapes already seen, for calibration. Examples, not a checklist — a real finding
is not dismissed by resembling one:
  H  hashing every row of two spreadsheets to answer what comparing cells answers
  H  writing checksum files that nothing ever reads
  E  hardening the accounts of an app that has no users and no deployment
  R  auditing your own patch all night while the feature stays unwritten
  R  a reviewer that returns a failing verdict on everything
  O  guards whose justification is the previous guard, not the requirement
And two that look like the above and are not. Report these:
  ✓  a digest that lets you skip re-reading a large file you already have
  ✓  a rare-looking input this project's own documentation example produces
Before running any check, answer: what specific failure would this detect, and
what would I do differently if it occurred? No answer means do not run it.
Say plainly when something is correct. Do not manufacture findings.
