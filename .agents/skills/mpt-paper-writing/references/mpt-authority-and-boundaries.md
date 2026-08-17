# MPT Authority And Boundaries

## Authority order

Before changing technical prose, inspect the narrowest relevant current files.
Use this order when sources disagree:

1. Current code, data paths, frozen manifests, and produced result artifacts.
2. `docs/research/METHOD_NOMENCLATURE.md`.
3. `docs/research/STAGE2_RESEARCH_PROTOCOL.md` and associated trial history.
4. `README.md` and current writing notes under `writing/`.
5. Legacy review summaries as historical guidance, never as current proof.

## Paper semantics

- Lead with fixed-window rough-terrain global planning. UAV sensing is an
  incidental experimental context, not the system's central identity.
- Distinguish fixed input/window dimensions from variable observed or
  traversable terrain support.
- Keep the boundary-aligned trajectory prior separate from masked terrain
  conditioning.
- Use general notation such as `n`, `d_y`, `K`, and `M` in Methods. Put concrete
  implementation counts in Experimental Setup.
- Define probability paths, velocity objectives, endpoint terms, and
  `J_PMF` before later stages reuse them. Keep curvature regularization
  distinct.
- Preserve a causal narrative: planning problem and information boundary,
  trajectory representation, conditioning, learning, then deployment.

## Stage boundary

- Treat the path representation and established Stage 1 interface separately
  from validation-gated Stage 2 mechanisms.
- Do not promote an experimental entry point, diagnostic, failed
  configuration, or unapproved result into the formal method.
- Leave undecided generator or training mechanisms explicitly pending.
- Keep diagnostic, validation, strict OOD, and final-test evidence labels
  separate. Preserve frozen manifests, denominators, seeds, and selection
  rules.
- Never inspect or use a protected final test merely to improve the paper.

## Deployment boundary

State the supported boundary positively: one planner replanning invocation,
with no full terrain map, online physical-cost evaluator, online optimizer,
candidate refinement, or candidate selection unless current code and the
frozen method explicitly establish otherwise.

Do not claim hard safety, mode preservation, route-distribution preservation,
runtime certification, or recovery of instance-specific unobserved terrain.

## Writing quality

Prefer dense publication prose with explicit subject and object attachment.
Avoid developer-documentation phrasing, undefined loss labels, generic slogans,
abstract colons, repetitive sentence openings, and word-count-only shortening.
Preserve equations, numerical results, limitations, and comparison boundaries
unless authoritative evidence supports changing them.
