# Figure Prompt v1: Method Overview

Use case: scientific-educational
Asset type: publication-quality academic method overview figure for a robotics/path-planning paper

## Prompt

Create a clean, wide, two-band method overview diagram for a rough-terrain global path planner. Use a white background, flat vector-like academic illustration, restrained blue/teal/orange/red palette, dark gray arrows, sans-serif typography, and no decorative scenery. The figure must be readable when reduced to a two-column IEEE paper width.

The figure has exactly two causal bands.

TOP BAND - OFFLINE TRAINING:
1. Expert route demonstrations: several smooth ground-vehicle routes with fixed start and goal poses.
2. Stage 1, Expert Route Prior Learning (ERPL): the generator learns long-range route structure from expert routes. Show the same observed terrain condition and task poses as the deployment condition. Do not show full terrain here.
3. Stage 2, Privileged Multi-objective Terrain Adaptation (PMTA): initialize from the Stage-1 generator and adapt the generated paths with three physically meaningful objectives: planning-support violation, rollover stability, and curvature violation. Show MGDA/gradient coordination as a small coordination node below the three objectives. Draw a dashed side input labelled “complete terrain - training only” entering only the rollover-stability objective. This input must visibly stop inside the training band and must not connect to the deployment band.
4. Output a single “adapted generator” block. This is the model used at deployment; do not draw a separate teacher or student model.

BOTTOM BAND - ONLINE PATH GENERATION:
1. Input thumbnail: a partial masked terrain observation in a fixed-size planning window, with an irregular mapped/admissible subset, unknown or forbidden cells outside it, and start/goal poses inside the admissible subset. Label the input exactly: “masked terrain observation + task poses”. Add one small dot or Gaussian blob labelled “source”; this is a standard Gaussian source for one endpoint-conditioned forward pass, not a Brownian bridge and not a diffusion sampling chain.
2. Arrow into the adapted generator. Label the arrow or nearby note: “one forward pass”. Do not show iterative optimization, cost evaluation, best-of-K selection, or online refinement.
3. Arrow into an orange module labelled “boundary-aligned cubic B-spline representation”. Inside this module show: task-coordinate canonicalization; endpoint position and heading analytically fixed by the decoder; network-predicted free interior control-point coordinates; a continuous differentiable spline curve. Do not call the source distribution Brownian bridge. Do not imply that the network predicts the full unconstrained path.
4. Output thumbnail: the generated global path overlaid on the same partial masked terrain observation. Show the path beginning at the prescribed start pose and ending at the prescribed goal pose, with heading arrows aligned at both ends. The path may be physically imperfect because PMTA improves feasibility but does not provide a runtime hard safety certificate.

Causal arrows must read left-to-right within each band. Keep the two bands visually separate. The top band explains how the generator is trained; the bottom band explains what is available and what happens during deployment. The mask is an input condition and information boundary, not a third contribution. The two core contributions are (i) the boundary-aligned trajectory representation and (ii) two-stage ERPL-to-PMTA learning.

## Exact labels to preserve

OFFLINE TRAINING
Expert route demonstrations
Stage 1: Expert Route Prior Learning (ERPL)
Stage 2: Privileged Multi-objective Terrain Adaptation (PMTA)
planning-support violation
rollover stability
curvature violation
MGDA gradient coordination
complete terrain - training only
adapted generator

ONLINE PATH GENERATION
masked terrain observation + task poses
standard Gaussian source
one forward pass
boundary-aligned cubic B-spline representation
fixed endpoint position and heading
predicted free interior control points
generated global path

## Negative constraints

Do not add a UAV, drone, rover, wolf, raven, Mars scene, or biological metaphor to this technical figure. Do not add a third training stage. Do not connect complete terrain to deployment. Do not use a giant feedback loop. Do not depict Brownian-bridge sampling, multi-step diffusion denoising, online ESDF/cost optimization, or hard runtime safety certification. Do not invent encoder, decoder, attention, or backbone submodules that are not specified. Avoid tiny text, rainbow colors, 3D perspective, glossy effects, heavy shadows, and stock illustrations.
