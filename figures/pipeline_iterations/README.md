# Pipeline figure iterations

Each numbered SVG/PNG is retained as a separate visual iteration. The current
candidate is `pipeline_v55.svg`.

## v55 structure

- The upper band shows the offline provenance of the deployed model:
  expert route demonstrations -> Stage 1 route learning -> Stage 2 physical
  adaptation -> adapted generator.
- The lower band shows the deployment path:
  masked terrain and task poses -> adapted generator -> boundary-aligned path
  representation and analytic B-spline decoder -> generated global path.
- Full terrain is shown only as a training-only annotation attached to the
  rollover-stability objective. It is not a deployment input.
- The boundary-aligned path representation is shown in the deployment spine;
  its small spline icon in Stage 1 indicates that route learning uses the same
  aligned path space without adding another pipeline block.
- `pipeline_v55_1200.png` and `pipeline_v55_800.png` are reduced-size review
  renders. The SVG embeds its terrain thumbnails as data URIs.
