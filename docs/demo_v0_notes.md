# Demo v0 Notes

## Objective

- Build a simulation replay prototype for retrieval-conditioned progress estimation.
- Show that retrieved context gives a better progress proxy than no-context and random-context baselines.

## Data and Backend

- Backend used: `replay2d`.
- Task instruction: `Pick the blue object and place it inside the green goal zone.`.
- Support memory size: `48` frames.
- Query rollout size: `24` frames.
- Retrieval setting: `top-k=3` with cosine similarity over frozen image embeddings.

## Pipeline

- Build a mini memory bank from support frames with task instruction, stage text, and pseudo progress labels.
- Retrieve top-k support entries for each query frame.
- Predict progress with a retrieval-weighted average of retrieved progress labels.
- Compare three modes: `no context`, `random context`, and `retrieved context`.

## Current Metrics

- MAE no context: `0.1296`.
- MAE random context: `0.2868`.
- MAE retrieved context: `0.0797`.
- Correlation retrieved vs oracle: `0.9735`.

## Limitations

- This sprint uses a lightweight progress proxy, not a learned reward head.
- The demo is single-task and stage labels are pseudo labels from stage windows / timestep progress.
- No IQL training, broad benchmark sweep, or OOD evaluation is included yet.

## Next Step

- Replace the retrieval-weighted proxy with a proper retrieval-conditioned reward model, then connect that reward to offline RL with IQL.
