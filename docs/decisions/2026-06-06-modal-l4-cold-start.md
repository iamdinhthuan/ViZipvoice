# Decision: Modal L4 Cold-Start Deployment

## Status

Accepted

## Context

The TTS service should run on Modal without holding an idle GPU for long periods. The user asked for NVIDIA L4 and cold-start behavior to reduce cost.

## Decision

The Modal worker function uses `gpu="L4"`, `min_containers=0`, `buffer_containers=0`, `max_containers=1` by default, and `scaledown_window=5`. The web ASGI function uses a separate lightweight FastAPI image with no GPU. The GPU model loads lazily inside the worker and is cached only while the worker container remains alive.

## Consequences

The first job after scale-to-zero has higher latency because Modal must start the worker and load the model. Idle GPU spend is minimized because no warm GPU container is kept by default. Throughput defaults to one worker container to avoid accidental parallel GPU spend; it can be raised with `VIZIPVOICE_MODAL_MAX_CONTAINERS`.

