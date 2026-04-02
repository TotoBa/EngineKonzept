# Proposer Architecture

Phase 5 introduces the first learned runtime component: a legality/policy proposer trained in Python and exported in a Rust-loadable bundle.

## Scope

This phase adds only:

- a PyTorch legality/policy proposer
- a deterministic training config and training loop
- held-out legal-set precision/recall reporting
- a `torch.export` bundle with JSON metadata
- Rust-side bundle loading and schema validation

It still does **not** add:

- latent dynamics
- opponent modeling
- planner rollouts
- classical search or handcrafted evaluation

## Model Inputs

The proposer consumes the existing deterministic Phase-3 encoding after a fixed-width packing step:

- `piece_tokens`: padded from `N x 3` to `32 x 3` with `[-1, -1, -1]`
- `square_tokens`: flattened `64 x 2`
- `rule_token`: flattened width `6`

This yields a flat input vector of size `230`.

The packing order is exported in the bundle metadata so Rust can reconstruct the same view later.

The `multistream_v2` arm is the first repo-local application of the broader architecture ideas captured in [arch.ideas.md](/home/torsten/EngineKonzept/docs/arch.ideas.md): preserve typed structure longer, fuse streams explicitly, and delay any more aggressive router/expert designs until the base representation itself is stronger.

## Model Shape

Phase 5 now carries two proposer architectures behind the same dataset and export contract.

### `mlp_v1`

The original proposer is intentionally simple: a shared MLP backbone plus two flat heads.

For the currently used `hidden_dim=128`, `hidden_layers=2`, `dropout=0.1` configuration:

- backbone: `Linear(230 -> 128)`, `ReLU`, `Dropout`, `Linear(128 -> 128)`, `ReLU`, `Dropout`
- legality head: `Linear(128 -> 20480)`
- policy head: `Linear(128 -> 20480)`

That configuration has `5,329,920` trainable parameters. Most of the size sits in the two `128 x 20480` output heads, not in the backbone.

### `multistream_v2`

The new architecture keeps the packed `230`-wide input contract but unpacks it back into three typed streams inside the model:

- piece stream: `32 x 3`
- square stream: `64 x 2`
- rule stream: `6`

It then applies:

- separate projections for piece, square, and rule streams
- one piece-to-square cross-attention block
- one square-to-piece cross-attention block
- masked pooling for pieces, dense pooling for squares, and rule-conditioned fusion
- small head-specific towers before the flat legality and policy heads

This keeps the current Phase-5 action-space contract intact while restoring some of the object-centric structure that was previously lost in the flat packing step.

The choice of this direction is deliberate. For this repository state, a typed multi-stream encoder is a lower-risk fit than early mixture-of-experts routing because:

- the Phase-3 encoder already exposes object-centric piece/square/rule structure
- the current Phase-5 data budget is still modest
- the main weakness is likely representational bias, not yet routing capacity
- the runtime/export contract must remain stable while Phase 5 is still offline-training only

This follows the same broad design pressure emphasized by work on permutation-aware set models and relational inductive biases, but keeps the actual Phase-5 implementation small enough to test and compare directly. The broader prioritization is captured in [model-roadmap.md](/home/torsten/EngineKonzept/docs/architecture/model-roadmap.md).

## Current Decision

For this repository state, the next preferred proposer direction is a factorized decoder over the existing move schema, not early mixture-of-experts routing.

Why:

- the move space is already factorized in the action-space layer
- the current flat `20480` heads dominate parameter count
- current results suggest policy needs better structure more than it needs heavier routing
- factorization preserves the exact same Rust legality authority and export boundary

## Model Outputs

The model predicts two flat tensors over the joint action vocabulary:

- legality logits over `64 * 64 * 5 = 20480` actions
- policy logits over the same `20480` actions

The flat action index matches the Phase-3 factorization:

`((from_index * 64) + to_index) * 5 + promotion_index`

The output contract remains identical across `mlp_v1` and `multistream_v2`, so existing datasets, metrics, export tooling, and Rust-side metadata validation remain compatible.

## Training Objective

Training combines:

- legality BCE-with-logits against the exact legal move set
- policy cross-entropy against `selected_action_encoding` when available

Held-out reporting currently includes:

- total loss
- legality loss
- policy loss
- legal-set precision
- legal-set recall
- legal-set F1
- policy top-1 accuracy on positions with selected-action labels
- training examples/second so CPU throughput changes remain measurable

## Training Runtime

The training config now also exposes a small runtime section for CPU-bound runs:

- `runtime.torch_threads`: optional override for `torch.set_num_threads`
- `runtime.dataloader_workers`: worker count for the PyTorch `DataLoader`

The trainer pre-packs dense legality targets once per split before epoch iteration. That keeps Phase 5 simple while avoiding repeated `20480`-wide target construction in every batch.

## Export Bundle

The proposer export bundle currently contains:

- `checkpoint.pt`: PyTorch checkpoint with model weights and training config
- `proposer.pt2`: `torch.export` program for later runtime integration
- `metadata.json`: fixed metadata filename containing schema, input layout, action-space sizes, threshold, and validation metrics

## Rust Boundary

The `inference` crate does not execute proposer inference yet.

In Phase 5 it is responsible for:

- loading `metadata.json`
- validating that the schema and dimensions are self-consistent
- verifying that the referenced exported-program and checkpoint files exist

That keeps the export contract explicit before later runtime integration work.
