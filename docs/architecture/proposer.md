# Proposer Architecture

Phase 5 introduces the first learned runtime component: a proposer trained in Python and exported in a Rust-loadable bundle.

## Scope

This phase adds only:

- a PyTorch proposer
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

Phase 5 now carries eight proposer architectures. Seven remain learned-legality baselines on the old flat-action contract; `symbolic_v1` is now the official exported proposer path.

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

### `factorized_v3`

The first factorized decoder arm keeps the same MLP-style backbone, but replaces the flat `20480` heads with smaller `from`, `to`, and `promotion` heads:

- `from`: `64`
- `to`: `64`
- `promotion`: `5`

The final flat action logits are reconstructed by summing the three component logits back into the canonical joint action space.

This architecture was intentionally tested as the smallest possible repo-local decoder factorization that preserves:

- the current dataset schema
- the current losses
- the current export contract
- the current Rust-side metadata validation

Measured outcome on the `10k` Pi-labeled corpus:

- parameter count dropped to `113,418`
- throughput remained good
- but both legality and policy quality collapsed

That makes `factorized_v3` a useful negative result, not a new default.

### `factorized_v4`

The next decoder arm restores coupling between move components without returning to a full flat output head.

It keeps:

- a shared MLP backbone
- small `from`, `to`, and `promotion` component heads
- the same flat `20480` output contract externally

But unlike `factorized_v3`, it conditions later move parts on earlier ones:

- `to` depends on state plus `from`-conditioned queries
- `promotion` depends on state plus both `from`- and `to`-conditioned queries

Measured outcome on the `10k` Pi-labeled corpus:

- parameter count: `4,359,434`
- validation `legal_set_f1`: `0.082032`
- verify `legal_set_f1`: `0.095668`
- verify `policy_top1_accuracy`: `0.010742`

This makes `factorized_v4` the current best legality arm, but not the best policy arm.

### `factorized_v5`

The next decoder arm keeps the strong legality structure from `factorized_v4`, but gives the policy side extra flat residual capacity through a small low-rank correction on top of the conditional factorized policy head.

Measured outcome on the `10k` Pi-labeled corpus:

- parameter count: `4,709,658`
- validation `legal_set_f1`: `0.055895`
- validation `policy_top1_accuracy`: `0.014648`
- verify `legal_set_f1`: `0.06438`
- verify `policy_top1_accuracy`: `0.014648`

That makes `factorized_v5` a better legality/policy balance than `factorized_v4`, but it still does not beat the current default on policy accuracy and no longer holds the best legal-F1 spot.

### `factorized_v6`

This arm keeps the legality side from `factorized_v4`, but strengthens the policy decoder again without returning to a full flat head.

The policy side now combines:

- the conditional factorized policy decoder from `factorized_v5`
- an explicit learned `from-to` interaction term
- a low-rank flat residual

Measured outcome on the `10k` Pi-labeled corpus:

- parameter count: `4,923,939`
- validation `legal_set_f1`: `0.10281`
- verify `legal_set_f1`: `0.123078`
- verify `policy_top1_accuracy`: `0.010742`

This makes `factorized_v6` the current strongest legality arm by a clear margin, but it still trails `current_default` on policy.

### `relational_v1`

This arm combines the typed multi-stream backbone direction from `multistream_v2` with the stronger factorized heads from the newer decoder line.

It keeps:

- piece/square/rule stream separation
- light cross-attention between piece and square streams
- factorized legality decoding
- the stronger policy-side pairwise coupling used in `factorized_v6`

Measured outcome on the `10k` Pi-labeled corpus:

- parameter count: `5,193,891`
- validation `legal_set_f1`: `0.065213`
- validation `policy_top1_accuracy`: `0.012695`
- verify `legal_set_f1`: `0.074294`
- verify `policy_top1_accuracy`: `0.01416`

That does not beat `current_default` on policy and does not beat `factorized_v6` on legality, but it does improve policy over the earlier factorized arms while keeping clearly stronger legality than the old flat MLP baseline.

### `symbolic_v1`

This arm replaces the learned legality head with exact symbolic legal-move generation and trains only a scorer over legal candidates.

It keeps:

- the same packed `230`-feature state input
- the same exact Rust legality authority
- the same action-space indices for supervision and evaluation

But it adds a separate symbolic side input per legal candidate:

- exact legal candidate list
- compact per-move flags such as capture, promotion, castle, en passant, and gives-check
- compact attack-context features derived from exact attacked-square maps
- small global tactical flags such as `in_check` and legal-move count

The current implementation is now the official proposer path:

- legality is no longer learned
- Rust/runtime export now carries an explicit symbolic side-input contract
- the trained artifact is exported as a Rust-loadable bundle

Measured outcome on the `10k` Pi-labeled corpus:

- validation `legal_set_f1`: `1.0`
- validation `policy_top1_accuracy`: `0.158203`
- verify `legal_set_f1`: `1.0`
- verify `policy_top1_accuracy`: `0.127441`

That is the strongest `10k` proposer result in the repository so far, and it now defines the current proposer direction in the repository.

## Current Decision

For this repository state, the repository now prefers exact symbolic legality plus learned candidate scoring over further learned-legality expansion.

Why:

- the move space is already factorized in the action-space layer
- the exact Rust legality authority is already stronger than any learned-legality arm on the current corpus
- current results suggest candidate scoring is now the main learned problem
- symbolic candidate scoring preserves the exact same action indices and legality authority while dropping wasted probability mass on illegal actions

The new results narrow that further:

- `factorized_v3` showed that purely additive factorization is too weak
- `factorized_v4` showed that conditional factorization can recover and substantially improve legality
- `factorized_v5` showed that extra policy-specific capacity can recover much of the lost policy signal without falling back to a full flat head
- `factorized_v6` showed that explicit policy-side `from-to` coupling can push legality further still, but not enough to win policy
- `relational_v1` showed that the typed backbone remains useful when paired with the stronger newer decoder heads
- `symbolic_v1` showed that exact symbolic legality plus candidate scoring is dramatically stronger on the current `10k` corpus than any learned-legality arm, and it now carries the official export/runtime contract

So the next proposer question is no longer "how should legality be learned?", but how to improve candidate scoring, symbolic move/context features, and downstream use of the symbolic candidate set.

## Model Outputs

The model predicts two flat tensors over the joint action vocabulary:

- legality logits over `64 * 64 * 5 = 20480` actions
- policy logits over the same `20480` actions

The flat action index matches the Phase-3 factorization:

`((from_index * 64) + to_index) * 5 + promotion_index`

The output contract remains identical across the learned-legality baselines `mlp_v1`, `multistream_v2`, `factorized_v3`, `factorized_v4`, `factorized_v5`, `factorized_v6`, and `relational_v1`, so existing datasets, metrics, export tooling, and Rust-side metadata validation remain compatible.

`symbolic_v1` preserves the same flat action indices for supervision and evaluation, but it adds a symbolic legal-candidate side input in the exported metadata and Rust-side runtime contract.

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

Checkpoint selection is now an explicit config choice as well:

- `legality_first`
- `policy_first`
- `balanced`

The `balanced` mode uses a weighted score over legality and policy metrics instead of always selecting by legal-set F1 first.

## Training Runtime

The training config now also exposes a small runtime section for CPU-bound runs:

- `runtime.torch_threads`: optional override for `torch.set_num_threads`
- `runtime.dataloader_workers`: worker count for the PyTorch `DataLoader`

The trainer pre-packs dense legality targets once per split before epoch iteration. That keeps Phase 5 simple while avoiding repeated `20480`-wide target construction in every batch.

The trainer now also supports explicit checkpoint-selection policy via the evaluation config. This matters for the factorized line because legality and policy can peak at different epochs.

## Export Bundle

The proposer export bundle currently contains:

- `checkpoint.pt`: PyTorch checkpoint with model weights and training config
- `proposer.pt2`: `torch.export` program for later runtime integration
- `metadata.json`: fixed metadata filename containing schema, input layout, action-space sizes, legality source, optional symbolic-input spec, and validation metrics

## Rust Boundary

The `inference` crate does not execute proposer inference yet, but it now also defines the official symbolic proposer input contract.

In Phase 5 it is responsible for:

- loading `metadata.json`
- validating that the schema and dimensions are self-consistent
- verifying that the referenced exported-program and checkpoint files exist
- building exact legal candidates plus symbolic per-move/global features for the symbolic proposer path

That keeps the export contract explicit before later runtime integration work.
