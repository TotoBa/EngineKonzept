# LAPv2 Codex-CLI Implementierungsplan (Rev 3)

> Begleitplan zum Paper **LAPv2 – Phasen-MoE und NNUE-artige Heads**.
> Zielrepo: `https://github.com/TotoBa/EngineKonzept.git` (Branch `lapv2`).
> Zielwerkzeug: **Codex-CLI**. Jeder Schritt ist eine in sich abgeschlossene,
> rückfallsichere Task, die als einzelner Codex-Auftrag ausführbar ist.

> **Was ist neu in Rev 3?** Policy ist jetzt ebenfalls ein NNUE-artiger
> Head, der seinen Feature-Transformer mit dem Value-Head teilt. Daraus
> ergibt sich ein neuer Datenpfad (Dual-Accumulator + Move-Delta), der in
> Schritt 3 als gemeinsame Infrastruktur eingeführt wird, bevor in
> Schritt 5 und 6 die beiden Heads daraufgesetzt werden.

---

## 0. Vorbemerkungen für Codex

**Lies das zuerst, bevor du irgendetwas tust.** Dieser Plan nennt Modul-
und Dateinamen auf Basis der LAPv1-Architekturübersicht (Intention Encoder,
State Embedder, Policy Head, Value Head, Sharpness Head, Opponent Head,
Deliberation Loop, T1/T2-Training, LAPv1-Artefakt). Die tatsächlichen
Pfade im Repo können abweichen. **Bevor du die erste Änderung schreibst,
erstelle per `rg`/`find` ein Mapping von logischem Namen zu echtem Pfad
und halte es in `docs/lapv2/path_map.md` fest.** Alle weiteren Schritte
referenzieren die logischen Namen.

**Harte Regeln für jeden Schritt:**

1. Arbeite auf einem eigenen Feature-Branch `lapv2/step-NN-<slug>`.
2. Keine nicht zum Schritt gehörenden Änderungen. Kein Drive-By-Refactoring.
3. Jeder Schritt endet mit: Tests grün, Lint grün, kurzer Changelog-Eintrag
   unter `docs/lapv2/CHANGELOG.md`.
4. Jeder Schritt hat ein explizites **Acceptance Criterion** und ein
   **Rollback** (wie man den Schritt wieder ausbaut). Beides MUSS erfüllt
   sein.
5. Wenn ein Schritt eine existierende Datei verändert, zuerst einen
   Golden-Test aufnehmen, der das bisherige Verhalten festhält.
6. LAPv1-Checkpoint-Kompatibilität ist ein **non-negotiable**: jeder
   Schritt bis einschließlich Schritt 9 muss einen Modus haben, in dem das
   Modell numerisch wie LAPv1 rechnet.

**Globaler Konfigurationsschalter:** Lege in der zentralen Config einen
Toplevel-Key `lapv2` mit Unterschaltern an:

```yaml
lapv2:
  enabled: false
  phase_moe: false           # Schritt 5
  dual_accumulator: false    # Schritt 4 (Datenpfad)
  nnue_value: false          # Schritt 6
  nnue_value_phase_moe: false  # Schritt 7
  nnue_policy: false         # Schritt 8
  sharpness_phase_moe: false # Schritt 9
  shared_opponent_readout: false  # Schritt 10
  distill_opponent: false    # Schritt 11
  accumulator_cache: false   # Schritt 12
  N_accumulator: 64
```

Solange alle Schalter `false` sind (außer N_accumulator), ist LAPv2
funktional inaktiv und das Repo verhält sich wie LAPv1. Jeder Schritt
schaltet genau einen Flag ein.

---

## Schrittübersicht

| # | Titel | Flag | Risiko |
|---|---|---|---|
| 1 | Repo-Audit und Path-Map | – | nil |
| 2 | Artefakt-Erweiterung (phase, king_sq, NNUE-Features, Move-Deltas) | – | niedrig |
| 3 | Phase-Router + generisches Phase-MoE-Modul | – | niedrig |
| 4 | Dual-Accumulator-Infrastruktur (FT, Builder, inkrementelles Update) | `dual_accumulator` | mittel |
| 5 | Phase-MoE in Intention Encoder + State Embedder | `phase_moe` | mittel |
| 6 | NNUE-Value-Head auf geteiltem FT (single phase) | `nnue_value` | mittel |
| 7 | NNUE-Value-Head als Phase-MoE | `nnue_value_phase_moe` | mittel |
| 8 | NNUE-Policy-Head (Successor-Scoring, geteilter FT) | `nnue_policy` | hoch |
| 9 | Sharpness-Head als Phase-MoE | `sharpness_phase_moe` | niedrig |
| 10 | Shared-Backbone-Opponent-Readout mit Δ-Operator | `shared_opponent_readout` | hoch |
| 11 | Optionaler Distillations-Loss | `distill_opponent` | niedrig |
| 12 | Deliberation-Loop: AccumulatorCache + Phase-Fixierung | `accumulator_cache` | mittel |
| 13 | T1/T2 Pipeline: Warm-Start, Loss-Balance, Phase-Gate | – | hoch |
| 14 | Arena-Harness, Reporting, Release-Gate | – | niedrig |

---

## Schritt 1 — Repo-Audit und Path-Map

**Ziel:** Codex kennt die echten Dateipfade und hat ein stabiles logisches
Vokabular für alle folgenden Schritte.

**Aufgaben:**

- Per `rg`/`find` folgende Symbole lokalisieren und in
  `docs/lapv2/path_map.md` als Markdown-Tabelle ablegen:
  - `IntentionEncoder` (oder äquivalent)
  - `StateEmbedder` / `StateEncoder`
  - `PolicyHead`
  - `ValueHead`
  - `SharpnessHead`
  - `OpponentHead`
  - `DeliberationLoop` / `inner_step` / Wrapper-Klasse
  - LAPv1-Artefakt-Loader (jsonl parser, tensor builder)
  - Stage-T1-Trainer
  - Stage-T2-Trainer
  - Arena-Runner (Thor-150, inner0/1/2/auto4)
  - Move-Generator / Move-Encoder (für Schritt 2 nötig)

- Die Tabelle enthält je: logischer Name, Datei, Klasse/Funktion,
  Kurzbeschreibung.
- Keine Codeänderungen in diesem Schritt.

**Acceptance:** `docs/lapv2/path_map.md` existiert, enthält alle 12
Einträge, CI grün (nur Doku-Änderung).

**Rollback:** Datei löschen.

---

## Schritt 2 — Artefakt-Erweiterung

**Ziel:** LAPv1-Artefakt um die für LAPv2 nötigen deterministischen Felder
ergänzen, ohne LAPv1 zu brechen.

**Neue Felder pro Sample:**

- `phase_index: int`               (0..3)
- `king_sq_white: int`             (0..63)
- `king_sq_black: int`             (0..63)
- `nnue_feat_white: List[int]`     aktive HalfKA-Indizes ~20–30
- `nnue_feat_black: List[int]`     aktive HalfKA-Indizes ~20–30

**Pro Kandidatenzug zusätzlich (für die ersten K legalen Züge):**

- `move_type: int`                 Hand-Hash auf 0..127
- `delta_white_leave: List[int]`   verlassende Indizes (≤4)
- `delta_white_enter: List[int]`   neue Indizes (≤4)
- `delta_black_leave: List[int]`
- `delta_black_enter: List[int]`
- `is_white_king_move: bool`       (für Dirty-Flag)
- `is_black_king_move: bool`

**Aufgaben:**

1. `engine/features/phase.py` neu anlegen mit `phase_score(pos)` und
   `phase_index(pos)` exakt wie in Paper Abschnitt 4.1 beschrieben.

2. `engine/features/nnue_features.py`:

   ```python
   FEATURES_PER_KING = 64 * 12          # piece_sq * piece_type_with_color
   TOTAL_FEATURES    = 64 * FEATURES_PER_KING

   def halfka_index(king_sq: int, piece_sq: int, piece_type_color: int) -> int:
       return king_sq * FEATURES_PER_KING + piece_sq * 12 + piece_type_color

   def halfka_active_indices(board, perspective: str) -> list[int]:
       """Aktive HalfKA-Indizes aus Sicht des angegebenen Königs.
       Eigener König ist NICHT Feature, er definiert den Index."""
       ...
   ```

   Reference: Stockfish NNUE HalfKA Spezifikation. Piece-Type-Encoding:
   `{P:0, N:1, B:2, R:3, Q:4, K:5}` + `6*is_opponent`. Der eigene König
   wird ausgelassen.

3. `engine/features/move_delta.py`:

   ```python
   def halfka_delta(board, move, perspective: str) -> tuple[list[int], list[int]]:
       """Gibt (leaving, entering) HalfKA-Indizes für diesen Zug aus
       der angegebenen Königsperspektive zurück."""
       ...

   def is_king_move(board, move, perspective: str) -> bool:
       """True wenn dieser Zug das Königsfeld der angegebenen Seite ändert.
       Berücksichtigt Rochade."""
       ...

   def move_type_hash(board, move) -> int:
       """Hand-Hash der Move-Klasse aus
       (piece, capture, promotion, special) auf 0..127."""
       ...
   ```

4. Den Artefakt-Builder erweitern, sodass die neuen Felder geschrieben
   werden. **Bestehende Felder unverändert.** Pro Sample werden die
   ersten K=64 legalen Züge mit Move-Deltas annotiert (K als
   Konfigurationsparameter, Default 64).

5. Den Artefakt-Loader abwärtskompatibel machen: fehlt ein neues Feld,
   gib Dummy-Default zurück und logge einmalig eine Warnung.

6. Unit-Tests:
   - `test_phase_index_start_position()` → 0
   - `test_phase_index_kbk_endgame()` → 3
   - `test_halfka_start_position_counts()` (30 aktive Features pro Seite,
     König nicht enthalten)
   - `test_halfka_index_uniqueness()` (alle 49152 Indizes sind paarweise
     verschieden für verschiedene Inputs)
   - `test_move_delta_quiet_pawn()` (1 leave, 1 enter pro Perspektive)
   - `test_move_delta_capture()` (2 leave, 1 enter pro Perspektive)
   - `test_move_delta_promotion()` (1 leave, 1 enter mit verschiedenem
     Piece-Type)
   - `test_move_delta_castling_marks_king_move()`
   - `test_move_delta_consistency_with_full_rebuild()`: für 100 zufällige
     Stellungen + 10 Züge pro Stellung verifizieren, dass `a_root +
     apply_delta == full_rebuild(after_move)` (sofern nicht King-Move)
   - `test_old_artefact_still_loads()` mit LAPv1-Fixture
   - `test_move_type_hash_in_range()`

7. Mini-Regenerierungs-Script `tools/rebuild_artifact.py` dokumentieren.

**Acceptance:**
- Alle neuen Tests grün.
- LAPv1-Trainingspipeline läuft einen Step auf neu gebautem Artefakt
  und Loss-Werte sind atol=1e-6 identisch zu vor dem Schritt.
- Insbesondere `test_move_delta_consistency_with_full_rebuild` ist
  GROUND TRUTH für alle weiteren NNUE-Schritte.

**Rollback:** Felder-Writer im Builder auskommentieren, Loader-Default
entfernt alle neuen Keys.

---

## Schritt 3 — Phase-Router und generischer Phase-MoE-Wrapper

**Ziel:** Beide Bausteine auf einmal anlegen, weil sie kompakt sind und
sich gegenseitig brauchen.

**Aufgaben:**

1. `engine/nn/phase_router.py`:

   ```python
   class PhaseRouter(nn.Module):
       NUM_PHASES = 4
       def forward(self, batch) -> torch.LongTensor:  # (B,)
           return batch["phase_index"]
   ```

2. `engine/nn/phase_moe.py`:

   ```python
   class PhaseMoE(nn.Module):
       def __init__(self, make_expert: Callable[[], nn.Module],
                    num_phases: int = 4):
           super().__init__()
           self.experts = nn.ModuleList([make_expert() for _ in range(num_phases)])

       def forward(self, x, phase_idx: torch.LongTensor, **kwargs):
           # hartes Routing pro Sample
           ...

       @classmethod
       def from_single(cls, pretrained: nn.Module, num_phases: int = 4):
           """Initialisiert alle Experten als deep-copy."""
           import copy
           moe = cls.__new__(cls)
           nn.Module.__init__(moe)
           moe.experts = nn.ModuleList([copy.deepcopy(pretrained) for _ in range(num_phases)])
           return moe
   ```

   - Hartes Routing, kein Top-k.
   - Achte auf nicht-batched kwargs (Positional Encodings etc): diese
     dürfen nicht gesliced werden. Konvention dokumentieren.

3. Tests:
   - `test_phase_router_passthrough()`
   - `test_phase_moe_equivalent_to_single_after_init()` — bei `from_single`
     mit identischer Phase bitgleich zum Original.
   - `test_phase_moe_routes_correctly()` mit Mock-Experten.
   - `test_phase_moe_gradients_only_flow_to_active_expert()`.

**Acceptance:** Tests grün, Module noch nirgends importiert.

**Rollback:** Dateien löschen.

---

## Schritt 4 — Dual-Accumulator-Infrastruktur

**Ziel:** Den NNUE-Datenpfad einführen, OHNE bereits einen Head daran zu
hängen. Dieser Schritt baut die Klassen, die in Schritt 6 (Value),
Schritt 8 (Policy) und Schritt 12 (Cache in der Loop) konsumiert werden.
Flag: `lapv2.dual_accumulator`.

**Aufgaben:**

1. `engine/nn/feature_transformer.py`:

   ```python
   class FeatureTransformer(nn.Module):
       """Eine pro Phase. Ist im Kern eine EmbeddingBag mit mode='sum',
       um die sparsame Index-Summe effizient zu rechnen."""
       def __init__(self, num_features: int = 49152, accumulator_dim: int = 64):
           super().__init__()
           self.ft = nn.EmbeddingBag(num_features, accumulator_dim, mode="sum")

       def build(self, indices: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
           """Vollständiger Aufbau eines Accumulators aus aktiven Indizes."""
           return self.ft(indices, offsets)

       def gather_rows(self, indices: torch.Tensor) -> torch.Tensor:
           """Holt FT-Zeilen für inkrementelle Updates. Shape (n, N)."""
           return self.ft.weight[indices]
   ```

2. `engine/nn/dual_accumulator.py`:

   ```python
   class DualAccumulatorBuilder:
       """Pure Funktion: nimmt FT_phase und Batch-Felder, baut a_white
       und a_black für jede Stellung im Batch."""
       def __call__(self, ft: FeatureTransformer, batch) -> tuple[Tensor, Tensor]:
           a_white = ft.build(batch["nnue_feat_white_indices"],
                              batch["nnue_feat_white_offsets"])
           a_black = ft.build(batch["nnue_feat_black_indices"],
                              batch["nnue_feat_black_offsets"])
           return a_white, a_black

   class IncrementalAccumulator:
       """Pflegt einen Accumulator über Move-Deltas. Verwendet im
       Eval-Pfad, NICHT im Trainingspfad."""
       def __init__(self, ft: FeatureTransformer):
           self.ft = ft
           self.a_white = None
           self.a_black = None
           self.dirty_white = False
           self.dirty_black = False

       def init_from_position(self, batch_pos): ...
       def apply_move(self, leave_w, enter_w, leave_b, enter_b,
                      is_king_w, is_king_b, full_rebuild_fn=None): ...
       def get(self, perspective: str) -> Tensor:
           """Erzwingt lazy Rebuild bei dirty flag."""
           ...
   ```

3. Collator-Erweiterung: in der Datenpipeline werden die sparsen
   Index-Listen der NNUE-Features in `(indices, offsets)`-Tupel für
   `EmbeddingBag` umgewandelt. Identisch für `nnue_feat_white` und
   `nnue_feat_black`.

4. **Wichtig:** In diesem Schritt wird der FT noch nicht in das Modell
   integriert. Die Klassen liegen einsatzbereit, werden aber nur in den
   Tests verwendet.

5. Tests:
   - `test_ft_build_shape()` (B, N)
   - `test_ft_build_matches_manual_sum()` für 5 Stellungen
   - `test_dual_accumulator_independent_white_black()`
   - `test_incremental_apply_move_matches_full_rebuild()` —
     ist der Hauptintegritätstest. Verwende denselben Stellungs-/Zug-Pool
     wie in `test_move_delta_consistency_with_full_rebuild` aus Schritt 2.
   - `test_incremental_king_move_triggers_rebuild()`
   - `test_dirty_flag_lazy_rebuild()`

**Acceptance:**
- Alle Tests grün.
- Insbesondere die Inkremental-Konsistenz ist atol=1e-6 erfüllt.
- Top-Level-Modell unverändert, Trainingspipeline läuft wie vor dem Schritt.

**Rollback:** Klassen löschen, Collator-Pfad ausschalten. Keine
Modell-Änderung betroffen.

---

## Schritt 5 — Phase-MoE in Intention Encoder und State Embedder

**Ziel:** Erste produktive Verwendung des `PhaseMoE`. Aktiviert durch Flag
`lapv2.phase_moe`.

**Aufgaben:**

1. In der Top-Level-Modellklasse die beiden Attribute `intention_encoder`
   und `state_embedder` bedingt durch `PhaseMoE.from_single(...)`
   ersetzen, wenn `cfg.lapv2.phase_moe == True`.

2. Forward-Pass anpassen: Phase-Tensor aus dem Batch über `PhaseRouter`
   holen, an beide MoE-Module durchreichen.

3. Checkpoint-Loader erweitern:
   - LAPv1-Checkpoint + Flag aktiv → `from_single`-Logik (4 Kopien).
   - LAPv2-Checkpoint → pro Experte separat laden.
   - `lapv2_version`-Feld im Checkpoint speichern.

4. Tests:
   - `test_lapv1_checkpoint_loads_into_phase_moe()` + Forward-Pass
     bitgleich auf Fix-Batch.
   - `test_phase_moe_off_is_bit_identical_to_v1()`.
   - `test_phase_moe_on_runs_training_step()` (no NaN).

**Acceptance:**
- Mit Flag off bitgleich zu LAPv1.
- Mit Flag on + LAPv1-Checkpoint bitgleich (weil from_single).
- Voller Trainingsschritt mit Flag on läuft ohne NaN.

**Rollback:** Flag auf false.

---

## Schritt 6 — NNUE-Value-Head auf geteiltem FT (single phase)

**Ziel:** Den alten dichten ValueHead durch eine NNUE-artige
Implementierung ersetzen, **mit FT als top-level Modul, der explizit für
spätere Wiederverwendung durch den Policy-Head ausgelegt ist**. In diesem
Schritt noch ohne Phase-MoE. Flag: `lapv2.nnue_value`.

**Aufgaben:**

1. Erweitere die Top-Level-Modellklasse um ein neues Attribut
   `self.ft: FeatureTransformer` (eine Instanz, single phase). Dieses
   Attribut wird ab jetzt das gemeinsame "FT für alle NNUE-artigen Heads"
   sein.

2. `engine/nn/value_head_nnue.py`:

   ```python
   class NNUEValueHead(nn.Module):
       def __init__(self, accumulator_dim: int = 64,
                    hidden: int = 32):
           super().__init__()
           self.adapter = nn.Linear(accumulator_dim, accumulator_dim)
           in_dim = 2 * accumulator_dim
           self.dense = nn.Sequential(
               nn.Linear(in_dim, hidden),
               ClippedReLU(),
               nn.Linear(hidden, 1),
           )

       def forward(self, a_stm: Tensor, a_other: Tensor) -> Tensor:
           a_stm   = clipped_relu(self.adapter(a_stm))
           a_other = clipped_relu(self.adapter(a_other))
           x = torch.cat([a_stm, a_other], dim=-1)
           return self.dense(x).squeeze(-1)
   ```

   `NNUEValueHead` empfängt fertige Accumulator. Es ist die Aufgabe der
   Top-Level-Modellklasse, FT auf den Batch anzuwenden und die richtige
   side-to-move-orientierte Reihenfolge zu bestimmen.

3. `ClippedReLU` als `torch.clamp(x, 0.0, 1.0)` Modul.

4. In der Top-Level-Modellklasse:

   ```python
   if cfg.lapv2.nnue_value:
       a_white, a_black = dual_acc_builder(self.ft, batch)
       stm_white_mask = batch["side_to_move"] == 0
       a_stm   = torch.where(stm_white_mask[:, None], a_white, a_black)
       a_other = torch.where(stm_white_mask[:, None], a_black, a_white)
       value = self.value_head_nnue(a_stm, a_other)
   else:
       value = self.value_head_legacy(h)   # alter Pfad
   ```

5. Checkpoint-Handling: bei Flag-Switch lädt der Loader die alten
   Value-Head-Gewichte **nicht**. Der NNUE-Pfad wird frisch initialisiert
   (FT std=1/√N). Warn-Log.

6. Inkrementalität wird in diesem Schritt **NICHT** im Eval-Pfad genutzt
   (das macht Schritt 12). Hier immer voller Rebuild pro Forward-Pass.

7. Tests:
   - `test_nnue_value_forward_shape()`
   - `test_nnue_value_start_position_finite()` (kein NaN)
   - `test_nnue_value_loss_decreases()` (10 Steps Mini-Batch)
   - `test_flag_off_uses_legacy_value_head()` (bitgleich zu LAPv1)
   - `test_ft_attribute_exists_for_future_policy()` (sanity check)

**Acceptance:** Tests grün. Mit Flag on läuft T1-Mini-Epoch ohne NaN, Loss
fällt messbar.

**Rollback:** Flag off, alter `value_head_legacy` bleibt im Modul.

---

## Schritt 7 — NNUE-Value-Head als Phase-MoE

**Ziel:** Den FT und den NNUE-Value-Head in einen Phase-MoE-Wrapper
einziehen. Aus `self.ft: FeatureTransformer` wird `self.ft:
PhaseMoE[FeatureTransformer]`. Aus `self.value_head_nnue: NNUEValueHead`
wird `self.value_head_nnue: PhaseMoE[NNUEValueHead]`. Flag:
`lapv2.nnue_value_phase_moe`.

**Aufgaben:**

1. Im Top-Level: bei aktivem Flag wird `self.ft` zu
   `PhaseMoE.from_single(FeatureTransformer(...))` und
   `self.value_head_nnue` zu `PhaseMoE.from_single(NNUEValueHead(...))`.

2. Der `dual_acc_builder` wird leicht erweitert: er bekommt zusätzlich
   den `phase_idx`-Tensor und ruft pro Sample den richtigen Phase-Experten
   des FT auf.

3. **Phase-Curriculum-Gate:** Config-Option
   `lapv2.nnue_phase_gate_steps: int`. Solange diese Step-Zahl nicht
   erreicht ist, werden nach jedem Optimizer-Step die FT-Phase-Experten
   per Hook auf ihren Mittelwert gezogen, ebenso die
   ValueHead-Phase-Experten. Nach dem Gate-Ende: Hook deaktivieren.

4. Tests:
   - `test_phase_nnue_value_warm_start_matches_single()` (bei frischem
     `from_single` und identischer Phase bitgleich)
   - `test_gate_mean_pull_converges_experts()` (nach 3 Steps mit Gate
     sind alle Experten gleich)
   - `test_phase_nnue_value_runs_all_phases()`

**Acceptance:** Mini-Training mit Flag grün, Loss fällt, keine NaN.

**Rollback:** Flag off, FT fällt auf Single-Instance zurück (Schritt 6).

---

## Schritt 8 — NNUE-Policy-Head auf demselben FT

**Ziel:** Der bestehende `PolicyHead` wird durch einen NNUE-artigen
Policy-Head ersetzt, der den FT mit dem Value-Head teilt. Dieser Schritt
ist der größte und semantisch tiefgreifendste; ich empfehle, ihn auf
mindestens drei Sub-PRs aufzuteilen.

Flag: `lapv2.nnue_policy`.

### 8.1 Move-Type-Embedding und MoveHead-Modul

`engine/nn/policy_head_nnue.py`:

```python
class NNUEPolicyHead(nn.Module):
    def __init__(self, accumulator_dim: int = 64,
                 move_type_vocab: int = 128,
                 move_type_dim: int = 16,
                 hidden: int = 32):
        super().__init__()
        self.adapter      = nn.Linear(accumulator_dim, accumulator_dim)
        self.move_type_emb = nn.Embedding(move_type_vocab, move_type_dim)
        in_dim = 2 * accumulator_dim + move_type_dim
        self.move_head = nn.Sequential(
            nn.Linear(in_dim, hidden),
            ClippedReLU(),
            nn.Linear(hidden, 1),
        )

    def score_moves(self,
                    a_root_stm: Tensor,         # (B, N)
                    a_succ_other: Tensor,       # (B, K, N)
                    move_type_ids: Tensor       # (B, K)
                    ) -> Tensor:                # (B, K)
        a_root = clipped_relu(self.adapter(a_root_stm))            # (B, N)
        a_root = a_root.unsqueeze(1).expand(-1, a_succ_other.size(1), -1)
        a_succ = clipped_relu(self.adapter(a_succ_other))          # (B, K, N)
        emb    = self.move_type_emb(move_type_ids)                 # (B, K, D)
        x      = torch.cat([a_root, a_succ, emb], dim=-1)
        return self.move_head(x).squeeze(-1)                       # (B, K)
```

### 8.2 Successor-Accumulator-Berechnung im Trainingspfad

Die Successor-Accumulator pro Kandidat müssen IM TRAININGSPFAD aus den
vorberechneten Move-Deltas (Schritt 2) gebaut werden. Effizient via
`EmbeddingBag` über die Delta-Indizes:

```python
def successor_accumulators_train(ft: FeatureTransformer,
                                 a_root: dict[str, Tensor],
                                 batch) -> dict[str, Tensor]:
    """Berechnet pro Kandidat (B, K, N) Successor-Accumulator
    für white und black."""
    # leave/enter pro Kandidat sind bereits als (indices, offsets)
    # im Batch vorbereitet
    leave_w = ft.build(batch["delta_white_leave_idx"],
                       batch["delta_white_leave_offsets"])  # (B*K, N)
    enter_w = ft.build(batch["delta_white_enter_idx"],
                       batch["delta_white_enter_offsets"])
    a_white_succ = a_root["white"].repeat_interleave(K, dim=0) - leave_w + enter_w
    # gleiches für black
    # bei is_*_king_move: voller Rebuild aus gespeicherten
    # nnue_feat_*_after_move (optional, in Schritt 8.3)
    return {"white": a_white_succ.view(B, K, N),
            "black": a_black_succ.view(B, K, N)}
```

### 8.3 King-Move-Behandlung im Trainingspfad

Bei King-Moves enthält das Delta-Schema NICHT alle nötigen Änderungen
(weil sich die Königsindizierung selbst ändert). Lösungsoptionen:

- **Option A (einfach):** Im Artefakt für King-Move-Kandidaten direkt das
  vollständige `nnue_feat_*_after_move` als Index-Liste mitspeichern.
  Pro King-Move ein zusätzlicher voller Index-Eintrag. Mehrkosten klein,
  weil King-Moves selten sind.
- **Option B (effizient):** Im Trainingspfad King-Move-Kandidaten in
  einen separaten Sub-Batch trennen und für sie den Accumulator über
  `ft.build()` neu aufbauen.

**Empfohlen: Option A.** Schritt 2 wird entsprechend erweitert
(`nnue_feat_white_after_move` und `nnue_feat_black_after_move` werden
optional pro Kandidat gespeichert, nur wenn `is_*_king_move`).

### 8.4 Integration in Top-Level

```python
if cfg.lapv2.nnue_policy:
    a_w, a_b = dual_acc_builder(self.ft, batch, phase_idx)  # vom Value-Pfad
    a_root_stm   = where_stm(a_w, a_b, batch["side_to_move"])
    a_succ       = successor_accumulators_train(self.ft, {...}, batch)
    a_succ_other = where_stm(a_succ["black"], a_succ["white"], batch["side_to_move"])
    move_logits  = self.policy_head_nnue.score_moves(
        a_root_stm, a_succ_other, batch["move_type_ids"])
else:
    move_logits  = self.policy_head_legacy(...)
```

Bei `lapv2.nnue_value_phase_moe == True` und `lapv2.nnue_policy == True`
wird auch der Policy-Head als `PhaseMoE.from_single(NNUEPolicyHead(...))`
gewrappt.

### 8.5 Loss-Balance Value/Policy

Weil beide Heads denselben FT konsumieren, müssen Gradienten balanciert
werden. Erweiterung des Loss-Computes um zwei Schalter:

```yaml
lapv2.loss_balance:
  value_loss_norm: ema     # ema der laufenden Value-Loss-Größe
  policy_loss_norm: ema
  adapter_decoupling: 0.0  # 0 = aus, >0 = Cosine-Penalty zwischen V- und P-Adapter
```

Implementiere die EMA-Normierung als einfache `running_mean` mit
Momentum 0.99. Cosine-Penalty optional.

### 8.6 Tests

- `test_policy_nnue_forward_shape()` (B, K)
- `test_policy_nnue_score_invariant_under_kings_move()`: Bei einem
  Königszug muss der Successor-Accumulator über den `_after_move`-Index
  gebaut werden, und das Ergebnis muss exakt mit `ft.build(features_after)`
  übereinstimmen.
- `test_policy_nnue_logits_change_with_move_type_embed()`
- `test_policy_nnue_loss_decreases()`
- `test_flag_off_uses_legacy_policy()` (bitgleich zu LAPv1)
- `test_value_and_policy_share_ft_object()` (sanity: `id(self.ft) ==
  id(value_head_path) == id(policy_head_path)`)
- `test_value_and_policy_gradients_both_flow_to_ft()`
- `test_loss_balance_keeps_ema_in_range()`

**Acceptance:**
- Alle Tests grün.
- Mit Flag on läuft T1-Mini-Epoch stabil.
- Pro Optimizer-Step werden FT-Gradienten von BEIDEN Loss-Termen
  beigetragen (verifiziert via Gradient-Hooks).

**Rollback:** Flag off, `policy_head_legacy` aktiv.

---

## Schritt 9 — Sharpness-Head als Phase-MoE

**Ziel:** Der kleinste verbliebene Head wird ebenfalls auf Phase-MoE
umgestellt. Bewusst NICHT NNUE-artig — Sharpness ist ein globales,
strukturelles Signal und soll explizit nicht königsfeldgebunden sein.

Flag: `lapv2.sharpness_phase_moe`.

**Aufgaben:**

1. `SharpnessHead` via `PhaseMoE.from_single(...)` einziehen.
2. Tests:
   - `test_sharpness_phase_moe_runs()`
   - `test_flag_off_bit_identical_to_v1()`

**Acceptance:** Tests grün, Mini-Epoch stabil.

**Rollback:** Flag off.

---

## Schritt 10 — Shared-Backbone-Opponent-Readout

**Ziel:** OpponentHead durch Δ-Operator + drei leichte Readouts ersetzen.
Flag: `lapv2.shared_opponent_readout`.

**Aufgaben:**

1. `engine/nn/opponent_readout.py`:

   ```python
   class DeltaOperator(nn.Module):
       """1-2 Layer Residual mit Cross-Attention zwischen h_root und move_embed."""
       def forward(self, h_root, move_embed): ...

   class OpponentReadout(nn.Module):
       def __init__(self, d_model, num_reply_slots):
           self.delta = DeltaOperator(d_model)
           self.reply_head       = nn.Linear(d_model, num_reply_slots)
           self.pressure_head    = nn.Linear(d_model, 1)
           self.uncertainty_head = nn.Linear(d_model, 1)

       def forward(self, h_root, move_embed, legal_reply_mask):
           h_next = self.delta(h_root, move_embed)
           pooled = h_next.mean(dim=1)
           reply_logits = self.reply_head(pooled).masked_fill(~legal_reply_mask, -1e9)
           pressure     = self.pressure_head(pooled).sigmoid().squeeze(-1)
           uncertainty  = self.uncertainty_head(pooled).sigmoid().squeeze(-1)
           return reply_logits, pressure, uncertainty
   ```

2. Im Wrapper der Deliberation-Loop: bei aktivem Flag die drei Signale
   aus `OpponentReadout` ziehen. Aggregationsformel unverändert:
   `reply_signal = best_reply - 10 * pressure - 10 * uncertainty`.

3. Legacy-Pfad bleibt erhalten. Bei Flag-Switch wird `OpponentReadout`
   frisch initialisiert.

4. Tests:
   - `test_opponent_readout_output_shapes()`
   - `test_reply_signal_identical_api_to_v1()` (Fake-Readout-Test)
   - `test_flag_off_uses_legacy_opponent()`

**Acceptance:** Tests grün, Arena-Smoke läuft.

**Rollback:** Flag off.

---

## Schritt 11 — Distillations-Loss für Opponent-Readout

**Ziel:** Δ-Readout lernt aus einem Lehrer (Hauptnetz auf echter
Successor-Stellung). Flag: `lapv2.distill_opponent`.

**Aufgaben:**

1. Trainings-Hook: für `distill_fraction` der Batches wird das Hauptnetz
   im No-Grad-Modus auf s' aufgerufen, liefert Teacher-Targets für
   reply_logits, pressure, uncertainty.

2. Loss:
   `L_distill = λ_r * KL(reply || teacher_reply)
              + λ_p * MSE(pressure, teacher_pressure)
              + λ_u * MSE(uncertainty, teacher_uncertainty)`
   Defaults: `λ_r=1.0, λ_p=0.5, λ_u=0.5`.

3. Laufzeit unverändert.

4. Tests:
   - `test_distill_loss_positive_on_random_init()`
   - `test_distill_loss_zero_when_match_teacher()`
   - `test_runtime_unchanged_with_distill_flag_on()`

**Acceptance:** Tests grün, Trainingskurven-Smoke zeigt Konvergenz zum
Teacher.

**Rollback:** Flag off.

---

## Schritt 12 — Deliberation-Loop: AccumulatorCache und Phase-Fixierung

**Ziel:** Die Loop nutzt jetzt den `IncrementalAccumulator` aus Schritt 4
im Eval-Pfad. Phase wird an der Wurzel fixiert. Successor-Accumulator
wird zwischen Policy-Scoring und Value-Refinement gecacht.

Flag: `lapv2.accumulator_cache`.

**Aufgaben:**

1. **Phase-Fixierung:** Phase-Index an der Wurzel berechnen, an alle
   inner steps durchreichen, Assertion für Konstanz pro Loop.

2. **AccumulatorCache:** Im Top der Loop wird `IncrementalAccumulator`
   für den FT-Phase-Experten initialisiert. Pro Kandidatenzug werden die
   bekannten Move-Deltas geholt; bei King-Move voller Rebuild via
   `ft.build()`.

3. **Wiederverwendung Policy → Value:** Wenn ein Kandidat in das
   Refinement-Top-k aufgenommen wird, wird der bereits berechnete
   Successor-Accumulator (aus Policy-Scoring) im Value-Refinement
   wiederverwendet. Cache-Key: `(candidate_id, inner_step)`.

4. Tests:
   - `test_phase_constant_over_loop()`
   - `test_accumulator_cache_eval_matches_no_cache()` — bitgleich auf
     einer Fix-Suite.
   - `test_cache_hit_for_top_k_candidates()` (mit Hook-Counter)

**Acceptance:** Tests grün. Eval-Run mit und ohne Cache liefert bitgleich
identische Policy-/Value-Outputs.

**Rollback:** Cache deaktivieren.

---

## Schritt 13 — T1/T2 Pipeline: Warm-Start, Loss-Balance, Phase-Gate

**Ziel:** Trainingsinfrastruktur auf LAPv2 anpassen.

**Aufgaben:**

1. **Warm-Start-Loader:** Utility, das einen LAPv1-T2-Checkpoint nimmt
   und einen LAPv2-Init-Checkpoint schreibt:
   - Phase-MoE-Module: 4 identische Kopien
   - FT (Phase-MoE): frisch initialisiert (std = 1/√N)
   - NNUEValueHead, NNUEPolicyHead: frisch initialisiert
   - OpponentReadout: frisch initialisiert
   - Alles andere 1:1 übernommen

2. **Load-Balancing-Regularisierer:** `L_balance = Σ_p w_p * L_p` mit
   `w_p = clamp(1/freq_p, min=0.5)`.

3. **Zweistufiges Phase-Curriculum-Gate:**
   - Stufe A (`gate_stage_a_steps`): Phase-MoE in Encoder/Embedder aktiv,
     NNUE-Heads als Phase-Mittel-Hook.
   - Stufe B (`gate_stage_b_steps`): NNUE-Heads ebenfalls phase-spezifisch.

4. **Adaptive Loss-Balance Value/Policy** wie in Schritt 8.5
   spezifiziert.

5. **Logging pro Epoch:**
   - Expertenauslastung pro Phase
   - Value-Loss und Policy-Loss getrennt pro Phase
   - Norm-Abstand der 4 FT-Experten (FT-Drift)
   - Cosine-Distanz V_adapter ↔ P_adapter (Adapter-Spezialisierung)
   - Reply-Consistency (Korrelation Readout vs. Lehrer, falls Distill an)

6. Tests:
   - `test_warm_start_checkpoint_forward_matches_lapv1()` für die
     Komponenten, die NICHT frisch initialisiert werden.
   - `test_load_balancing_weights_in_range()`
   - `test_phase_gate_pulls_experts_in_stage_a()`
   - `test_phase_gate_releases_in_stage_b()`

**Acceptance:** Voller T1-Run (mehrere hundert Steps) mit allen Flags an
läuft stabil, Loss fällt, keine NaN, alle Logs vorhanden.

**Rollback:** Flags abschalten, Trainer fällt auf LAPv1-Pfad zurück.

---

## Schritt 14 — Arena-Harness, Reporting, Release-Gate

**Ziel:** Fair reproduzierbarer Vergleich LAPv1 vs. LAPv2.

**Aufgaben:**

1. Arena-Runner um LAPv2-Checkpoint-Typ erweitern.
2. Vergleichsprotokoll erzeugt Report mit:
   - Elo-Differenz pro Budget (inner0, inner1, inner2, auto4)
   - Signifikanz via SPRT oder Binomial-CIs
   - Phase-Perplexity, FT-Drift, Reply-Consistency
   - Policy-Value-Konsistenz: Korrelation zwischen NNUE-Policy-Logit eines
     Zugs und negiertem NNUE-Value des Successors (sollte hoch sein, weil
     sie denselben FT teilen)
3. **Release-Gate:** LAPv2-Checkpoint ist release-würdig, wenn in
   mindestens drei der vier Budgets kein signifikant schlechterer Elo
   gegenüber LAPv1 gemessen wird und in mindestens einem Budget
   signifikant besser. Besonderes Augenmerk auf inner1 als
   Out-of-Distribution-Indikator.
4. Reports nach `reports/lapv2/run_<date>/`.

**Acceptance:** Arena-Runner läuft auf 5 Openings mit beiden Checkpoints
und erzeugt den vollen Report ohne Fehler.

**Rollback:** Arena-Runner hat LAPv1-Pfad als Default.

---

## Anhang A — Codex-CLI Task-Vorlage

```
Lies docs/lapv2/plan.md und führe ausschließlich Schritt <N> aus.
Arbeite auf Branch lapv2/step-<N>-<slug>.
Halte dich exakt an Acceptance Criterion und Rollback-Regel.
Wenn du Unklarheiten findest, aktualisiere docs/lapv2/path_map.md
bevor du Code schreibst.
Am Ende: Tests, Lint, Changelog-Eintrag, PR-Beschreibung mit
Acceptance-Check-Liste.
```

## Anhang B — Abhängigkeitsgraph

```
  1 ──> 2 ──> 3 ──> 4 ──┐
                        ├─> 6 ──> 7 ──┐
                        │             ├─> 12 ──> 13 ──> 14
                        └─> 8 ────────┤
                  3 ──> 5 ────────────┤
                                  9 ──┤
                                 10 ──┤
                                 11 ──┘
```

- Schritt 4 (Dual-Accumulator-Infrastruktur) ist die Basis für Value
  und Policy NNUE.
- Schritt 6 (NNUE-Value single) muss vor Schritt 7 (Value Phase-MoE)
  und vor Schritt 8 (Policy NNUE) kommen, weil 6 das `self.ft`-Attribut
  einführt, das 8 wiederverwendet.
- Schritt 8 (NNUE-Policy) hängt von 4, 6 ab. Idealerweise auch nach 7,
  damit beide Heads parallel als Phase-MoE laufen.
- Schritte 9, 10, 11 sind orthogonale Spuren, die nach 5 starten können.
- Schritt 12 hängt von 4, 6, 7, 8 ab.
- Schritt 13 fasst alles zusammen.

## Anhang C — Abbruchkriterien

Stoppe sofort und öffne ein Issue statt einen PR, wenn:

- Ein Schritt verlangt Änderungen an Modulen, die nicht in
  `path_map.md` stehen.
- Acceptance Criterion lässt sich ohne zusätzliche, nicht im Plan
  vorgesehene Flags nicht erfüllen.
- Ein früherer Schritt erzeugt numerische Drift gegenüber LAPv1, die
  nicht durch den jeweiligen Warm-Start erklärbar ist.
- Parameterzahl nach Schritt 8 weicht um mehr als 10 % vom Paper-
  Schätzwert (~58,6 M) ab.
- `test_value_and_policy_share_ft_object` schlägt fehl — das ist die
  zentrale Architektur-Invariante von Rev 3.
- `test_incremental_apply_move_matches_full_rebuild` schlägt fehl —
  die NNUE-Inkrementalität ist die zweite zentrale Invariante.

## Anhang D — Zukünftige Vereinfachung (nicht in diesem Plan)

Schritt 8 implementiert die **eigenständige NNUE-Policy** mit eigenem
Adapter, MoveHead und Move-Type-Embedding. Eine spätere, radikalere
Variante wäre die **volle Value-Policy-Unifikation**:

`logit(m) = − NNUEValueHead( a_succ_other, a_succ_stm )`

also Policy = negierte Value-Bewertung der Successor-Stellung. Das würde
den Policy-Head als trainierbare Komponente ganz eliminieren und
Konsistenz zwischen Value und Policy strukturell garantieren. Die
Migration wäre nicht-trivial, weil sich die Loss-Semantik ändert. Diese
Vereinfachung wird **explizit nicht in diesem Plan** verfolgt und ist
Gegenstand eines möglichen Rev-4-Plans.
