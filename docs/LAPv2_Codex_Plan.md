# LAPv2 Codex-CLI Implementierungsplan (Rev 3.1)

> Begleitplan zum Paper **LAPv2 – Phasen-MoE und NNUE-artige Heads**
> (Rev 3.1, Datei `LAPv2_Paper.md`).
> Zielrepo: `https://github.com/TotoBa/EngineKonzept.git` (Branch `lapv2`).
> Zielwerkzeug: **Codex-CLI**. Jeder Schritt ist eine in sich
> abgeschlossene, rückfallsichere Task, die als einzelner Codex-Auftrag
> ausführbar ist.

> **Was ist neu in Rev 3.1 gegenüber Rev 3?**
>
> 1. **Neuer Schritt 0 — LAPv1-Baseline-Reproduktion.** Der Lauf
>    `stage2_fast_arena_all_unique_v4` ist vor dem Arena-Block
>    abgebrochen und liefert keine nutzbaren LAPv1-Elo-Zahlen. Ohne
>    diese Zahlen ist kein LAPv2-Gewinn belegbar. Schritt 0 reproduziert
>    die LAPv1-Stage-2-Pipeline sauber bis zum Arena-Report und legt
>    drei benannte Baseline-Checkpoints an.
> 2. **Warm-Start-Quelle präzisiert.** Der Warm-Start für LAPv2 erfolgt
>    aus dem `freeze_inner_hard`-Checkpoint mit dem höchsten joint
>    `val_top1`, nicht aus dem späteren `joint_heads_mix`- oder
>    `joint_backbone_mix`-Checkpoint. Begründung siehe
>    `LAPv2_Paper.md` §15.
> 3. **Regression-Watchdog** als Querschnittsregel für alle Schritte ab
>    Schritt 4. Misst joint-`val_top1` vor und nach jedem Flag-Flip und
>    rollt zurück, wenn mehr als 1 pp verloren geht.
> 4. **Release-Gate um die hard/selection-Teilmenge erweitert.** Nicht
>    nur joint-Elo, sondern auch selection-Elo muss nicht signifikant
>    schlechter werden, und in mindestens einem Budget signifikant
>    besser.
>
> Repo-Ausführungsnotiz (2026-04-08):
> Die Annahme aus Rev 3.1, `stage2_fast_arena_all_unique_v4` sei vor dem
> Arena-Block abgebrochen, ist im aktuellen Repo-Stand veraltet. Unter
> `/srv/schach/engine_training/phase10/lapv1_stage2_fast_arena_all_unique_v4/`
> liegen vollständige Verify- und Arena-Artefakte. Für die lokale
> Umsetzung dieses Plans wird `v4` deshalb als vorhandene LAPv1-Baseline
> behandelt; ein zusätzlicher Reproduktionslauf ist keine Vorbedingung
> mehr für Schritt 1 ff.
>
> Alle anderen Schritte (1–14) sind inhaltlich unverändert zu Rev 3 und
> werden hier nur in der Übersicht referenziert. Der Detailtext für
> Schritt 1–14 liegt in der Rev-3-Vorgängerdatei und wird von diesem
> Plan als maßgeblich übernommen; Rev 3.1 fügt lediglich den Schritt 0,
> präzisierte Acceptance-Kriterien in Schritt 13 und 14, und die
> Querschnittsregel „Regression-Watchdog" hinzu.

---

## 0. Vorbemerkungen für Codex

**Lies das zuerst, bevor du irgendetwas tust.** Dieser Plan nennt
Modul- und Dateinamen auf Basis der LAPv1-Architekturübersicht
(Intention Encoder, State Embedder, Policy Head, Value Head, Sharpness
Head, Opponent Head, Deliberation Loop, T1/T2-Training,
LAPv1-Artefakt). Die tatsächlichen Pfade im Repo können abweichen.
Bevor du die erste Codeänderung schreibst, erstellst du per `rg`/`find`
ein Mapping von logischem Namen zu echtem Pfad und hältst es in
`docs/lapv2/path_map.md` fest. Alle weiteren Schritte referenzieren die
logischen Namen.

**Harte Regeln für jeden Schritt:**

1. Arbeite auf einem eigenen Feature-Branch
   `lapv2/step-NN-<slug>`.
2. Keine nicht zum Schritt gehörenden Änderungen. Kein Drive-By-
   Refactoring.
3. Jeder Schritt endet mit: Tests grün, Lint grün, kurzer
   Changelog-Eintrag unter `docs/lapv2/CHANGELOG.md`.
4. Jeder Schritt hat ein explizites **Acceptance Criterion** und ein
   **Rollback**.
5. Wenn ein Schritt eine existierende Datei verändert, zuerst einen
   Golden-Test aufnehmen, der das bisherige Verhalten festhält.
6. LAPv1-Checkpoint-Kompatibilität ist non-negotiable: jeder Schritt
   bis einschließlich Schritt 9 muss einen Modus haben, in dem das
   Modell numerisch wie LAPv1 rechnet.
7. **(neu, Rev 3.1)** Ab Schritt 4 ist der **Regression-Watchdog**
   (siehe Querschnittsregel unten) Pflicht an jedem Flag-Flip.
8. **(neu, Rev 3.1)** Kein Schritt 1..14 startet, bevor Schritt 0
   erfolgreich abgeschlossen ist und die drei benannten
   Baseline-Checkpoints existieren.

**Globaler Konfigurationsschalter:** in der zentralen Config Toplevel-
Key `lapv2`:

```yaml
lapv2:
  enabled: false
  phase_moe: false           # Schritt 5
  dual_accumulator: false    # Schritt 4
  nnue_value: false          # Schritt 6
  nnue_value_phase_moe: false  # Schritt 7
  nnue_policy: false         # Schritt 8
  sharpness_phase_moe: false # Schritt 9
  shared_opponent_readout: false  # Schritt 10
  distill_opponent: false    # Schritt 11
  accumulator_cache: false   # Schritt 12
  N_accumulator: 64

  warm_start:
    source_checkpoint: "models/lapv1/baseline_v5/freeze_inner_hard_best.pt"
    require_min_val_top1: 0.90   # harte Schranke

  watchdog:
    enabled: true
    metric: "val_top1"
    epsilon: 0.01        # max. 1 pp Verlust erlaubt
    on_violation: "rollback_and_lower_lr"
    lr_factor_on_retry: 0.3
```

---

## Querschnittsregel — Regression-Watchdog (neu in Rev 3.1)

Motivation: Der Lauf `stage2_fast_arena_all_unique_v4` zeigt einen
reproduzierbaren Sprung von 0,9076 auf 0,8060 joint-`val_top1` beim
Übergang von `freeze_inner_hard` auf `joint_heads_mix`. Gleicher Sprung
wäre in LAPv2 an jedem Flag-Flip möglich, weil mehrere Flag-Flips
strukturell eine Head-Unfreeze-Operation plus Loss-Rebalance darstellen.

**Regel:** An jedem Stage-Übergang UND an jedem LAPv2-Flag-Flip ab
Schritt 4 gilt:

1. **Before-Messung**: vor dem Flip joint-Validation laufen lassen und
   `val_top1_before`, `root_top1_before`, `root_incorrect_gain_before`
   persistent loggen.
2. **Flip anwenden** und **N Smoke-Steps** trainieren (Default
   `N = 0`, d. h. nur Flag-Flip ohne Training, für reine
   Bitgleichheits-Schritte wie Schritt 3 und 5).
3. **After-Messung**: nach N Schritten joint-Validation laufen lassen.
4. **Prüfung**:
   - `val_top1_after ≥ val_top1_before − ε` mit ε = 0,01.
   - `root_incorrect_gain_after ≥ 0,5 · root_incorrect_gain_before`.
5. **Aktion bei Verletzung**: Checkpoint auf den Stand vor dem Flip
   zurücksetzen. Lernraten der gerade freigegebenen/neuen Parameter-
   gruppen um `lr_factor_on_retry` (Default 0,3) senken. Flip
   wiederholen. Nach 3 gescheiterten Versuchen Flag auf false
   zurücksetzen und Issue öffnen.

Technische Umsetzung: neuer Utility-Modul `engine/training/watchdog.py`
mit Funktion

```python
def watchdog_guarded_flip(
    run, flip_fn, val_fn,
    epsilon: float = 0.01,
    require_gain_preserved: bool = True,
) -> WatchdogResult: ...
```

Jeder Schritt ab 4 ruft dies am Übergang auf. Die Messwerte aller
Flips werden in `reports/lapv2/watchdog_log.jsonl` gesammelt.

**Test-Vorgaben für den Watchdog selbst:**

- `test_watchdog_accepts_noop_flip()` — Flag-Flip ohne Gewichtsänderung
  bleibt akzeptiert (bitgleich).
- `test_watchdog_rejects_synthetic_regression()` — künstlich injiziertes
  Gewichts-Shuffling wird erkannt und zurückgerollt.
- `test_watchdog_accepts_within_epsilon_drift()` — Drift kleiner als ε
  passiert.

---

## Schrittübersicht

| #  | Titel                                                                           | Flag                      | Risiko | Rev |
|----|---------------------------------------------------------------------------------|---------------------------|--------|-----|
| 0  | **LAPv1-Baseline-Reproduktion + Arena**                                         | –                         | niedr. | 3.1 |
| 1  | Repo-Audit und Path-Map                                                         | –                         | nil    | 3   |
| 2  | Artefakt-Erweiterung (phase, king_sq, NNUE-Features, Move-Deltas)               | –                         | niedr. | 3   |
| 3  | Phase-Router + generisches Phase-MoE-Modul                                      | –                         | niedr. | 3   |
| 4  | Dual-Accumulator-Infrastruktur                                                  | `dual_accumulator`        | mittel | 3   |
| 5  | Phase-MoE in Intention Encoder + State Embedder                                 | `phase_moe`               | mittel | 3   |
| 6  | NNUE-Value-Head auf geteiltem FT (single phase)                                 | `nnue_value`              | mittel | 3   |
| 7  | NNUE-Value-Head als Phase-MoE                                                   | `nnue_value_phase_moe`    | mittel | 3   |
| 8  | NNUE-Policy-Head (Successor-Scoring, geteilter FT)                              | `nnue_policy`             | hoch   | 3   |
| 9  | Sharpness-Head als Phase-MoE                                                    | `sharpness_phase_moe`     | niedr. | 3   |
| 10 | Shared-Backbone-Opponent-Readout mit Δ-Operator                                 | `shared_opponent_readout` | hoch   | 3   |
| 11 | Optionaler Distillations-Loss                                                   | `distill_opponent`        | niedr. | 3   |
| 12 | Deliberation-Loop: AccumulatorCache + Phase-Fixierung                           | `accumulator_cache`       | mittel | 3   |
| 13 | T1/T2 Pipeline: Warm-Start, Loss-Balance, Phase-Gate                            | –                         | hoch   | 3.1 |
| 14 | Arena-Harness, Reporting, Release-Gate                                          | –                         | niedr. | 3.1 |

Abhängigkeitsgraph unverändert, erweitert um Schritt 0 als neue Wurzel:

```
  0 ──> 1 ──> 2 ──> 3 ──> 4 ──┐
                              ├─> 6 ──> 7 ──┐
                              │             ├─> 12 ──> 13 ──> 14
                              └─> 8 ────────┤
                        3 ──> 5 ────────────┤
                                        9 ──┤
                                       10 ──┤
                                       11 ──┘
```

---

## Schritt 0 — LAPv1-Baseline-Reproduktion + Arena (neu in Rev 3.1)

**Ziel:** Eine saubere, vollständig gelaufene LAPv1-Stage-2-Referenz
herstellen, inklusive Arena, als unverrückbare Messlatte für alle
LAPv2-Schritte. Die drei benannten Baseline-Checkpoints werden explizit
gespeichert und sind Eingabe für Schritt 13.

**Kontext aus dem v4-Lauf:**

Der Lauf `stage2_fast_arena_all_unique_v4` (Log im Repo unter
`logs/lapv1/`) zeigt folgende Eigenschaften, die Schritt 0 reproduzieren
und beheben muss:

- Epoch-by-Epoch joint-val_top1: 0,9045 → **0,9076** → 0,8060 →
  0,8050 → (abgebrochen).
- selection-val_top1 im `freeze_inner_hard`-Regime: 0,796–0,797.
- rollbacks pro Epoch: 0 → 0 → 100 → 6302 → (abgebrochen).
- `root_incorrect_gain`: 0,0070 → 0,1177 → 0,0122 → 0,0163 →
  (abgebrochen).
- Arena-Block fehlt komplett.
- {!!Anmerkung: Zum Zeitpunkt der Überarbeitung war der v4-Lauf noch nicht durch}

**Aufgaben:**

1. **v4-Lauf analysieren und dokumentieren.** Lege
   `reports/lapv1/v4_analysis.md` an, die genau diese Zahlen aus dem
   Logfile extrahiert und die drei in `LAPv2_Paper.md` §15 benannten
   Befunde tabelliert. Kein Training in diesem Unterschritt.

2. **Baseline-Report erzeugen.** Ablage nach
   `reports/lapv1/stage2_baseline_v5/report.md` mit:
   - Tabelle der Metriken pro Epoch wie oben.
   - Arena-Elo pro Budget, SPRT/Binomial-CI.
   - selection-Elo separat.
   - Plot `val_top1_over_epochs.png` (epoch auf x, joint/val/selection
     auf y).
   - Liste der drei Checkpoint-Dateien mit SHA-256.

3. **Report für finale Überarbeitung des Plans(Schritte 1–12 ab Zeile ) nutzen**
   - Überarbeite den folgenden Plan noch einmal final, anhand der Ergebnisse und Erkenntnisse
   - Starte den Plan erst, wenn die Analyse vollständig und ausgewertet wurde
   - commit + push, wenn der Plan dann final überarbeitet ist, danach kann mit der schrittweise Implementierung begonnen werden.
   
**Acceptance:**

- `models/lapv1/baseline_v5/freeze_inner_hard_best.pt` existiert und
  hat joint-`val_top1 ≥ 0,905` auf der Referenz-Validation.
- `models/lapv1/baseline_v5/joint_heads_mix_best.pt` existiert.
- `models/lapv1/baseline_v5/joint_backbone_mix_best.pt` existiert.
- `reports/lapv1/stage2_baseline_v5/report.md` enthält Arena-Elo für
  alle vier Budgets.
- `reports/lapv1/v4_analysis.md` existiert und benennt die drei
  Phänomene aus §15 des Papers explizit.
- Changelog-Eintrag in `docs/lapv2/CHANGELOG.md` unter
  `## Schritt 0`.

**Rollback:** Dateien und Branch löschen; keine Auswirkungen auf den
Hauptcode.

**Hinweis zum weiteren Plan:** Der in `configs/lapv2/base.yaml`
verwendete Warm-Start-Checkpoint ist ab jetzt
`models/lapv1/baseline_v5/freeze_inner_hard_best.pt`. Begründung:
Abschnitt 15.4 des Papers; der spätere `joint_heads_mix_best` erbt
bereits die 10-pp-Regression, der `joint_backbone_mix_best` ebenfalls.

---

## Schritte 1–12 — unverändert zu Rev 3

Die Detailbeschreibung bleibt exakt wie in `LAPv2_Codex_Plan.md` Rev 3
(Original-Dokument im Repo unter `docs/lapv2/plan_rev3.md` ablegen). Im
Codex-Prompt wird pro Aufgabe nur der jeweilige Schritt referenziert.
Die einzige Änderung ist die Querschnittsregel Regression-Watchdog,
die automatisch an jedem Flag-Flip ab Schritt 4 angewandt wird.

Kurzzusammenfassung zur Navigation:

- **Schritt 1** — Repo-Audit, `docs/lapv2/path_map.md` anlegen.
- **Schritt 2** — Artefakt-Erweiterung um `phase_index`, `king_sq_*`,
  `nnue_feat_*`, Move-Deltas (K = 64 Kandidaten pro Sample); Loader
  abwärtskompatibel; voller Testsatz inklusive
  `test_move_delta_consistency_with_full_rebuild`.
- **Schritt 3** — `PhaseRouter` und generischer `PhaseMoE`-Wrapper mit
  `from_single`; keine Top-Level-Integration.
- **Schritt 4** — `FeatureTransformer` (EmbeddingBag),
  `DualAccumulatorBuilder`, `IncrementalAccumulator`;
  Haupt-Integritätstest `test_incremental_apply_move_matches_full_rebuild`.
- **Schritt 5** — `PhaseMoE.from_single` für `intention_encoder` und
  `state_embedder`.
- **Schritt 6** — `NNUEValueHead` auf single-phase FT (self.ft wird
  top-level Attribut für spätere Policy-Wiederverwendung).
- **Schritt 7** — FT und Value-Head als Phase-MoE.
- **Schritt 8** — `NNUEPolicyHead` auf demselben FT,
  Successor-Accumulator im Trainingspfad, King-Move via
  `_after_move`-Liste (Option A), Loss-Balance Value/Policy.
  `test_value_and_policy_share_ft_object` ist harter Guard.
- **Schritt 9** — Sharpness-Head als Phase-MoE (nicht NNUE-artig).
- **Schritt 10** — `OpponentReadout` mit Δ-Operator.
- **Schritt 11** — optionaler Distillations-Loss.
- **Schritt 12** — Phase-Fixierung + `AccumulatorCache` in der
  Deliberation-Loop, bitgleich zum No-Cache-Pfad.

Die vollständigen Aufgaben, Code-Skelette, Tests, Acceptance- und
Rollback-Regeln dieser Schritte sind gegenüber Rev 3 unverändert und
werden beim Codex-Einsatz über die unveränderte Rev-3-Datei
referenziert. Der Watchdog gilt ab Schritt 4 für jeden Flag-Flip.

---

## Schritt 13 — T1/T2 Pipeline (präzisiert in Rev 3.1)

**Ziel:** Trainingsinfrastruktur auf LAPv2 anpassen, mit korrigiertem
Warm-Start und obligatorischem Watchdog.

**Änderungen gegenüber Rev 3:**

1. **Warm-Start-Loader liest aus der v5-Baseline, nicht aus einem
   "letzten" LAPv1-Checkpoint.**

   ```python
   # engine/training/warm_start.py
   def build_lapv2_warm_start(
       src_path: Path = Path("models/lapv1/baseline_v5/"
                             "freeze_inner_hard_best.pt"),
       dst_path: Path = Path("models/lapv2/warm_start_from_v5.pt"),
   ) -> None:
       """
       Erzeugt einen LAPv2-Init-Checkpoint aus dem v5-Baseline.
       Phase-MoE-Module: 4 identische Kopien (from_single).
       FT (PhaseMoE):   frisch initialisiert std = 1/sqrt(N_accumulator).
       NNUEValueHead:   frisch initialisiert.
       NNUEPolicyHead:  frisch initialisiert.
       OpponentReadout: frisch initialisiert.
       Alles andere: 1:1 übernommen.
       Schreibt lapv2_version, source_checkpoint_sha256, warm_start_epoch
       in die Metadaten.
       """
   ```

   Harter Guard im Loader: wenn das Quelldokument als Stage-2-Phase
   nicht `freeze_inner_hard` angibt oder joint-`val_top1` < 0,90
   protokolliert, **Exception mit expliziter Fehlermeldung**, die auf
   Paper §15.4 und Schritt 0 verweist.

2. **Load-Balancing-Regularisierer** wie in Rev 3 spezifiziert.

3. **Zweistufiges Phase-Curriculum-Gate** wie in Rev 3 spezifiziert.

4. **Adaptive Loss-Balance Value/Policy** wie in Schritt 8.5
   spezifiziert.

5. **Watchdog obligatorisch an Stufe A → Stufe B.** Die zweite
   Gate-Stufe (NNUE-Heads phase-spezifisch) ist strukturell identisch
   zu einem Head-Unfreeze. Ohne Watchdog droht hier exakt die
   Regression aus dem v4-Lauf. Der Watchdog läuft im
   `rollback_and_lower_lr`-Modus, nicht nur im Warn-Modus.

6. **Logging pro Epoch:**
   - Expertenauslastung pro Phase
   - Value-Loss und Policy-Loss getrennt pro Phase
   - Norm-Abstand der 4 FT-Experten (FT-Drift)
   - Cosine-Distanz `V_adapter` ↔ `P_adapter`
   - Reply-Consistency (Korrelation Readout vs. Lehrer, falls Distill an)
   - Watchdog-Events (JSONL)

7. **Tests:**
   - `test_warm_start_loads_from_freeze_inner_hard_only()` — Quelle mit
     falscher Stage-2-Phase wird abgelehnt.
   - `test_warm_start_checkpoint_forward_matches_lapv1()` für die
     Komponenten, die NICHT frisch initialisiert werden.
   - `test_load_balancing_weights_in_range()`
   - `test_phase_gate_pulls_experts_in_stage_a()`
   - `test_phase_gate_releases_in_stage_b()`
   - `test_stage_b_transition_guarded_by_watchdog()` — verifiziert, dass
     der Übergang Stufe A → Stufe B den Watchdog aufruft und bei
     injizierter Regression zurückrollt.

**Acceptance:**

- Voller T1-Run (mehrere hundert Steps) mit allen Flags an läuft
  stabil, Loss fällt, keine NaN, alle Logs vorhanden.
- Watchdog protokolliert Stufe A → Stufe B als akzeptiert ODER als
  kontrolliert zurückgerollt (nicht als stumm verschluckte Regression).
- joint-`val_top1` nach Stufe-A-Initialisierung ist bitgleich zum
  Warm-Start-Checkpoint (weil `from_single` und frischer FT strukturell
  LAPv1-äquivalent sind, solange NNUE-Flag off ist — dies ist Teil des
  Tests).

**Rollback:** Flags abschalten, Trainer fällt auf LAPv1-Pfad zurück.

---

## Schritt 14 — Arena-Harness, Reporting, Release-Gate (präzisiert in Rev 3.1)

**Ziel:** Fair reproduzierbarer Vergleich LAPv1 (Baseline v5) vs. LAPv2.

**Aufgaben:**

1. Arena-Runner um LAPv2-Checkpoint-Typ erweitern.

2. Vergleichsprotokoll erzeugt Report mit:
   - Elo-Differenz pro Budget (`inner0`, `inner1`, `inner2`, `auto4`)
     auf der joint Thor-150-Suite.
   - **(neu) Elo-Differenz pro Budget auf der selection/hard-Suite.**
     Begründung: §15.3 Befund B und §13.1 letzter Punkt.
   - Signifikanz via SPRT oder Binomial-CIs.
   - Phase-Perplexity, FT-Drift, Reply-Consistency.
   - Policy-Value-Konsistenz: Korrelation zwischen NNUE-Policy-Logit
     eines Zugs und negiertem NNUE-Value des Successors.
   - Rollback-Rate pro Epoch und `root_incorrect_gain` aus dem
     LAPv2-T2-Lauf, gegenübergestellt zu den v5-Baseline-Zahlen.

3. **Release-Gate (Rev 3.1).** LAPv2-Checkpoint ist release-würdig,
   wenn **alle** folgenden Bedingungen erfüllt sind:
   - Joint-Elo: in mindestens drei der vier Budgets kein signifikant
     schlechterer Elo gegenüber LAPv1-v5, und in mindestens einem
     Budget signifikant besser.
   - Selection-Elo: in allen vier Budgets kein signifikant schlechterer
     Elo, und in mindestens einem Budget signifikant besser. Der
     selection-Gewinn ist die eigentliche LAPv2-These; ohne ihn ist
     der Architekturumbau nicht gerechtfertigt.
   - joint-`val_top1` mindestens gleichauf mit
     `freeze_inner_hard_best` aus der v5-Baseline, minus ε = 0,01.
   - `root_incorrect_gain` mindestens halb so groß wie im besten v5-
     `freeze_inner_hard`-Epoch (0,1177). Fällt dieser Wert unter
     0,05, ist die Inner-Loop-Korrekturkraft kollabiert — kein
     Release.
   - Watchdog-Log zeigt keine stumm verschluckte Regression.

4. Reports nach `reports/lapv2/run_<date>/`.

**Acceptance:** Arena-Runner läuft auf der vollen Thor-150 mit beiden
Checkpoints und erzeugt den vollen Report (joint + selection, vier
Budgets) ohne Fehler. Release-Gate-Check ist im Report ein eigener
Abschnitt mit boolescher Summe und Begründung pro Kriterium.

**Rollback:** Arena-Runner hat LAPv1-v5 als Default.

---

## Anhang A — Codex-CLI Task-Vorlage

```
Lies docs/lapv2/plan.md (Rev 3.1) und führe ausschließlich Schritt <N> aus.
Arbeite auf Branch lapv2/step-<N>-<slug>.
Halte dich exakt an Acceptance Criterion und Rollback-Regel.
Wenn der Schritt ab Schritt 4 liegt, ist der Regression-Watchdog
Pflicht an jedem Flag-Flip (siehe Querschnittsregel).
Wenn du Unklarheiten findest, aktualisiere docs/lapv2/path_map.md
bevor du Code schreibst.
Bevor du Schritt 1..14 startest, verifiziere per ls, dass
models/lapv1/baseline_v5/freeze_inner_hard_best.pt existiert.
Wenn nicht: brich ab und verweise auf Schritt 0.
Am Ende: Tests, Lint, Changelog-Eintrag, PR-Beschreibung mit
Acceptance-Check-Liste und (ab Schritt 4) Watchdog-Log-Auszug.
```

## Anhang B — Abhängigkeitsgraph

```
  0 ──> 1 ──> 2 ──> 3 ──> 4 ──┐
                              ├─> 6 ──> 7 ──┐
                              │             ├─> 12 ──> 13 ──> 14
                              └─> 8 ────────┤
                        3 ──> 5 ────────────┤
                                        9 ──┤
                                       10 ──┤
                                       11 ──┘
```

- **Schritt 0** ist die neue Wurzel: ohne reproduzierte LAPv1-Baseline
  startet Schritt 1 nicht.
- Schritt 4 (Dual-Accumulator) ist die Basis für Value und Policy NNUE.
- Schritt 6 (NNUE-Value single) muss vor Schritt 7 (Value Phase-MoE)
  und vor Schritt 8 (Policy NNUE) kommen, weil 6 das `self.ft`-Attribut
  einführt, das 8 wiederverwendet.
- Schritt 8 (NNUE-Policy) hängt von 4, 6 ab.
- Schritte 9, 10, 11 sind orthogonale Spuren nach Schritt 5.
- Schritt 12 hängt von 4, 6, 7, 8 ab.
- Schritt 13 fasst alles zusammen und aktiviert den
  Regression-Watchdog im `rollback_and_lower_lr`-Modus.
- Schritt 14 misst gegen die v5-Baseline aus Schritt 0.

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
- `test_value_and_policy_share_ft_object` schlägt fehl — zentrale
  Architektur-Invariante.
- `test_incremental_apply_move_matches_full_rebuild` schlägt fehl —
  zweite zentrale Invariante.
- **(neu, Rev 3.1)** Der Regression-Watchdog rollt einen Schritt
  dreimal in Folge zurück — dann liegt ein struktureller Regime-Switch
  vor, der Plan muss überarbeitet werden.
- **(neu, Rev 3.1)** Schritt 0 kann die v4-Regression (joint-`val_top1`
  0,9076 → 0,8060 am Stage-2-Phasenübergang) nicht im v5-Baseline-Lauf
  reproduzieren. Dann ist die Ursache nicht verstanden und der Plan
  greift an der falschen Stelle.

## Anhang D — Zukünftige Vereinfachung (nicht in diesem Plan)

Schritt 8 implementiert die eigenständige NNUE-Policy mit eigenem
Adapter, MoveHead und Move-Type-Embedding. Eine spätere, radikalere
Variante wäre die volle Value-Policy-Unifikation
`logit(m) = −NNUEValueHead(a_succ_other, a_succ_stm)`. Diese
Vereinfachung wird explizit nicht in diesem Plan verfolgt und ist
Gegenstand eines möglichen Rev-4-Plans.

## Anhang E — Referenzen im Repo

Dateien, die Codex als erste liest, in exakt dieser Reihenfolge:

1. `docs/lapv2/LAPv2_Paper.md` (Rev 3.1) — Architektur, insbesondere §9
   und §15.
2. `docs/lapv2/LAPv2_Codex_Plan.md` (diese Datei, Rev 3.1) — dieser
   Plan.
3. `docs/lapv2/plan_rev3.md` — Detailtext der Schritte 1–12 (Rev 3),
   unverändert.
4. `reports/lapv1/v4_analysis.md` — liegt nach Schritt 0 vor.
5. `reports/lapv1/stage2_baseline_v5/report.md` — liegt nach Schritt 0
   vor und ist Messlatte für Schritt 14.
