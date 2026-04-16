# ExecPlan: Frontier-Deliberation im Netz statt klassischer Suche

## Status

Stand `2026-04-16`:

- ExecPlan dokumentiert
- Implementierung dieses Arbeitsgangs startet mit einem kleinen, messbaren
  Ausbau des bestehenden LAPv1/LAPv2-Deliberation-Loops
- Rust-UCI-Weiterbau ist bewusst gestoppt; Fokus liegt auf dem trainierbaren
  Netzpfad

## Ziel

Die bestehende bounded recurrent deliberation soll nicht durch klassische Suche
ersetzt werden, sondern innerhalb des Netzes strukturierter werden.

Der erste Schritt bleibt absichtlich eng:

- der existierende Top-K-Refinement-Pfad wird explizit als kleine Frontier
  behandelt
- die Frontier-Selektion nutzt Unsicherheit und Frontier-Historie statt nur
  rohe Kandidatenscores
- Training und Evaluation bekommen sichtbare Frontier-Metriken
- die neuen Signale bleiben vollständig innerhalb des bestehenden
  LAPv1/LAPv2-Trainingspfads sichtbar

## Nicht-Ziele

- kein Alpha-Beta, Negamax, PVS, TT, MCTS oder PUCT
- keine Brettbaum-Expansion
- keine verdeckte symbolische Suchheuristik als Runtime- oder Trainingsabkürzung
- keine vollständige Mehrpfad-Latentdynamik mit eigenem Zustand pro Kandidat in
  diesem Schritt
- keine Änderung des laufenden Trainingsclusters

## Architekturvertrag

### 1. Frontier statt Einspur-Deliberation

Der vorhandene Deliberation-Loop refinert bereits mehrere Root-Kandidaten pro
Schritt. Dieser Schritt macht den Vertrag explizit:

- `top_k_refine` ist die aktive Frontier-Größe
- die Frontier ist bounded und root-only
- Kandidaten dürfen revisitet oder fallengelassen werden
- Rollback bleibt latent und root-only

Es entsteht kein Suchbaum.

### 2. Uncertainty steuert Revisit vs. Novelty

Die Frontier-Selektion soll bei niedriger Unsicherheit eher stabil bleiben und
bereits gute Kandidaten weiterverfolgen.

Bei hoher Unsicherheit soll sie eher weniger oft besuchte Kandidaten wieder in
die Frontier ziehen.

Das ist bewusst MCTS-inspiriert, aber nicht MCTS:

- keine Baumstatistik
- keine Visit-Backups
- keine UCT-Formel
- nur bounded Kandidatenpriorisierung am Root

### 3. Erfolg wird über Trainingsmetriken sichtbar gemacht

Die Änderung gilt erst dann als sinnvoll, wenn sie im normalen
LAPv1/LAPv2-Training sichtbar wird.

Neue Metriken müssen deshalb mindestens erfassen:

- Frontier-Turnover zwischen Schritten
- Revisit-Rate
- Frontier-Stabilität
- Unique-Candidate-Coverage
- wie oft der finale Gewinner schon in der Frontier lag

Diese Kennzahlen sollen direkt in den Trainings- und Validation-Summaries
landen.

## Tieferer Folgepfad

Dieser erste Schritt ist nur die kleinste saubere Stufe. Der eigentliche
Zielpfad bleibt:

1. kleine Frontier über `top_k` Kandidaten
2. Unsicherheits-gesteuerte Revisit-/Drop-Logik
3. opponent-conditioned Verfeinerung pro Kandidat
4. globales Frontier-Memory
5. später optional echte Mehrpfad-Latenzen mit Rollback auf Frontier-Ebene

Das sind brauchbare MCTS-Prinzipien ohne Suchbaum.

## Implementierungsschritte

1. ExecPlan dokumentieren.
2. `CandidateSelector` uncertainty-/history-aware machen.
3. Frontier-Historie im Deliberation-Loop mitführen:
   - vorige Frontier
   - Besuchszähler
   - Frontier-Wechsel
4. Echte per-candidate Frontier-Zustände und Frontier-Memory ergänzen:
   - kandidatenspezifische Latents
   - kandidatenspezifische Memory-Summaries
   - additive Frontier-Delta-Updates ohne Suchbaum
5. Frontier-Metriken im Modellvertrag und Trainer auswertbar machen.
6. Validation-/Train-Logs um Frontier-Kennzahlen ergänzen.
7. Tests für Selector, Loop, Modellvertrag und Trainer-Metriken ergänzen.
8. Architektur-Doku für LAPv1 nachziehen.

## Tests

- `python3 -m pytest python/tests/test_deliberation_loop.py`
- `python3 -m pytest python/tests/test_lapv1_model.py`
- `python3 -m pytest python/tests/test_lapv1_trainer.py python/tests/test_lapv1_trainer_stage2.py`
- `python3 -m ruff check python`

## Exit-Kriterien

- kein klassischer Suchmechanismus wurde eingeführt
- der Loop nutzt Frontier-Historie sichtbar im Code
- Frontier-Metriken erscheinen in Training und Validation
- die Änderungen sind durch isolierte Tests abgesichert
- Doku beschreibt klar, was jetzt implementiert ist und was noch aussteht

## Risiken

- zu aggressive Frontier-Novelty kann die guten Root-Priors zerstören
- zu konservative Revisit-Boni bringen kaum Veränderung gegenüber dem Ist-Zustand
- Frontier-Metriken können gut aussehen, ohne Arena-Stärke zu verbessern
- echte Mehrpfad-Latentzustände bleiben weiterhin ein separater, größerer
  Architektur-Schritt
