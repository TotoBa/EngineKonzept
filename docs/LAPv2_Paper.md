# LAPv2: Phasen-MoE und NNUE-artige Heads
## Architekturerweiterung der LAPv1-Schachengine (EngineKonzept) — Revision 3.1

> **Rev 3.1** gegenüber Rev 3: Unverändert in Kernarchitektur. Präzisiert im
> Abschnitt 9 (Trainingspipeline) und um Abschnitt 15 erweitert, das den
> Stage-2-Lauf `stage2_fast_arena_all_unique_v4` auswertet. Der Befund
> aus diesem Lauf macht den Warm-Start-Checkpoint und einen
> Regression-Watchdog explizit — beides hätte in Rev 3 implizit bleiben
> können und hätte dort mit hoher Wahrscheinlichkeit eine wiederholbare
> 10-Prozentpunkt-Regression beim Aktivieren der NNUE-Heads erzeugt.

## Zusammenfassung

LAPv2 erweitert LAPv1 um drei zusammenhängende Änderungen. Erstens wird ein
hartes Mixture-of-Experts mit vier Experten pro internem Modul eingeführt,
das entlang einer schachlich kalibrierten Spielphasenachse routet. Jeder
Experte hat exakt die Breite des entsprechenden LAPv1-Moduls. Zweitens
werden Value-Head und Policy-Head als NNUE-artige Netze umgebaut: pro
Phase liefert ein königsfeldindizierter Feature-Transformer (FT) einen
inkrementell pflegbaren Accumulator, und beide Heads lesen aus demselben
FT — der Value-Head als Skalarbewertung des aktuellen Zustands, der
Policy-Head als differentielle Bewertung der Successor-Accumulator pro
Kandidatenzug. Drittens wird der eigenständige OpponentHead durch einen
kleinen Opponent-Readout auf dem geteilten Backbone ersetzt, optional mit
Distillation aus vollem Rekursionsaufruf im Training. Die Erweiterung
bleibt mit Deliberation-Loop, Phase-5-all-unique-Korpus und der
T1/T2-Pipeline kompatibel. Das Gesamtparameterbudget sinkt gegenüber
Rev 2 von ~71 M auf ~58 M, weil der dominante FT zwischen Value und
Policy geteilt wird.

---

## 1. Ausgangspunkt: LAPv1

LAPv1 ist eine symbolisch-reichhaltige, bewertungsgetriebene
Deliberation-Architektur. Die Eingabe besteht aus Piece-Tokens,
Square-Tokens, State-Context, Reachability-Edges und vorkompilierten
Teacher-Targets. Der Kern besteht aus Intention-Encoder und
State-Embedder, gefolgt von vier Köpfen: Policy, Value, Sharpness,
Opponent. Darüber läuft eine iterative Deliberation-Loop mit
Candidate-Scoring, Top-k-Refinement, Value-/Sharpness-Bewertung,
Stabilitätsprüfung und optionalem Rollback, begrenzt durch Budget oder
Early-Stop.

Trainiert wird zweistufig: Stage T1 als CPU-feasibler Bootstrap mit
19,8 M Parametern, danach Stage T2 als Warm-Start, das die Deliberation
explizit mit einem Inner-Step-Curriculum 1 → 2 → 4 trainiert. Reine
Laufzeit-Deliberation ohne passendes Training operiert
out-of-distribution; die Trainierbarkeit der inneren Schritte ist daher
zentral.

## 2. Problemanalyse

### 2.1 Phasenunabhängige Gewichte sind ein Kompromiss

In LAPv1 teilen sich alle Module ihre Gewichte über alle Stellungen.
Schach zerfällt jedoch empirisch in Regime mit sehr verschiedenen
dominanten Heuristiken. Ein einzelnes Gewichtsbündel muss alle Regime
simultan modellieren, was Interferenz erzeugt und Value- und
Policy-Signale in Übergangsstellungen unscharf macht.

### 2.2 Value- und Policy-Head ohne NNUE-Struktur verschenken CPU-Stärke

Die im klassischen Schach-Engine-Umfeld dominante CPU-Bewertungs-
architektur ist NNUE (Efficiently Updatable Neural Network, seit
Stockfish NNUE state of the art). Ihre Stärke beruht auf drei
strukturellen Eigenschaften: einem königsfeldindizierten
Feature-Transformer als erster Schicht, einem inkrementell pflegbaren
Accumulator über sparsame Piece-Square-Features, und einem flachen
dichten Kopf mit clipped-ReLU. Ein klassischer dichter Value-MLP
verschenkt sowohl die Königsfeldkonditionierung als auch die
inkrementelle Auswertbarkeit. Klassische Policy-Heads verschenken
zusätzlich, dass ein Zug strukturell nichts anderes ist als eine kleine,
sparsame Veränderung der Featuremenge.

### 2.3 OpponentHead ist semantisch inkohärent

Der aktuelle OpponentHead ist ein eigenständiges Subnetz mit eigenem
Repräsentationsraum, billig aber ohne Gewichtsteilung mit dem Hauptnetz.
Ein voller Rekursionsaufruf wäre semantisch konsistent, aber pro Top-k-
Kandidat in jeder inneren Iteration zu teuer. Ein kleiner
Opponent-Readout auf dem geteilten Backbone ist der tragfähige
Mittelweg.

### 2.4 (neu in 3.1) Empirisch beobachtete Regression am Phasenübergang

Der Stage-2-Lauf `stage2_fast_arena_all_unique_v4` (siehe Abschnitt 15)
liefert einen bis Rev 3 nicht dokumentierten, aber für LAPv2
sicherheitsrelevanten Befund: Der Übergang vom
`freeze_inner_hard`-Regime in das `joint_heads_mix`-Regime reduziert die
Validation-Top1-Accuracy um rund 10 Prozentpunkte (0,9076 → 0,8060) und
bricht gleichzeitig die korrektive Kraft der Deliberation-Loop ein
(root_incorrect_gain fällt von 0,118 auf 0,012). Das ist relevant für
LAPv2, weil Schritt 13 des Migrationsplans genau diese Choreographie
noch einmal spielt — nur zusätzlich mit frisch initialisierten
FT-Matrizen und zwei neuen Heads (NNUEValueHead, NNUEPolicyHead) am
geteilten FT. Ohne explizite Gegenmaßnahme würde LAPv2 mit hoher
Wahrscheinlichkeit eine ähnliche oder schlechtere Regression zeigen, die
nicht der LAPv2-Architektur anzulasten wäre, sondern dem
Trainingsregime-Übergang.

## 3. Architekturübersicht LAPv2

LAPv2 behält Datenpfad, Deliberation-Loop und T1/T2-Pipeline bei.
Verändert werden die inneren Module, die Head-Struktur und der
Gegner-Zweig. Formal:

```
f_LAPv1(x) = Heads( Deliberation( Embed( x ) ) )
f_LAPv2(x) = Heads_NNUE( Deliberation( Embed_PhaseMoE( x ) ),
                         r_phase(x), AccPair(x) )
```

`r_phase` ist ein deterministischer, regelbasierter Router ohne
Gradienten, der eine Stellung auf einen von vier Phasenindizes abbildet.
`AccPair(x)` ist das Paar `(a_white, a_black)` der pro Königsperspektive
berechneten NNUE-Accumulator. Das Phase-Routing ist hart: pro
Forward-Pass wird genau ein Experte aktiviert.

### 3.1 Komponenten im Überblick

| Ebene | LAPv1 | LAPv2 |
|---|---|---|
| Intention-Encoder | einfach | PhaseMoE (4 Experten) |
| State-Embedder | einfach | PhaseMoE (4 Experten) |
| FT (Feature-Transformer) | — | PhaseMoE, shared Value+Policy |
| Value-Head | dense MLP | NNUEValueHead auf FT |
| Policy-Head | dense | NNUEPolicyHead auf demselben FT (Successor-Scoring) |
| Sharpness-Head | dense | PhaseMoE, nicht NNUE |
| Opponent-Head | eigenständiges Subnetz | Δ-Readout auf shared Backbone |
| Deliberation-Loop | unverändert | AccumulatorCache, phase-fixiert |

## 4. Phase-MoE mit vollwertigen Experten

### 4.1 Kalibrierte Phasendefinition

Die Phase wird nicht per Halbzugzähler bestimmt, sondern als
deterministische Funktion des Rohzustands. Als Grundlage dient ein
Stockfish-artiger Phase-Score, der das nicht-Bauern-Material auf dem
Brett zählt. Diese Metrik ist in klassischen Engines seit langem
kalibriert und liefert ein robustes, monoton abnehmendes Signal vom
vollen Brett bis zum Endspiel.

```python
# Stockfish-artige Phase-Gewichte
def phase_score(pos):
    s  = 1 * count(pos, ['N','n','B','b'])
    s += 2 * count(pos, ['R','r'])
    s += 4 * count(pos, ['Q','q'])
    return s                               # Startposition: 24
```

Der rohe Score wird mit drei strukturellen Zusatzsignalen kombiniert,
die jeweils bekannte Engine-Heuristiken spiegeln: Damenpräsenz
(Hauptindikator für Mittelspiel vs. Endspiel), Entwicklungsgrad
(Minor-Figuren noch auf dem Heimatfeld als Indikator Eröffnung) und
Bauernstruktur (blockierte Zentralbauern als Indikator für spätes
Mittelspiel).

```python
def phase_index(pos):
    s            = phase_score(pos)             # 0..24
    queens       = count(pos, ['Q','q'])        # 0..2
    minors_home  = minors_on_home_rank(pos)     # 0..8
    blocked_pawn = central_blocked_pawns(pos)   # 0..4
    # Eröffnung: viel Material UND wenig entwickelt
    if s >= 22 and minors_home >= 5:
        return 0
    # Endspiel: harte Materialschwelle oder Damen weg und wenig Material
    if s <= 8 or (queens == 0 and s <= 12):
        return 3
    # spätes Mittelspiel: mittleres Material oder verkeilte Struktur
    if s <= 14 or blocked_pawn >= 3:
        return 2
    # frühes Mittelspiel als Default
    return 1
```

Die Reihenfolge der Tests ist bewusst: Eröffnung zuerst, Endspiel
zweitens, spätes Mittelspiel drittens, frühes Mittelspiel als Default.
Die Schwellen sind Stockfish-nah und werden einmalig während T2
kalibriert.

### 4.2 Expertenstruktur und Größe

Jedes phasenabhängige Modul `M` wird durch eine Expertengruppe
`{M_0, M_1, M_2, M_3}` ersetzt, in der jeder Experte exakt die volle
Breite des ursprünglichen LAPv1-Moduls besitzt. Hartes Routing aktiviert
pro Forward-Pass genau einen Experten:

```
M_phase(x) = M_{ r_phase(x) }(x)
```

Die Experten starten identisch initialisiert aus dem Stage-T2-Checkpoint
von LAPv1. In den ersten T2-Iterationen verhält sich LAPv2 damit
numerisch wie LAPv1; die Experten divergieren erst durch phasenspezifische
Gradienten, was den Warm-Start stabilisiert und MoE-Kaltstartkollaps
vermeidet.

## 5. Dual-Accumulator und gemeinsamer Feature-Transformer

### 5.1 Motivation

Der zentrale Baustein der NNUE-artigen Heads von LAPv2 ist ein pro
Phase trainierter Feature-Transformer `FT_phase`, dessen Ausgabe — der
Accumulator — von zwei verschiedenen Köpfen (Value und Policy) genutzt
wird. Statt den FT in jedem Head zu duplizieren, wird er einmal pro
Phase angelegt und über alle Heads geteilt. Das ist sowohl semantisch
sauber (eine einzige Repräsentation der Stellung) als auch der
entscheidende Hebel für die Parameterökonomie.

### 5.2 Feature-Definition (HalfKA)

Ein Feature ist das Tupel `(own_king_square, piece_square,
piece_type_with_color)`. Über 64 Königsfelder × 64 Figurenfelder × 12
Piece-Type-Klassen ergibt sich ein Featureraum von höchstens 49 152
Indizes pro Königsperspektive, wovon pro Stellung nur etwa 20 bis 30
aktiv sind. Der eigene König ist nicht selbst Feature, sondern
definiert den Index. Die Sparsity ist hier der Punkt: die erste Schicht
ist effektiv eine indizierte Summierung, keine dichte
Matrixmultiplikation.

### 5.3 Dual-Accumulator pro Phase

Pro Stellung werden zwei Accumulator parallel gepflegt: `a_white`
(HalfKA aus Sicht des weißen Königs) und `a_black` (HalfKA aus Sicht des
schwarzen Königs). Beide haben Dimension N = 64 und werden aus
demselben `FT_phase` berechnet, jedoch über disjunkte Index-Slices.

```
a_white = Σ FT_phase[i]   für aktive Indizes i ∈ HalfKA(K_white, ·)
a_black = Σ FT_phase[i]   für aktive Indizes i ∈ HalfKA(K_black, ·)
```

Beide Accumulator werden inkrementell aktualisiert: bei einem Zug
verschwindet ein Feature, bei Schlagen ein zweites, ein neues entsteht
— das sind in der Regel zwei bis drei Indizes pro Accumulator. Nur bei
einem Königszug oder einer Rochade desjenigen Königs, dessen Index
gerade aktiv ist, muss der entsprechende Accumulator komplett neu
berechnet werden.

### 5.4 Dimensionierung und Parameterkonsequenz

Die Wahl N = 64 ist bewusst klein gehalten — Stockfish verwendet 1024,
doch dort ist das Ziel Millionen Knoten pro Sekunde. LAPv2 zielt auf
bewertungsgetriebene Deliberation, nicht auf maximale NPS, und hat
zudem oberhalb des FT noch das volle LAPv1-Backbone
(Intention-Encoder, State-Embedder), das Repräsentationskapazität
liefert. Pro `FT_phase` ergibt sich damit:

```
|FT_phase| = 49 152 · 64 ≈ 3,15 M Parameter
```

Über vier Phasen sind das ca. 12,6 M Parameter — und dieser Block trägt
nun BEIDE Heads (Value und Policy). Die Erkenntnis: der
Hauptkostenfaktor jedes NNUE-artigen Netzes ist der FT, und die
Parametereinsparung gegenüber zwei separaten FTs ist beträchtlich.

## 6. Value-Head als NNUE-artiges Netz

### 6.1 Architektur pro Phase

Der Value-Head pro Phase liest aus dem Dual-Accumulator und besteht aus
drei kleinen Komponenten:

- **Value-Adapter `V_adapter_phase`**: lineare Projektion `R^N → R^N`,
  eine pro Phase. Erlaubt dem Value-Pfad, eine eigene Sicht auf den
  geteilten FT-Latent zu lernen.
- **Side-aware Konkatenation**: das Paar
  `(ClippedReLU(a_stm), ClippedReLU(a_other))` wird in dieser
  Reihenfolge verkettet, wobei stm = side to move.
- **Dichter Value-Kopf `V_dense_phase`**: zwei Linearschichten mit
  Hidden 32 und Clipped-ReLU, abschließende skalare Ausgabe als
  Centipawn-Wert, umskaliert auf den LAPv1-Value-Bereich.

```
v(s) = V_dense_phase( [ClippedReLU(V_adapter(a_stm));
                       ClippedReLU(V_adapter(a_other))] )
```

Die Königsfeldkonditionierung ist intrinsisch: durch HalfKA bestimmt
das Königsfeld der jeweiligen Seite automatisch, welche Slices des FT
angesprochen werden. Es gibt keine 64 separaten König-Experten, sondern
64 disjunkte Index-Slices pro `FT_phase` — die sparsamste mögliche
Realisierung.

### 6.2 Inkrementelle Auswertbarkeit

Weil die erste Schicht eine Summe über wenige Featureindizes ist, kann
der Accumulator bei einem Zug differenziell aktualisiert werden. Für
die Deliberation-Loop ist diese Inkrementalität direkt nutzbar: die
Top-k-Kandidaten-Nachfolgestellungen unterscheiden sich jeweils nur
durch einen Zug vom Wurzelzustand, und der Accumulator kann pro
Kandidat in konstanter Zeit bezüglich der geänderten Features
aktualisiert werden. Königszüge und Rochaden werden per Dirty-Flag und
lazy Rebuild behandelt.

## 7. Policy-Head als NNUE-artiges Netz

### 7.1 Kernidee: ein Zug ist ein sparsamer Feature-Delta

Die zentrale Beobachtung ist, dass ein Schachzug strukturell nichts
anderes ist als eine kleine, sparsame Veränderung der
HalfKA-Featuremenge: ein bis zwei Indizes verschwinden (Quellfeld der
ziehenden Figur, gegebenenfalls Schlagopferindex), ein bis zwei Indizes
erscheinen (Zielfeld, gegebenenfalls Promotionsfigur). Damit lässt sich
pro Zug ein Delta-Accumulator in O(1) berechnen, und der
Successor-Accumulator des Zugs ist einfach `a_root + delta`. Der
NNUE-artige Policy-Head bewertet einen Zug, indem er diesen
Successor-Accumulator durch einen kleinen dichten Kopf laufen lässt.

Konzeptionell ist das eine Art Static-Successor-Scoring: jeder
Kandidatenzug erzeugt einen virtuellen Successor-Latent ohne
tatsächlichen Brett-Update, und der Policy-Score ist eine differenzielle
Bewertung dieses Latents gegenüber dem Wurzel-Latent.

### 7.2 Architektur pro Phase

Pro Phase besteht der Policy-Head aus folgenden Komponenten:

- **Policy-Adapter `P_adapter_phase`**: lineare Projektion `R^N → R^N`,
  analog zum Value-Adapter, eigenständig pro Phase.
- **Move-Type-Embedding `move_type_embed`**: ein kleines Embedding der
  Dimension 16, indiziert über eine Hand-codierte Move-Klasse (siehe
  7.4).
- **Move-Head `MoveHead_phase`**: ein zweischichtiger MLP mit Hidden 32
  und Clipped-ReLU, der pro Kandidat einen Skalar-Logit ausgibt.
  Eingabe ist die Konkatenation aus Wurzel-Latent (stm-Sicht),
  Successor-Latent (other-Sicht) und Move-Type-Embedding.

```
logit(m) = MoveHead_phase( [ P_adapter(a_stm);
                             P_adapter(a_succ_other(m));
                             move_type_embed[type(m)] ] )
```

Das Schema ist asymmetrisch und folgt dem klassischen Negamax-Prinzip:
der Wurzel-Latent wird aus der eigenen Königsperspektive (stm) gelesen,
der Successor-Latent aus der gegnerischen Königsperspektive (other),
weil nach dem Zug die Seite gewechselt hat und dann der gegnerische
König die HalfKA-Indizierung definiert. Beide Accumulator existieren
ohnehin durch das Dual-Accumulator-Schema, sodass die Zuordnung
lediglich eine Slice-Auswahl ist.

### 7.3 Successor-Accumulator durch Delta-Update

```python
def successor_accumulators(a_white, a_black, FT_phase, m):
    leave_w, enter_w = halfka_delta(m, king='white')
    leave_b, enter_b = halfka_delta(m, king='black')
    if not is_king_move_white(m):
        a_white_succ = a_white
        for i in leave_w: a_white_succ -= FT_phase[i]
        for i in enter_w: a_white_succ += FT_phase[i]
    else:
        a_white_succ = full_rebuild(FT_phase, successor_features(m, 'white'))
    if not is_king_move_black(m):
        a_black_succ = a_black
        for i in leave_b: a_black_succ -= FT_phase[i]
        for i in enter_b: a_black_succ += FT_phase[i]
    else:
        a_black_succ = full_rebuild(FT_phase, successor_features(m, 'black'))
    return a_white_succ, a_black_succ
```

Die Update-Kosten pro Kandidat sind dominiert von der Zahl wechselnder
Features (typisch 4 bis 6 Indizes über beide Accumulator zusammen). Bei
den meisten Zügen ist das eine konstante Anzahl von
FT-Zeilenadditionen.

### 7.4 Move-Type-Embedding

Das Delta-Schema kodiert vollständig, welche Felder sich ändern, aber
nicht direkt schachliche Move-Klassen wie "Schlagzug" oder "Promotion".
Diese werden über ein kleines diskretes Embedding nachgereicht. Die
Move-Klasse wird aus Ziehender Figur (6), Schlagopfer (6), Promotion (5)
und Sondertyp (4) gebildet, in der Praxis sind viele Kombinationen
unmöglich. Eine kompakte Hand-Hash-Funktion bildet auf maximal 128
Indizes ab, aus denen ein 16-dimensionales Embedding gelesen wird.
Parameter pro Phase: 128 · 16 = 2048, vernachlässigbar.

### 7.5 Verhältnis zur reinen Wert-Politik-Unifikation

Eine noch radikalere Variante wäre die direkte Unifikation:
`logit(m) = −Value_phase(successor)`. Diese Variante ist semantisch
noch sauberer, weil sie Value und Policy strukturell zwingend
konsistent macht. LAPv2 wählt dennoch den eigenständigen Policy-Head
mit eigenem P_adapter und MoveHead aus zwei Gründen: erstens lässt sich
die Migration aus dem bestehenden LAPv1-Policy-Head sanfter realisieren;
zweitens erlaubt der eigene Pfad, schachliche Move-Klassen über das
Move-Type-Embedding einzubringen. Die volle Unifikation bleibt als
spätere Vereinfachung explizit auf der Roadmap (siehe auch Anhang D des
Codex-Plans).

## 8. Opponent-Zweig: Shared Backbone mit Δ-Readout

Der OpponentHead wird durch einen leichten Readout auf dem geteilten
Backbone ersetzt. Sei `h(s)` das vom State-Embedder gelieferte Latent
der Wurzelstellung und `a` ein Kandidatenzug. Statt eines vollen
zweiten State-Embedder-Pass wird ein kleiner Delta-Operator verwendet:

```
h'(s, a) = Δ( h(s), move_embed(a) )
```

Auf `h'(s, a)` operieren drei Readouts: `reply_logits`, `pressure`,
`uncertainty`. Ihre Aggregation im Wrapper bleibt identisch zu LAPv1:

```
reply_signal = best_reply − 10 · pressure − 10 · uncertainty
```

Während des Trainings wird optional ein Distillations-Lehrer benutzt:
das Hauptnetz selbst, aufgerufen auf der echten Nachfolgestellung `s'`
aus Gegnersicht. Der Δ-Readout lernt diese Lehrersignale zu
approximieren. Zur Laufzeit entfällt der Lehrer.

## 9. Datenpfad und Trainingspipeline

### 9.1 Artefakterweiterung

Das LAPv1-Artefakt wird rückwärtskompatibel um folgende deterministisch
berechenbare Felder pro Sample ergänzt: `phase_index`, `king_sq_white`,
`king_sq_black`, `nnue_features_white`, `nnue_features_black`.
Zusätzlich werden für jedes Sample für die ersten K legalen Züge die
Move-Type-Klasse und die Delta-Indizes pro Königsperspektive
vorberechnet und abgelegt. Ein Loader, der die neuen Felder ignoriert,
funktioniert als LAPv1-Loader weiter.

### 9.2 Stage T1 — Fast Bootstrap

T1 wird unverändert als CPU-feasibler Bootstrap gefahren, jedoch bereits
mit allen LAPv2-Komponenten aktiv. Alle Phase-Experten starten identisch
initialisiert. Die FT-Matrizen werden klein initialisiert
(`N(0, 1/√N)`) und lernen ihre Piece-Square-Salienzen aus den vereinten
Value- und Policy-Targets.

### 9.3 Stage T2 — Warm-Start mit Curriculum (präzisiert in 3.1)

T2 übernimmt das Inner-Step-Curriculum 1 → 2 → 4 und die
Per-Step-Supervision. **Der Warm-Start-Checkpoint für LAPv2 ist nicht
der "letzte" LAPv1-T2-Checkpoint, sondern der T2-Checkpoint mit der
höchsten `val_top1` auf der joint-Validation aus dem
`freeze_inner_hard`-Regime.** Im konkreten v4-Lauf ist das der
epoch-2-Checkpoint (val_top1 ≈ 0,9076), nicht der spätere
`joint_heads_mix`- oder `joint_backbone_mix`-Checkpoint. Siehe
Abschnitt 15 für die Begründung.

Neu in T2 für LAPv2 sind drei Ergänzungen:

1. Ein **Load-Balancing-Regularisierer** über die Phasen.
2. Ein optionaler **Distillations-Loss** aus vollem Rekursionsaufruf auf
   den Opponent-Readout.
3. Ein **zweistufiges Phase-Curriculum-Gate**. In der ersten
   Gate-Stufe wird nur die Phase-MoE in State-Embedder und
   Intention-Encoder aktiviert, während die NNUE-Heads noch als
   phasenunabhängige geteilte Köpfe trainiert werden. In der zweiten
   Stufe werden auch die NNUE-Heads in ihre Phase-Experten aufgeteilt.

### 9.4 Loss-Balance Value/Policy

Weil Value und Policy denselben FT teilen, müssen ihre Gradienten
balanciert sein, damit kein Head den FT in seine eigene Richtung
dominiert. T2 verwendet eine adaptive Gewichtung der beiden
Loss-Komponenten, in der der Value-Loss auf einem laufenden Mittel um
1.0 normiert wird und der Policy-Loss separat. Optional wird ein
kleiner Cosine-Penalty zwischen `V_adapter` und `P_adapter` genutzt.

### 9.5 (neu) Regression-Watchdog am Phasenübergang

Aus dem v4-Befund (Abschnitt 15) folgt eine harte Trainingsregel für
LAPv2: an jedem Übergang zwischen Stage-2-Phasen (`freeze_inner_hard` →
`joint_heads_mix` → `joint_backbone_mix`) und zusätzlich an jedem
Flag-Flip (`phase_moe`, `nnue_value`, `nnue_policy`, …) wird
**unmittelbar vor und nach dem Übergang** die joint-Validation gefahren
und `val_top1_after ≥ val_top1_before − ε` geprüft (Default
ε = 0,01 = 1 pp). Wird das Kriterium verletzt, wird der Übergang
automatisch verworfen und der vorherige Checkpoint wiederhergestellt.
Alternativ kann die Lernrate des gerade neu freigegebenen Parameter-
blocks um Faktor 0,3 reduziert und der Übergang wiederholt werden. Der
Watchdog loggt jeden Treffer explizit.

## 10. Deliberation-Loop unter LAPv2

Die Loop bleibt strukturell identisch. Das Phase-Routing wird an der
Wurzelstellung einmalig fixiert und über die Loop konstant gehalten.
Beide Accumulator (white, black) werden an der Wurzel einmal
vollständig berechnet und dann pro Top-k-Kandidat differenziell
fortgeschrieben. Der Successor-Accumulator wird vom Policy-Head zur
Move-Bewertung genutzt; derselbe Accumulator wird, falls der Kandidat
in die Refinement-Top-k-Selektion gerät, vom Value-Head erneut
konsumiert, ohne zweite Berechnung. Diese Cache-Lokalität ist einer
der Hauptgewinne der NNUE-Policy: Move-Scoring und Successor-
Value-Bewertung teilen sich den Accumulator-Pfad.

## 11. Parameter- und Laufzeitbudget

Gegenüber Rev 2 (~70,9 M) sinkt das Gesamtbudget um etwa 12 M auf
ca. 58 M. Der Grund ist ausschließlich das FT-Sharing zwischen Value
und Policy: ein einziger FT-Block pro Phase trägt nun beide Heads,
während in Rev 2 nur der Value-Head NNUE-artig war und der Policy-Head
als vollwertige Phase-MoE 12,4 M Parameter belegte. Die FLOPs pro
Forward-Pass wachsen nur moderat: hartes Routing aktiviert pro Modul
nur einen Experten, der NNUE-Pfad ist nach dem initialen
Accumulator-Aufbau inkrementell, und Policy-Scoring teilt sich den
Accumulator-Pfad mit der Value-Bewertung. Der dominante CPU-Mehraufwand
pro Deliberation-Schritt liegt im Δ-Operator des Opponent-Readouts und
wird auf 15 bis 25 Prozent gegenüber LAPv1 geschätzt.

## 12. Migrationsplan (Grobstruktur)

LAPv2 wird inkrementell eingeführt, um Regressionsrisiken zu minimieren
und Arenavergleichbarkeit zu erhalten. Jeder Schritt erzeugt einen
gültigen, rückfallsicheren Checkpoint. Die vollständige
Implementierungsanleitung für Codex-CLI befindet sich in der
begleitenden Datei `LAPv2_Codex_Plan.md` (Rev 3.1).

1. LAPv1-Artefakt um `phase_index`, `king_sq`, NNUE-Features und
   Move-Deltas rückwärtskompatibel erweitern.
2. Phase-MoE-Wrapper und harten Router in Intention-Encoder und
   State-Embedder einziehen.
3. Dual-Accumulator-Infrastruktur und FT-Block einführen, zunächst nur
   als Datenpfad.
4. NNUE-Value-Head auf dem geteilten FT, eine Phase, dann als
   Phase-MoE.
5. NNUE-Policy-Head auf demselben FT, mit Move-Delta-Pfad und
   Move-Type-Embedding.
6. Sharpness-Head als Phase-MoE.
7. OpponentHead durch Δ-Operator und Shared-Backbone-Readouts ersetzen,
   optionalen Distillations-Loss integrieren.
8. Arenavergleich LAPv1 gegen LAPv2 über bestehende Thor-150-
   Eröffnungssuiten.

## 13. Erwartete Effekte und Risiken

### 13.1 Erwartete Effekte

- Strukturell konsistente Value- und Policy-Bewertung durch geteilten
  FT und intrinsische Königsfeldkonditionierung.
- Deutliche Parameterreduktion im Policy-Pfad (von ~12 M auf ~70 K),
  weil das teure FT geteilt wird.
- Cache-Lokalität in der Deliberation-Loop.
- Geringere Interferenz zwischen Stellungsregimes durch harte
  Phase-MoE mit vollwertigen Experten.
- Konsistentere Reply-Signale durch Shared-Backbone-Opponent-Readout.
- **Stärkere relative Gewinne auf der "hard"/selection-Teilmenge
  erwartet**, weil das 10-pp-Gap zwischen selection_top1 und val_top1
  aus dem v4-Lauf exakt das Regime kennzeichnet, in dem NNUE-
  Policy-Scoring strukturell helfen sollte (siehe Abschnitt 15).

### 13.2 Risiken und Gegenmaßnahmen

- Gradientenkonflikt zwischen Value und Policy am geteilten FT:
  begegnet durch adaptive Loss-Balance, separate Adapter und optionalen
  Cosine-Penalty.
- Move-Type-Embedding zu schwach: konfigurierbare Dimension,
  Tactical-Smoke.
- FT-Instabilität in frühen Epochen: zweistufiges
  Phase-Curriculum-Gate und schwache L2-Regularisierung.
- Accumulator-Rebuild bei Königszügen: Dirty-Flag und lazy Rebuild.
- Trainingsinstabilität durch gleichzeitige Einführung mehrerer Achsen:
  gestufter Migrationsplan mit abschaltbaren Schritten.
- **(neu) Wiederholung der Stage-2-Phasenübergangs-Regression**:
  begegnet durch den Regression-Watchdog aus 9.5 und durch den
  korrigierten Warm-Start-Checkpoint aus 9.3.

## 14. Fazit

LAPv2 macht Value und Policy zu zwei NNUE-artigen Lese-Sichten auf
einen einzigen, phasenabhängigen Feature-Transformer. Die
Königsfeldkonditionierung wird zu einem strukturellen, nicht gelernten
Routing; das Move-Scoring wird zu einem differenziellen
Successor-Accumulator-Aufruf. Die Phase-MoE mit vollwertigen Experten
reduziert Interferenz zwischen Stellungsregimes. Der Opponent-Zweig
wird durch einen geteilten Backbone mit leichtgewichtigem Δ-Readout
ersetzt. Das Gesamtbudget bleibt mit ~58 M Parametern moderat.

Die Kernthese bleibt: Schachspielstärke entsteht nicht aus bloßer
Laufzeit-Deliberation, sondern aus explizit trainierter,
architektonisch geführter Deliberation. LAPv2 ist der nächste
konsequente Schritt auf dieser Linie.

---

## 15. (neu) Analyse des Laufs `stage2_fast_arena_all_unique_v4`

Dieser Abschnitt wertet den letzten verfügbaren LAPv1-Stage-2-Lauf aus,
weil er die unmittelbare Referenz-Baseline für LAPv2 ist und einen
nicht-trivialen Befund liefert, der in die Trainingspipeline
zurückfließen muss.

### 15.1 Konfigurationsgerüst

```
stage                = T2
initial_checkpoint   = models/lapv1/stage1_fast_all_unique_v1/bundle/checkpoint.pt
epochs               = 5
batch_size           = 512
train_examples       = 150 000 (hard-mix aus 675 386)
validation_examples  =  15 000 (hard) bzw.  75 043 (joint)
data_access          = lazy_jsonl

stage2_phase         | epochs | trainable groups
---------------------+--------+-----------------------------------------
freeze_inner_hard    | 1..2   | inner_delta_network, inner_loop_core  (6,87 M trainierbare Parameter)
joint_heads_mix      | 3..4   | + root_heads                          (17,07 M)
joint_backbone_mix   | 5..5   | + root_backbone                       (20,05 M)
```

Die hard-Teilmenge stammt aus einem Curriculum-Filter mit
`candidate_count_cap=8`, `gap_cap_cp=128`, `value_cap_cp=256`, gemittelter
Score 5,23, gemittelter Gap 286,66 cp.

### 15.2 Gemessene Kernmetriken pro Epoch

| Epoch | Phase | min/max inner | train_loss | val_top1 | root_top1 | val_mrr | root_incorrect_gain | rollbacks |
|---|---|---|---|---|---|---|---|---|
| 1 | freeze_inner_hard | 1/2 | 1,9331 | 0,9045 | 0,9043 | 0,9454 | 0,0070 | 0 |
| 2 | freeze_inner_hard | 2/4 | 1,8509 | **0,9076** | 0,9043 | 0,9473 | **0,1177** | 0 |
| 3 | joint_heads_mix   | 2/2 | 2,0689 | 0,8060 | 0,8060 | 0,8879 | 0,0122 | 100 |
| 4 | joint_heads_mix   | 4/4 | 2,0456 | 0,8050 | 0,8074 | 0,8873 | 0,0163 | 6302 |
| 5 | joint_backbone_mix | —   | — | — | — | — | — | — (Log bricht im Training ab) |

Die `selection_top1`-Werte (separate Validation auf der harten
Teilmenge) liegen konsistent bei 0,7961 bzw. 0,7969 im
`freeze_inner_hard`-Regime und 0,8060 bzw. 0,8066 im
`joint_heads_mix`-Regime. Das joint-val_top1 und das selection_top1
**konvergieren** im `joint_heads_mix`-Regime, aber nicht weil selection
besser wird, sondern weil joint schlechter wird.

### 15.3 Drei sicherheitsrelevante Beobachtungen

**A. Phasenübergangs-Regression am Head-Unfreeze.** Der Sprung
epoch 2 → epoch 3 (`freeze_inner_hard` → `joint_heads_mix`) reduziert
`val_top1` um etwa 10 pp (0,9076 → 0,8060) und gleichzeitig bricht der
`root_incorrect_gain` — also der Anteil an Fällen, in denen die
Deliberation-Loop einen falschen Root-Pick korrigiert — um eine
Größenordnung ein (0,1177 → 0,0122). Die Rollback-Rate springt
gleichzeitig von 0 auf 100 (epoch 3) bzw. 6302 (epoch 4). Interpretation:
das gleichzeitige Unfreezen der Root-Heads und das Umstellen auf den
hard-Mix verschiebt den Arbeitspunkt so stark, dass die Inner-Loop ihre
vorher gelernte korrektive Kraft verliert. Das ist **kein** numerisches
Instabilitätsphänomen (keine NaN, Loss bleibt beschränkt), sondern ein
Regime-Switch.

**B. Gap zwischen selection_top1 und val_top1.** Im
`freeze_inner_hard`-Regime liegt `val_top1` bei 0,904 und
`selection_top1` bei 0,796 — ein Gap von rund 11 pp. Das ist exakt die
Teilmenge, in der taktische Differenzierung und präzises Move-Scoring
den Unterschied machen. Es ist damit auch die Teilmenge, in der
**NNUE-Policy strukturell den größten Gewinn erwarten lässt**, weil
NNUE-Policy genau das ist: präzises, königsfeld-konditioniertes
Move-Scoring auf der Basis differentieller Successor-Repräsentationen.

**C. Arena-Block fehlt.** Der Lauf heißt `stage2_fast_arena_…`, aber
das Log bricht innerhalb von epoch 5 (batch 480/1320) im Training ab,
bevor irgendeine Arena-Messung geschieht. Die
LAPv1-Arena-Referenzzahlen existieren damit im v4-Lauf nicht. Für LAPv2
bedeutet das: der allererste Schritt der Migration ist eine
reproduzierbare, vollständig gelaufene LAPv1-Baseline inklusive Arena —
ohne diese Zahlen ist kein LAPv2-Gewinn belegbar.

### 15.4 Konsequenzen für LAPv2

Daraus folgen drei konkrete Entscheidungen, die in Abschnitt 9 und im
begleitenden Codex-Plan verankert sind:

1. **Warm-Start-Checkpoint korrigiert.** LAPv2 warm-startet aus dem
   `freeze_inner_hard`-Checkpoint mit der höchsten joint-`val_top1`
   (im v4-Lauf: epoch 2, 0,9076), **nicht** aus dem späteren
   `joint_heads_mix`- oder `joint_backbone_mix`-Checkpoint. Andernfalls
   erbt LAPv2 schon 10 pp unterhalb der besten verfügbaren Baseline.
2. **Regression-Watchdog** (Abschnitt 9.5) wird an jedem
   Phasenübergang und an jedem LAPv2-Flag-Flip obligatorisch.
3. **Primärmetrik für den Release-Gate** ist nicht nur die Elo-Differenz
   auf Thor-150, sondern zusätzlich die **Differenz auf der
   selection/hard-Teilmenge** — genau dort erwarten wir den
   NNUE-strukturellen Gewinn, und genau dort hat die LAPv1-Baseline
   ihren dokumentierten Schwachpunkt.

Zusätzlich wird vor Beginn von Schritt 1 des Migrationsplans ein
kurzer LAPv1-Reproduktionslauf gefordert, der (a) den v4-Abbruch
behebt, (b) den Arena-Block nachliefert und (c) die drei Baseline-
Checkpoints (`freeze_inner_hard_best`, `joint_heads_mix_best`,
`joint_backbone_mix_best`) explizit speichert.
