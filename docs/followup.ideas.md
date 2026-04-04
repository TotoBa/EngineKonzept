Ich habe mir `TotoBa/EngineKonzept` und passende Forschungsrichtungen dazu angesehen. Mein Eindruck: Das Repo ist schon jetzt nicht auf „klassische Engine + bisschen NN“ angelegt, sondern sehr klar auf einen modularen *latent adversarial planning*-Stack. Der vorgesehene Laufweg ist bereits `position -> encoder -> legality/policy proposer -> latent dynamics -> opponent module -> recurrent planner -> WDL + move selection -> UCI output`, und die Projektregeln schließen klassische Runtime-Mechaniken wie Alpha-Beta, Quiescence, TT, LMR oder eine handgebaute statische Fallback-Eval explizit aus. Aktuell ist Phase 5 realisiert: exakte symbolische Regeln, faktorisiertes Action-Schema, deterministischer object-centric Encoder, Rust-Orakel für Labels und ein absichtlich einfacher Phase-5-Proposer. Phasen 6–8 sind schon als Dynamics-, Opponent- und Recurrent-Planner-Phasen mit Unsicherheitssteuerung vorgezeichnet. ([GitHub][1])

Die wichtigste Konsequenz daraus ist: Der eigentliche Freiheitsgrad des Projekts liegt nicht in „MLP vs. Transformer“, sondern in der *Verschaltung mehrerer Teilnetze*. Genau dafür ist das Repo gut geeignet. Es hat bereits eine saubere Rust/Python-Grenze, einen exakten Rust-Daten-/Label-Orakelpfad und `torch.export`-Bundles mit Metadaten auf der Rust-Seite. Das ist fast ideal, um mehrere kleinere Netze oder Teilmodule zu orchestrieren, statt früh ein einziges monolithisches Modell zu erzwingen. ([GitHub][2])

Der erste große Hebel ist aus meiner Sicht schon in Phase 5 sichtbar: Das Repo benutzt bereits ein *faktorisiertes* Move-Schema (`from` 64, `to` 64, `promotion` 5), aber der aktuelle Proposer decodiert trotzdem mit zwei flachen `20480`-Heads. Die Doku sagt selbst, dass der meiste Parameteraufwand in genau diesen Output-Heads sitzt. Das schreit geradezu nach einem *faktorisierten Decoder* oder Pointer-artigen Proposer: zuerst Quelle, dann Ziel bedingt auf Quelle, dann Promotion bedingt auf Quelle+Ziel. Das passt viel besser zur vorhandenen Repräsentation und dürfte generalisierbarer sein als auf Dauer zwei riesige flache Köpfe über allen `64*64*5` Aktionen. ([GitHub][3])

Mein stärkster Vorschlag wäre deshalb eine modulare Architektur, die ich sinngemäß so bauen würde:

```text
position
 -> exact symbolic pre-pass + regime flags
 -> structured encoder E
 -> router R
 -> proposer P
 -> top-k candidate moves
 -> dynamics G (for imagined moves)
 -> opponent model O
 -> recurrent planner H with memory
 -> arbiter A
 -> exact legality/end safety check
 -> UCI
```

Der Clou ist: *symbolische Buchstützen, neuraler Mittelteil*. Also am Anfang und Ende bewusst handcodierte, exakte Gates; dazwischen aber kein klassischer Search-Fallback, sondern ein Verband gekoppelter Netze.

**1. Encoder `E`: object-relational statt flach.**
Das Repo liefert schon jetzt `piece_tokens`, `square_tokens` und `rule_token`. Das ist eine fast perfekte Vorlage für einen zweistromigen Encoder: ein Stück-/Objektstrom für die bis zu 32 Figuren, ein Quadratstrom für die 64 Felder, plus ein globaler Regel-Slot. Ich würde die beiden Ströme mit Relational-/Graph-Attention oder einem kleinen Graph-Transformer koppeln und als Ausgaben sowohl einen globalen Root-Latent `z_root` als auch Figuren-/Feld-Latents behalten. Das ist viel näher an der Schachstruktur als ein flach gepackter 230er Vektor. Forschungseitig passt das sehr gut: Slot Attention argumentiert für austauschbare objektzentrierte Slots; Battaglia et al. argumentieren generell für relationale inductive biases; TreeQN/VPN/TD-MPC2/UniZero zeigen, dass Planen in abstrakten oder latenten Zuständen tragfähig sein kann. ([GitHub][3])

**2. Proposer `P`: faktorisiert, nicht flach.**
Hier würde ich nicht nur die Policy, sondern auch die *Legality-Struktur* faktorisiert modellieren: etwa `p(from|s)`, `p(to|from,s)`, `p(prom|from,to,s)` plus eine kleine Konsistenz-/Unsicherheitskomponente. Das kann seriell laufen oder als Fork-Join: ein Kopf für Quelle, einer für Ziel, einer für Promotion, einer für Unsicherheit, einer für einen „tactical urgency“-Score. Die kombinierten Ausgaben werden dann zur Planner-Eingabe. Das Schöne: Die gesamte bestehende Phase-5-Datenpipeline bleibt fast unangetastet, weil die Labels schon aus dem faktorisierten Action-Schema kommen. ([GitHub][3])

**3. Dynamics `G`: lokaler Updater statt globaler Rekonstrukteur.**
Für Phase 6 würde ich *kein* Netz bauen, das bei jedem Schritt diffus das ganze Brett neu „halluziniert“. Das Repo will `z = E(s)`, `z' = G(z,a)`, One-Step-Reconstruction, Drift-Messung und separate Behandlung von Spezialzügen. Das spricht eher für einen *lokalen* Dynamics-Block: der Zug-Embedding `a` triggert Updates an den betroffenen Figuren-Slots, den betroffenen Feldern und am Regel-Slot. Spezialfälle wie Rochade, En-passant und Umwandlung würde ich sogar über kleine Mikro-Experten oder dedizierte Übergangsköpfe behandeln, weil die Phase-6-Kriterien diese Fälle explizit getrennt sehen wollen. Für längere interne Rollouts würde ich dann hybrid werden: Schritt 1 eher exakt rekonstruktiv, spätere innere Deliberation eher *value-equivalent* oder abstrakt, um Drift und Kosten zu begrenzen. Genau diese Richtung steckt schon in VPN, TreeQN und moderneren Weltmodellen wie TD-MPC2 und UniZero. ([GitHub][4])

**4. Opponent-Modul `O`: unbedingt explizit, nicht implizit mitschleifen.**
Ich halte das Opponent-Netz für einen der spannendsten Teile des Repos. Es sollte nicht nur „den wahrscheinlichsten Reply“ ausspucken, sondern mehrere Dinge: eine Reply-Verteilung, einen Threat-/Pressure-Vektor, vielleicht einen Draw-/Fortress-Score und eine eigene Unsicherheit. In der MARL-Literatur ist das sehr plausibel: Es gibt Arbeiten zu explizitem Opponent Modeling, sogar mit Mixture-of-Experts-Struktur, und PR2 zeigt den Nutzen davon, wie der Gegner auf *dein* zukünftiges Verhalten reagiert. Für EngineKonzept wäre das fast genau die natürliche Interpretation von Phase 7. ([Proceedings of Machine Learning Research][5])

**5. Planner `H`: bounded recurrent deliberation statt Suchbaum.**
Der Planner sollte dann nicht zu einem versteckten Alpha-Beta-Ersatz verkommen, sondern zu einem festen, begrenzten inneren Deliberationsprozess. Mein Bild wäre: `H` bekommt `z_root`, die Top-k-Vorschläge aus `P`, die imaginierten Folge-Latents aus `G`, die Reply-Signale aus `O` und eigene Memory-Slots. Dann macht `H` 1 bis T interne Schritte und verbessert Kandidatenscores, WDL und Unsicherheit. Dass Phase 8 explizit Memory-Slots und uncertainty-guided compute allocation will, passt hervorragend dazu. Als Kern würde ich entweder einen kleinen Planner-Transformer oder einen Mamba-/SSM-artigen Planner erwägen; Mamba-2 wurde als 2–8x schnellerer Kern bei konkurrenzfähiger Leistung beschrieben, was für eine bounded recurrent Runtime attraktiv ist. ([GitHub][4])

Bis hierhin ist das schon eine saubere serielle Kette. Aber dein Gedanke mit *verknüpften Experten* ist noch spannender, und dafür gibt es inzwischen echte Anknüpfungspunkte.

Experten an sich sind keine wilde Fantasie mehr. Switch Transformers zeigen, dass Sparse-MoE große Kapazität bei konstantem Compute pro Sample ermöglichen; Expert Choice Routing geht noch weiter und erlaubt eine variable Zahl aktiver Experten pro Eingabe. Im Schach selbst gibt es bereits Arbeiten zu phasenspezifischen Experten in Kombination mit MCTS (M2CTS), und 2026 taucht sogar eine style-geroutete Chess-MoE-Arbeit auf (*Mixture-of-Masters*). Das heißt: *Experten* sind bereits plausibel. Neu wäre hier eher die Kombination aus Experten *innerhalb eines suchfreien latenten Gegenspieler-Planers* statt in MCTS oder reinen Chess-LMs. ([Journal of Machine Learning Research][6])

Daraus würde ich eine **Sparse Expert Federation** bauen. Der Router `R` bekommt `z_root` plus wenige exakte Regime-Merkmale aus dem symbolischen Start-Gate: in check, Promotionsnähe, Wiederholungsrisiko, Materialphase, vielleicht „quiet vs. forcing“. Dann aktiviert er unterschiedliche Experten für unterschiedliche Module. Zum Beispiel:

* Proposer-Experten: *tactical forcing*, *quiet positional*, *endgame/draw*, *promotion race*.
* Dynamics-Experten: *normal move update*, *castling*, *en passant*, *promotion*.
* Opponent-Experten: *best-reply tactical*, *best-reply positional*, *draw-holding / fortress*.
* Planner-Experten: *optimistic*, *pessimistic*, *tactical*, *strategic*.

Wichtig ist dabei: Nicht jeder Experte muss in jedem Modul aktiv sein. Genau da wird es interessant. Die Ausgabe des Routers kann *fan-out* machen: dieselbe Zustandsbeschreibung wählt andere Experten für den Proposer als für den Planner. Und die kombinierten Expert-Ausgaben werden dann in einem Arbiter zusammengeführt.

Noch interessanter finde ich eine zweite, weniger offensichtliche Linie: **Optionen nicht als Makrozüge, sondern als Makro-*Denkmodi***. In der hierarchischen RL gibt es die Idee einer Policy über Optionen mit gelerntem Ende/Abbruch. Für EngineKonzept würde ich das aber nicht auf Brettaktionen mappen, sondern auf Deliberationsprogramme: „taktische Widerlegung“, „ruhige Verbesserung“, „Abwicklung zum Remis“, „Promotion-Rennen“, „Königssicherheit zuerst“. Ein kleiner Controller wählt also nicht *welchen Zug* du spielst, sondern *welcher Planungsmodus* 1 bis T innere Schritte übernehmen darf; ein Termination-Head entscheidet, ob der Modus weiterläuft oder an einen anderen übergibt. Genau das passt verblüffend gut zu deinem Wunsch nach handcodierten Abzweigungen *zwischen* den Netzblöcken. Ich habe in meiner Suche keine direkte Schach-Engine gefunden, die so einen suchfreien „Option-over-deliberation“-Ansatz explizit baut; das wirkt auf mich klar untererforscht. ([cir.nii.ac.jp][7])

Die aus meiner Sicht kühnste, aber sehr reizvolle Variante wäre eine **Dual-Mind-/Debate-Architektur**. Also: ein schnelles Intuitionsnetz liefert sofort einen Prior und ein grobes WDL; dann laufen nur bei unsicheren oder widersprüchlichen Kandidaten weitere Spezialisten los, etwa ein taktischer Kritiker, ein pessimistischer Kritiker, ein positioneller Kritiker. Ein Arbiter fusioniert das. Die *Uneinigkeit* dieser Köpfe wird direkt zum Unsicherheitssignal und damit zum Compute-Regler für Phase 8. Das ist nicht aus der Luft gegriffen: Neuere Weltmodell-Arbeiten kombinieren bereits „schnelle“ statistische Dynamik mit einem langsameren logisch-strukturierten Korrektursystem, um Langhorizont-Imagination stabiler zu machen. Für dieses Repo ist das besonders passend, weil der symbolische Kern ohnehin als exakte Randinstanz bestehen bleibt. ([OpenReview][8])

Eine weitere Wildcard, die ich ernsthaft prüfen würde, ist **Backward Latent Planning**. Also nicht nur vorwärts `G(z,a)`, sondern ein zweites Netz, das von erwünschten Endmustern rückwärts Subgoals im Latentraum vorschlägt: Mattnetz, stabile Remisfestung, Damenumwandlung, Königssicherheits-Reparatur. Vorwärts- und Rückwärtsmodell treffen sich dann in der Mitte. In der Robotik gibt es frische Arbeiten zu latentem Backward Planning; auf Schach und speziell auf so einen modulierten, suchfreien Pipeline-Stack habe ich dazu in meiner Suche nichts Direktes gefunden. Das riecht nach echter Neuheit. ([OpenReview][9])

Trotz all dieser Modularität würde ich *immer* eine monolithische Kontrolllinie mitlaufen lassen: einen großen suchfreien Transformer als Baseline. 2024 wurde gezeigt, dass ein hinreichend großer, auf vielen Partien und Engine-Labels trainierter Transformer erstaunlich starkes Schach *ohne explizite Suche* spielen kann. Ich würde ihn nicht als Zielarchitektur nehmen, weil er schlechter zu eurem Phasenplan, zu Opponent Modeling und zu expliziter Deliberation passt. Aber als Kontrollbaseline ist er Gold wert: Dann wisst ihr jederzeit, ob eure Modularität wirklich etwas bringt oder nur schöner aussieht. ([Hugging Face][10])

Unterm Strich würde ich für *EngineKonzept* deshalb diese Reihenfolge empfehlen:

1. **Sofort**: flachen Phase-5-Proposer durch einen faktorisierten Pointer-/Conditional-Proposer ersetzen.
2. **Dann**: object-relational Encoder + lokaler Dynamics-Updater mit Regel-Slot und Spezialzug-Köpfen.
3. **Dann**: explizites Opponent-Modul, das Reply-Verteilung und Threat-Signale liefert.
4. **Dann**: bounded recurrent Planner mit Memory-Slots und explizitem Unsicherheitskopf.
5. **Erst danach**: Sparse Experts / Debate / Options darüberlegen, sobald die dichte Basis funktioniert.
6. **Parallel**: monolithischen Searchless-Transformer als Benchmark mitschleppen.

Das ist aus meiner Sicht die Kombination aus größter Repo-Passung, höchster wissenschaftlicher Anschlussfähigkeit und bester Chance auf einen wirklich eigenständigen Workflow. Sie nutzt die vorhandene objektzentrierte Kodierung, das exakte Rust-Orakel und die publizierten Phasenpläne voll aus, ohne in klassische Suchmechanik zurückzurutschen. ([GitHub][1])

Der sinnvollste nächste Schritt wäre jetzt ein konkretes Modulschema mit Tensorformen, Export-Grenzen und Trainingszielen pro Netz.

[1]: https://github.com/TotoBa/EngineKonzept "https://github.com/TotoBa/EngineKonzept"
[2]: https://github.com/TotoBa/EngineKonzept/blob/main/docs/architecture/datasets.md "https://github.com/TotoBa/EngineKonzept/blob/main/docs/architecture/datasets.md"
[3]: https://github.com/TotoBa/EngineKonzept/blob/main/docs/architecture/encoding.md "https://github.com/TotoBa/EngineKonzept/blob/main/docs/architecture/encoding.md"
[4]: https://github.com/TotoBa/EngineKonzept/blob/main/PLANS.md "https://github.com/TotoBa/EngineKonzept/blob/main/PLANS.md"
[5]: https://proceedings.mlr.press/v48/he16.html "https://proceedings.mlr.press/v48/he16.html"
[6]: https://jmlr.org/papers/v23/21-0998.html "https://jmlr.org/papers/v23/21-0998.html"
[7]: https://cir.nii.ac.jp/crid/1360013267175317248 "https://cir.nii.ac.jp/crid/1360013267175317248"
[8]: https://openreview.net/forum?id=Bzlt5tPFT6 "https://openreview.net/forum?id=Bzlt5tPFT6"
[9]: https://openreview.net/forum?id=DJiouYdH19&noteId=U6FMGnas9Z "https://openreview.net/forum?id=DJiouYdH19&noteId=U6FMGnas9Z"
[10]: https://huggingface.co/papers/2402.04494 "https://huggingface.co/papers/2402.04494"
