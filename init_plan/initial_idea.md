was wäre, wenn man den move-generator in das netz integriert, also die eingabe nur noch die stellung ist und das netz intern selbst herausfindet, welche züge möglich sind und welche wahrscheinlicher als andere sind, oder WDL-Ausgabe - wie teuer wäre das? welche eingaben würden sich für tiefere netze ergeben. vielleicht sind diese eingaben der schlüssel für ein netz, dass auf cpu gut ausgeführt werden kann
____
Ja — **prinzipiell ist das möglich**, aber es verschiebt das Problem von „Züge explizit erzeugen“ zu „Legalität implizit approximieren“. Und genau da wird es teuer.

Der Kernpunkt ist:

**Ein Netz kann gut lernen, welche Züge plausibel sind.**
**Ein Netz ist deutlich schlechter darin, die Schachregeln exakt und billig zu emulieren.**

Das sind zwei verschiedene Aufgaben.

---

## 1. Was hieße „Move-Generator im Netz“ konkret?

Es gibt drei Stufen:

### A. Nur **Policy-Head**

Eingabe ist nur die Stellung, Ausgabe ist eine Verteilung über einen festen Zugraum, z. B. wie bei AlphaZero:

* 8x8 Ursprungsfelder
* pro Feld ~73 mögliche Aktionstypen
* insgesamt **4672 mögliche Outputs**

Dann sagt das Netz:
„Diese 20 Züge sind wahrscheinlich gut.“

Das ist machbar.

### B. Policy-Head **ohne externen Legalitäts-Maskierer**

Das Netz soll selbst lernen:

* welcher Zug legal ist
* welcher illegal ist
* welcher gut ist

Das ist viel härter, weil es nicht nur „gute Züge“ lernen muss, sondern auch exakt:

* Pins
* Schachgebote und nur zulässige Evasionen
* Rochaderechte
* en passant
* Promotions
* Slider-Blockaden
* Selbstschach vermeiden

Das ist nicht nur Statistik, sondern fast schon ein **diskretes Regelwerk**.

### C. Vollständig neurale Suche

Dann bräuchtest du eigentlich zusätzlich ein internes Modell von:

* „Was passiert nach dem Zug?“
* „Wie sieht die Nachfolgestellung aus?“
* „Welche Folgezüge entstehen dann?“

Dann bist du fast bei einem **gelernten Schach-Simulator**. Das wäre extrem teuer und in der Praxis für CPU-Engines kaum attraktiv.

---

## 2. Wie teuer wäre das?

## Der billige Teil: WDL

Eine **WDL-Ausgabe** ist fast gratis.

Ob das Netz am Ende ausgibt:

* 1 Skalar
* oder 3 Logits für W/D/L

macht fast keinen Unterschied.
Die Mehrkosten im letzten Layer sind klein.

## Der teure Teil: voller Move-Output

Sobald du statt eines Werts **tausende Zuglogits** ausgibst, wird der Output-Head teuer.

Beispiel:

* Hidden Layer: 256 Neuronen
* Policy-Head: 4672 Outputs

Dann hat allein der letzte Layer ungefähr:

* **256 x 4672 ≈ 1.2 Mio Gewichte**

Das ist noch nicht absurd groß.
Aber wenn du das **an jedem Suchknoten** ausführen willst, ist es auf CPU brutal.

### Größenordnung

Wenn pro Knoten grob 1–3 Mio MACs nur für die Policy anfallen, dann bist du sehr schnell **zwei bis drei Größenordnungen langsamer** als eine klassische, hochoptimierte CPU-Engine mit einfacher Move-Generation + NNUE-Style Eval.

Der Grund ist nicht nur die Rechenmenge, sondern auch:

* schlechtere Cache-Lokalität
* mehr Memory-Bandwidth
* weniger Inkrementalität
* breiter Output statt schmaler Eval

**WDL-only:** billig
**WDL + full policy an jedem Knoten:** teuer
**full policy + implizite Legalität:** noch teurer, weil du tiefere/reichere Repräsentationen brauchst

---

## 3. Warum ist Legalität so schwer für ein Netz?

Weil Legalität in Schach oft von **globalen, exakten Abhängigkeiten** lebt.

Beispiele:

* Ein Springerzug ist lokal.
* Ein Läuferzug hängt an einer ganzen Ray-Blockade.
* Ein Königszug hängt an gegnerischen Angriffsfeldern.
* Ein Zug kann pseudo-legal sein, aber wegen Pin illegal.
* en passant kann plötzlich eine Linie öffnen und dadurch illegal werden.

Ein Netz muss also intern Dinge rekonstruieren wie:

* Besetzungsstruktur
* Angriffsrelationen
* Ray-Casts
* Pins
* Checkmasken
* Blockmasken
* Sonderregeln

Das ist möglich, aber das Netz lernt dann im Grunde Teile eines **symbolischen Regelinterpreters** nach.
Und das ist fast nie die billigste Lösung auf CPU.

---

## 4. Der wahrscheinliche Sweet Spot

Ich glaube nicht, dass der Schlüssel ist:
**„den Move-Generator komplett wegzulernen“**

Sondern eher:

**„den symbolischen Teil minimal, billig und exakt halten — und nur das Ranking / die Auswahl / die WDL neural machen.“**

Also:

* Symbolisch:

  * Brettzustand
  * Legalitätsprüfung
  * Zuganwendung
  * vielleicht billige Attack-Maps
* Neural:

  * Priorisierung
  * Kandidatenauswahl
  * Value/WDL
  * Search-Control

Das ist fast sicher CPU-freundlicher.

---

## 5. Welche Eingaben wären für tiefere Netze interessant?

Wenn dein Ziel **CPU-effiziente tiefe Netze** ist, dann ist die spannendste Frage nicht „wie tief?“, sondern:

**Welche Inputs linearisieren das Problem so gut, dass das Netz klein bleiben kann?**

Da gibt es drei Klassen.

---

## 6. Rohinputs: einfach, aber teuer

### Klassische Board-Planes

Zum Beispiel:

* 12 Piece-Planes
* Side to move
* Rochaderechte
* en passant
* Halbzugzähler

Das ist sauber und allgemein.
Aber tiefe Conv-/Transformer-Netze auf diesen Rohinputs sind auf CPU meist zu teuer für volle Suchintegration.

Gut für:

* Forschung
* kleine MCTS-Setups
* Root-eval

Schlecht für:

* Millionen Knoten pro Sekunde

---

## 7. Sparse / piece-centric Inputs: viel interessanter für CPU

Das ist die Richtung, die für CPU am ehesten trägt.

### A. Stücklisten statt 8x8-Bilder

Eingabe pro Figur:

* Farbe
* Typ
* Feld
* evtl. Relativposition zum König
* ob gezogen / relevant für Rochade
* ob pinned-Kandidat
* lokale Mobilitätsschätzung

Das kann man als kleine Tabelle oder Embeddings kodieren.

Vorteil:

* viel weniger redundante Nullen
* bessere CPU-Nutzung
* näher an dem, was NNUE-artige Systeme gut können

### B. King-relative Features

Sehr stark für Schach:

* Felder relativ zum eigenen König
* Felder relativ zum gegnerischen König
* Figur-zu-König-Distanzen
* Angreifer in Königszonen
* offene Linien auf den König

Das komprimiert strategisch viel Signal.

### C. Ray-basierte Features

Für Sliding Pieces entscheidend:

* nächster Blocker auf jeder Ray
* X-Ray Beziehungen
* Kontaktlinien auf Könige / Damen / Türme
* freie bzw. halbfreie Linien

Das ist genau die Art von Input, die einem kleinen Netz viel Arbeit abnimmt.

---

## 8. Symbolische Hilfsinputs — wahrscheinlich der eigentliche Schlüssel

Wenn du wirklich ein CPU-gutes Netz willst, würde ich sehr stark über **vorgerechnete Hilfsmasken** nachdenken.

Nicht als kompletter Move-Generator, sondern als **strukturierende Eingabe**.

Zum Beispiel:

* Angriffsmap Weiß
* Angriffsmap Schwarz
* Checker-Maske
* Pin-Maske
* Discoverable-attack-Maske
* King-danger-Zonen
* Besetzungsbitboards nach Linie/Diagonale
* Bauernhebel / Passer / rückständige Bauern
* Mobility Counts pro Figur

Dann hat das Netz nicht mehr die Aufgabe, die Regeln komplett neu zu entdecken, sondern bekommt bereits die entscheidenden geometrischen Invarianten.

Das ist oft viel billiger als ein tieferes Netz.

---

## 9. Welche Inputs helfen speziell bei „Move-Generator im Netz“?

Wenn du das Netz legalitätsnah machen willst, wären diese Inputs besonders wertvoll:

### Für jede Figur:

* Typ, Farbe, Feld
* ist sie aktuell gepinnt?
* Pin-Richtung
* Anzahl pseudo-legaler Ziele
* greift den gegnerischen König an?
* verteidigt eigenen König?

### Global:

* Side to move
* eigene/generische Attack-Maps
* Checkstatus
* Double-check ja/nein
* Castling rights
* en passant square
* Occupancy
* Slider rays
* Blocker-Informationen
* Wiederholungs-/50-Züge-Kontext, falls WDL ernst gemeint ist

### Für Ausgabestruktur:

* mögliche Ursprungssquares als Kandidaten
* pro Kandidat ein Ziel-Embedding
* Promotion-Flags

Dann kann das Netz sagen:

1. **welche Figur sollte ziehen**
2. **wohin**
3. **mit welchem Spezialtyp**

Das ist oft billiger als ein flacher 4672er Output.

---

## 10. CPU-freundliche Architekturideen

## Idee 1: Faktorisierte Policy

Statt 4672 Züge direkt:

* Head 1: Wahrscheinlichstes **From-Square**
* Head 2: bedingt darauf **Move-Type / To-Square**
* Head 3: Promotion-Typ

Das reduziert Kosten und passt besser zur Schachstruktur.

## Idee 2: Piece-centric Scoring

Das Netz bewertet erst jede Figur:

* „welche Figur ist zugreif?“
* danach nur deren Kandidatenzüge

Das kann sehr billig sein, wenn du pro Figur nur kleine MLPs nutzt.

## Idee 3: Symbolischer Kandidatengenerator + neuraler Reranker

Das ist wahrscheinlich der beste CPU-Kompromiss:

* klassisch alle legalen Züge erzeugen
* pro Zug ein kleines Zug-Feature bauen
* kleines Netz rankt Kandidaten
* Wertnetz liefert WDL

Damit bleibt Legalität exakt, und das Netz sieht trotzdem direkt die Zugstruktur.

## Idee 4: Conditional Computation

Nur für Top-K Kandidaten rechnest du einen teureren Zug-Head.
Der Rest bleibt billig.

---

## 11. Was ich für unwahrscheinlich halte

Dass ein CPU-starkes Schachnetz dadurch entsteht, dass man sagt:

**„Wir geben nur das Brett rein und lassen das Netz intern komplett selbst herausfinden, was legal ist.“**

Das klingt elegant, ist aber wahrscheinlich ineffizient.

Warum?

Weil du damit Rechenzeit auf etwas verschwendest, das ein paar Bitboard-Operationen exakt und fast kostenlos lösen.

Das Netz sollte besser seine Kapazität für Dinge verwenden, die symbolische Methoden schlechter können:

* langfristige Kompensation
* Initiative
* Königssicherheit als Kontinuum
* unklare Opfer
* Drawishness / Fortress-Tendenzen
* Move ordering
* Search selectivity

---

## 12. Meine eigentliche Hypothese

Der „Schlüssel“ für eine CPU-gute Architektur ist sehr wahrscheinlich:

**nicht rohe Bretter + tiefes Netz**,
sondern

**stark strukturierte, sparse, regelnahe Inputs + kleines bis mittleres Netz**.

Also ungefähr:

* symbolischer Frontend-Teil
* sehr gute Input-Features
* kleiner Policy-Head
* WDL-Head
* vielleicht inkrementelle Updates

Das ist eher NNUE-Philosophie erweitert um Policy, nicht AlphaZero auf CPU.

---

## 13. Ein konkretes Experiment, das ich sofort bauen würde

### Variante A

* Input:

  * piece lists
  * occupancy
  * attack maps
  * pin/check masks
  * castling/en-passant
* Heads:

  * WDL
  * From-square
  * To-square conditional
* Legalität:

  * weiterhin symbolisch verifiziert

Dann misst du:

* Elo
* NPS
* Cache-Verhalten
* Trefferquote bei Top-1/Top-3 Move Ordering
* Sucheffizienz

### Variante B

Dasselbe, aber **ohne Attack-/Pin-Hilfsinputs**.

Wenn A deutlich besser ist, weißt du:
Nicht Netz-Tiefe war der Hebel, sondern **die richtige Zwischenrepräsentation**.

---

## 14. Die knappe Antwort auf deine Hauptfrage

**Ja, ein Netz kann den Move-Generator teilweise internalisieren.**
**Komplett zu ersetzen ist wahrscheinlich auf CPU unökonomisch.**

* **WDL-Ausgabe:** billig
* **Policy über alle Züge:** machbar, aber teuer an jedem Suchknoten
* **Legalität implizit lernen:** noch teurer und riskanter
* **CPU-Schlüssel:** sparse, strukturierte, regelnahe Eingaben statt roher Boards und riesiger tiefer Netze

Mein Bauchgefühl wäre:

> Der beste CPU-Pfad ist nicht „neuraler Move-Generator“, sondern
> **„symbolisch exakte Legalität + neuronale Priorisierung auf sehr guten Inputs“**.

Ich kann dir daraus direkt als Nächstes ein **konkretes Architekturdiagramm für so eine Engine** machen — mit Input-Tensoren, Heads und grober Kostenabschätzung pro Knoten.
