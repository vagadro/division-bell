# Scale and concurrency

Times in ms, wall-clock from Python to FalkorDB in local Docker and back. The graphs are templated (only their size and shape matter here). Blast radius gets slower as the whole graph grows, even for a small affected set: the query plan starts from every agent's items instead of the one wrong fact, so cost follows graph size. Fine at demo scale; not a constant-time lookup.

## Graph size

Blast radius and retract of a company's headcount fact, on random companies.

| Items in graph | Queries | Load time | Blast radius median / p95 / max | Retract median / p95 / max | Exact answers |
|---|---|---|---|---|---|
| 452 | 30 | 0 s | 0.9 / 1.5 / 1.9 | 2.0 / 3.1 / 5.8 | 30/30 |
| 4,695 | 100 | 5 s | 2.8 / 3.1 / 4.2 | 3.3 / 3.8 / 4.1 | 100/100 |
| 46,536 | 100 | 153 s | 16.4 / 16.7 / 20.5 | 14.6 / 16.0 / 16.2 | 100/100 |

## Shapes

| Shape | Size | Items below the root | Blast radius ms | Found | Retract ms | Flagged |
|---|---|---|---|---|---|---|
| chain | 10 | 10 | 0.7 | 10 | 0.6 | 10 |
| chain | 50 | 50 | 1.8 | 50 | 0.9 | 50 |
| chain | 200 | 200 | 10.7 | 200 | 2.0 | 200 |
| fan-out | 1000 | 1000 | 21.4 | 1000 | 6.9 | 1000 |
| diamond lattice | 5 | 10 | 0.7 | 10 | 0.6 | 10 |
| diamond lattice | 10 | 20 | 4.1 | 20 | 1.1 | 20 |
| diamond lattice | 15 | 30 | 110.6 | 30 | 16.3 | 30 |
| diamond lattice | 20 | 40 | 3434.0 | 40 | 453.2 | 40 |

Size: chain = links, fan-out = items on the root, lattice = layers (2 items each, 2^layers paths).

## Concurrency

20 threads, each round: one retract per company, plus a decision built on that company's headcount chain (racing the retract) and one built on its funding fact (never retracted).

| Round | Companies | Retracts ok | Mistakes counted | Safe writes ok | Racing writes accepted | Active items built on a retracted fact |
|---|---|---|---|---|---|---|
| 1 | 30 | 30/30 | 30 | 30/30 | 13/30 | 0 |
| 2 | 30 | 30/30 | 30 | 30/30 | 16/30 | 0 |
| 3 | 30 | 30/30 | 30 | 30/30 | 15/30 | 0 |
| 4 | 30 | 30/30 | 30 | 30/30 | 18/30 | 0 |
| 5 | 30 | 30/30 | 30 | 30/30 | 16/30 | 0 |

A racing write is accepted if it landed before the retract (then the retract flags it) and refused after. The last column must be 0.
